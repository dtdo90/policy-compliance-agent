import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pandas as pd
from pyspark.sql import SparkSession, functions as F

from ..core.config import load_config
from ..core.paths import resolve_project_path
from . import joiner_hive as base
from . import vl_excel_hive as excel_hive


CONFIG = load_config()
METADATA_SETTINGS = CONFIG.get("metadata", {})
OUTPUT_SETTINGS = CONFIG.get("outputs", {})


SPARK_APP_NAME = "metadata_enrichment"
BASE_METADATA_INPUT_PATH = excel_hive.METADATA_OUTPUT_PATH
MANUAL_MATCHES_XLSX_PATH = resolve_project_path(
    METADATA_SETTINGS.get("manual_matches_xlsx_path", "unmatched_vl_excel_metadata.xlsx")
)
METADATA_OUTPUT_PATH = resolve_project_path(
    OUTPUT_SETTINGS.get("metadata_enriched_output_path", "data/results/merged_client_metadata_with_enrichment.json")
)
DETAILS_OUTPUT_PATH = resolve_project_path(
    OUTPUT_SETTINGS.get("metadata_enrichment_details_path", "data/results/manual_match_link_enriched.csv")
)
MANUAL_MATCHES: List[tuple[str, list[str]]] = []
MANUAL_MATCHES_XLXS_PATH = MANUAL_MATCHES_XLSX_PATH


def _find_column(columns, keywords: List[str]) -> Optional[str]:
    for column_name in columns:
        column_text = str(column_name).strip().lower()
        if all(keyword.lower() in column_text for keyword in keywords):
            return column_name
    return None


def _distinct_non_empty(values: List[Any]) -> List[Any]:
    out: List[Any] = []
    seen = set()
    for value in values:
        if value in (None, "", [], {}):
            continue
        key = json.dumps(value, sort_keys=True, ensure_ascii=True) if isinstance(value, (dict, list)) else str(value)
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


def _singleton_or_none(values: List[Any]) -> Optional[Any]:
    values = _distinct_non_empty(values)
    return values[0] if len(values) == 1 else None


def _parse_normalized_timestamp(value: Any) -> Optional[pd.Timestamp]:
    if value in (None, ""):
        return None
    parsed = pd.to_datetime(value, errors="coerce", dayfirst=True)
    if pd.isna(parsed):
        return None
    return parsed.normalize()


def _flatten_ptcc_descriptions(value: Any) -> List[str]:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        return _distinct_non_empty([str(item).strip() for item in value if str(item).strip()])
    text = str(value).strip()
    return [text] if text else []


def _resolve_manual_matches_path() -> Path:
    if MANUAL_MATCHES_XLSX_PATH.exists():
        return MANUAL_MATCHES_XLSX_PATH
    if MANUAL_MATCHES_XLXS_PATH.exists():
        return MANUAL_MATCHES_XLXS_PATH
    raise FileNotFoundError(
        f"Manual match workbook not found: {MANUAL_MATCHES_XLSX_PATH} or {MANUAL_MATCHES_XLXS_PATH}"
    )


def _load_manual_matches() -> pd.DataFrame:
    global MANUAL_MATCHES

    input_path = _resolve_manual_matches_path()
    df_raw = pd.read_excel(input_path, dtype=str)
    transcript_col = "transcript_id" if "transcript_id" in df_raw.columns else _find_column(df_raw.columns, ["transcript", "id"])
    op_code_col = next((column_name for column_name in df_raw.columns if "Op Code" in str(column_name)), None)
    if transcript_col is None or op_code_col is None:
        raise ValueError(
            "Manual match workbook must contain transcript_id and an op code column containing 'Op Code'."
        )

    scenario_columns = {
        str(scenario_id): column_name
        for column_name in df_raw.columns
        if str(column_name).strip().isdigit() and 1 <= int(str(column_name).strip()) <= 14
        for scenario_id in [int(str(column_name).strip())]
    }

    MANUAL_MATCHES = []
    aggregated: Dict[str, Dict[str, Any]] = {}
    for _, row in df_raw.iterrows():
        transcript_id = str(row.get(transcript_col, "")).strip()
        op_code_raw = row.get(op_code_col)
        if not transcript_id or transcript_id.lower() == "nan" or pd.isna(op_code_raw) or not str(op_code_raw).strip():
            continue

        op_code_list = _distinct_non_empty([code.strip() for code in str(op_code_raw).split(",") if code.strip()])
        if not op_code_list:
            continue

        MANUAL_MATCHES.append((transcript_id, op_code_list))
        entry = aggregated.setdefault(
            transcript_id,
            {
                "transcript_id": transcript_id,
                "manual_op_codes": [],
                "excel_cif_no": None,
                "excel_trade_date_raw": None,
                "excel_input_date_raw": None,
                "ground_truth": {str(scenario_id): None for scenario_id in range(1, 15)},
            },
        )
        entry["manual_op_codes"] = _distinct_non_empty(entry["manual_op_codes"] + op_code_list)

        excel_cif_no = excel_hive.normalize_digits(row.get(excel_hive.EXCEL_CIF_COL))
        if entry["excel_cif_no"] is None and excel_cif_no is not None:
            entry["excel_cif_no"] = excel_cif_no
        if entry["excel_trade_date_raw"] is None and pd.notna(row.get(excel_hive.TRADE_DT_COL)):
            entry["excel_trade_date_raw"] = str(row.get(excel_hive.TRADE_DT_COL)).strip()
        if entry["excel_input_date_raw"] is None and pd.notna(row.get(excel_hive.DT_COL)):
            entry["excel_input_date_raw"] = str(row.get(excel_hive.DT_COL)).strip()

        for scenario_id in range(1, 15):
            column_name = scenario_columns.get(str(scenario_id))
            if column_name is None:
                continue
            value = row.get(column_name)
            if entry["ground_truth"][str(scenario_id)] is None and pd.notna(value) and str(value).strip():
                entry["ground_truth"][str(scenario_id)] = str(value).strip()

    df_manual = pd.DataFrame(list(aggregated.values()))
    print(f"Loaded {len(df_manual)} manual transcript matches from: {input_path}")
    print(f"Collected {len(MANUAL_MATCHES)} transcript/op_code pairs into MANUAL_MATCHES")
    return df_manual


def _load_voice_log_rows(transcript_ids: Set[str]) -> pd.DataFrame:
    df_vl = excel_hive._load_voice_log_match_rows(transcript_ids)
    if df_vl.empty:
        return df_vl

    duplicate_ids = df_vl["transcript_id"][df_vl["transcript_id"].duplicated()].unique().tolist()
    if duplicate_ids:
        sample_ids = ", ".join(sorted(duplicate_ids)[:10])
        print(
            "Voice-log parsing produced duplicate transcript IDs for manual enrichment: "
            f"{len(duplicate_ids)} duplicates (examples: {sample_ids})"
        )

    df_vl = (
        df_vl.sort_values(["transcript_id", "source_mapping_file"])
        .drop_duplicates(subset=["transcript_id"], keep="first")
        .rename(columns={"call_date": "call_date_vl"})
    )
    return df_vl


def _build_match_input_rows(df_manual: pd.DataFrame, df_vl: pd.DataFrame) -> pd.DataFrame:
    df_manual = df_manual.merge(
        df_vl[
            [
                "transcript_id",
                "call_id",
                "call_date_vl",
                "phone_number_vl",
                "dialed_in_number",
                "direction",
                "source_mapping_file",
            ]
        ],
        on="transcript_id",
        how="left",
    )

    rows = []
    for _, row in df_manual.iterrows():
        for index, excel_op_code in enumerate(row["manual_op_codes"]):
            rows.append(
                {
                    "excel_row_id": f"{row['transcript_id']}::{index}",
                    "transcript_id": row["transcript_id"],
                    "call_id": row.get("call_id"),
                    "call_date_vl": row.get("call_date_vl"),
                    "phone_number_vl": row.get("phone_number_vl"),
                    "dialed_in_number": row.get("dialed_in_number"),
                    "direction": row.get("direction"),
                    "source_mapping_file": row.get("source_mapping_file"),
                    "excel_op_code": excel_op_code,
                    "excel_cif_no": row.get("excel_cif_no"),
                    "excel_trade_date_raw": row.get("excel_trade_date_raw"),
                    "excel_input_date_raw": row.get("excel_input_date_raw"),
                }
            )

    return pd.DataFrame(rows)


def _load_latest_contact_by_cif(spark: SparkSession, cif_values: List[str]) -> Dict[str, Dict[str, Any]]:
    if not cif_values:
        return {}

    cif_df = spark.createDataFrame(
        [{"main_cif_no": cif_value} for cif_value in cif_values],
        "main_cif_no string",
    )
    contact_df = base._load_contact_metadata(spark, cif_df)
    latest_by_cif: Dict[str, Dict[str, Any]] = {}
    for row in contact_df.collect():
        row_dict = row.asDict(recursive=True)
        cif_no = row_dict.get("contact_cif_number")
        if not cif_no:
            continue
        current = latest_by_cif.get(cif_no)
        if current is None or (
            row_dict.get("contact_date_key") is not None
            and (
                current.get("contact_date_key") is None
                or row_dict["contact_date_key"] > current["contact_date_key"]
            )
        ):
            latest_by_cif[cif_no] = row_dict
    return latest_by_cif


def _serialize_for_csv(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=True)
    return value


def load_metadata_with_enrichment() -> Dict[str, Dict[str, Any]]:
    metadata_lookup = base.load_client_metadata_from_file(input_path=BASE_METADATA_INPUT_PATH)
    manual_df = _load_manual_matches()
    if manual_df.empty:
        return metadata_lookup

    transcript_ids = {str(transcript_id).strip() for transcript_id in manual_df["transcript_id"] if str(transcript_id).strip()}
    df_vl = _load_voice_log_rows(transcript_ids)
    match_input_rows = _build_match_input_rows(manual_df, df_vl)
    if match_input_rows.empty:
        print("No manual transcript/op_code rows were available for enrichment.")
        return metadata_lookup

    spark = SparkSession.builder.appName(SPARK_APP_NAME).enableHiveSupport().getOrCreate()
    try:
        match_input_df = excel_hive._excel_matches_to_spark(spark, match_input_rows).withColumn(
            "excel_input_datetime_ts", base._timestamp_col("excel_input_date_raw")
        )
        txn_df = excel_hive._load_transaction_metadata_for_excel_rows(spark, match_input_df)
        enriched_df = match_input_df.join(F.broadcast(txn_df), on="excel_row_id", how="left")
        enriched_rows = [row.asDict(recursive=True) for row in enriched_df.collect()]

        description_df = base._load_description_metadata(
            spark, match_input_df.select(F.col("excel_op_code").alias("op_code")).dropna().dropDuplicates()
        )
        ptcc_by_op_code = {
            str(row["op_code"]).strip(): row["ptcc_description"]
            for row in description_df.collect()
            if row["op_code"] is not None and str(row["op_code"]).strip()
        }

        cif_values = _distinct_non_empty(
            manual_df["excel_cif_no"].tolist() + [row.get("main_cif_no") for row in enriched_rows]
        )
        latest_contact_by_cif = _load_latest_contact_by_cif(spark, cif_values)
    finally:
        spark.stop()

    transcript_to_rows: Dict[str, List[Dict[str, Any]]] = {}
    details_rows: List[Dict[str, Any]] = []
    for row in enriched_rows:
        row["ptcc_description"] = ptcc_by_op_code.get(str(row.get("excel_op_code") or "").strip()) or []
        transcript_to_rows.setdefault(str(row["transcript_id"]).strip(), []).append(row)
        details_rows.append(
            {
                key: _serialize_for_csv(value)
                for key, value in row.items()
            }
        )

    manual_by_transcript = {str(row["transcript_id"]).strip(): row for _, row in manual_df.iterrows()}
    enriched_count = 0
    for transcript_id, manual_row in manual_by_transcript.items():
        candidate_rows = transcript_to_rows.get(transcript_id, [])
        manual_op_codes = _distinct_non_empty(list(manual_row["manual_op_codes"]))

        transaction_candidates = []
        candidate_client_names: List[str] = []
        candidate_rm_names: List[str] = []
        candidate_main_cif_nos: List[str] = []
        candidate_trade_dates: List[str] = []
        candidate_ptcc_descriptions: List[str] = []

        for row in candidate_rows:
            transaction_candidates.append(
                {
                    "excel_op_code": row.get("excel_op_code"),
                    "op_code": row.get("op_code"),
                    "main_cif_no": row.get("main_cif_no"),
                    "trade_date_raw": row.get("trade_date_raw"),
                    "trade_datetime": row.get("trade_datetime"),
                    "input_date_raw": row.get("input_date_raw"),
                    "creation_date": row.get("creation_date"),
                    "order_expiry_date": row.get("order_expiry_date"),
                    "order_initiation": row.get("order_initiation"),
                    "portfolio_begin_date": row.get("portfolio_begin_date"),
                    "portfolio_end_date": row.get("portfolio_end_date"),
                    "portfolio_code": row.get("portfolio_code"),
                    "portfolio_status": row.get("portfolio_status"),
                    "instrument_ud_type": row.get("instrument_ud_type"),
                    "instrument_ud_sub_type": row.get("instrument_ud_sub_type"),
                    "instrument_code": row.get("instrument_code"),
                    "order_person_spoken_to": row.get("order_person_spoken_to"),
                    "rm_name": row.get("rm_name"),
                    "product_risk_profile": row.get("product_risk_profile"),
                    "product_risk_profile_num": row.get("product_risk_profile_num"),
                    "client_risk_profile": row.get("client_risk_profile"),
                    "client_risk_profile_num": row.get("client_risk_profile_num"),
                    "ptcc_description": row.get("ptcc_description"),
                    "txn_match_count": row.get("txn_match_count"),
                    "txn_candidate_count": row.get("txn_candidate_count"),
                    "txn_candidate_details": row.get("txn_candidate_details"),
                }
            )
            if row.get("order_person_spoken_to"):
                candidate_client_names.append(str(row["order_person_spoken_to"]).strip())
            if row.get("rm_name"):
                candidate_rm_names.append(str(row["rm_name"]).strip())
            if row.get("main_cif_no"):
                candidate_main_cif_nos.append(str(row["main_cif_no"]).strip())
            if row.get("trade_date_raw"):
                candidate_trade_dates.append(str(row["trade_date_raw"]).strip())
            candidate_ptcc_descriptions.extend(_flatten_ptcc_descriptions(row.get("ptcc_description")))

        contact_candidates = []
        excel_cif_no = manual_row.get("excel_cif_no")
        if excel_cif_no and excel_cif_no in latest_contact_by_cif:
            latest = latest_contact_by_cif[excel_cif_no]
            contact_candidates.append(
                {
                    "source": "excel_cif_no",
                    "cif_no": excel_cif_no,
                    "biz_dt": latest.get("biz_dt"),
                    "client_phone_number": latest.get("client_phone_number"),
                }
            )
        for cif_no in _distinct_non_empty(candidate_main_cif_nos):
            latest = latest_contact_by_cif.get(cif_no)
            if latest is None:
                continue
            contact_candidates.append(
                {
                    "source": "main_cif_no",
                    "cif_no": cif_no,
                    "biz_dt": latest.get("biz_dt"),
                    "client_phone_number": latest.get("client_phone_number"),
                }
            )

        candidate_client_phone_numbers = _distinct_non_empty(
            [candidate.get("client_phone_number") for candidate in contact_candidates]
        )
        primary_contact = contact_candidates[0] if contact_candidates else None
        if primary_contact and primary_contact.get("source") != "excel_cif_no":
            primary_contact = None
        if primary_contact is None and len(candidate_client_phone_numbers) == 1:
            matching = [candidate for candidate in contact_candidates if candidate.get("client_phone_number") == candidate_client_phone_numbers[0]]
            primary_contact = matching[0] if matching else None

        trade_date = _parse_normalized_timestamp(manual_row.get("excel_trade_date_raw"))
        input_date_only = _parse_normalized_timestamp(manual_row.get("excel_input_date_raw"))
        resolved_op_codes = _distinct_non_empty([row.get("op_code") for row in candidate_rows])
        enriched_entry = {
            "client_name": _singleton_or_none(candidate_client_names),
            "rm_name": _singleton_or_none(candidate_rm_names),
            "trade_date": trade_date,
            "input_date_only": input_date_only,
            "client_phone_number": primary_contact.get("client_phone_number") if primary_contact else None,
            "biz_dt": primary_contact.get("biz_dt") if primary_contact else None,
            "call_id": manual_row.get("call_id"),
            "call_date_vl": manual_row.get("call_date_vl"),
            "phone_number_vl": manual_row.get("phone_number_vl"),
            "dialed_in_number": manual_row.get("dialed_in_number"),
            "direction": manual_row.get("direction"),
            "source_mapping_file": manual_row.get("source_mapping_file"),
            "main_cif_no": _singleton_or_none(candidate_main_cif_nos),
            "op_code": _singleton_or_none(resolved_op_codes),
            "excel_op_code": _singleton_or_none(manual_op_codes),
            "excel_cif_no": excel_cif_no,
            "trade_date_raw": manual_row.get("excel_trade_date_raw"),
            "input_date": manual_row.get("excel_input_date_raw"),
            "trade_datetime": None,
            "creation_date": None,
            "order_expiry_date": None,
            "order_initiation": _singleton_or_none([row.get("order_initiation") for row in candidate_rows]),
            "portfolio_begin_date": _singleton_or_none([row.get("portfolio_begin_date") for row in candidate_rows]),
            "portfolio_end_date": _singleton_or_none([row.get("portfolio_end_date") for row in candidate_rows]),
            "portfolio_code": _singleton_or_none([row.get("portfolio_code") for row in candidate_rows]),
            "portfolio_status": _singleton_or_none([row.get("portfolio_status") for row in candidate_rows]),
            "instrument_ud_type": _singleton_or_none([row.get("instrument_ud_type") for row in candidate_rows]),
            "instrument_ud_sub_type": _singleton_or_none([row.get("instrument_ud_sub_type") for row in candidate_rows]),
            "instrument_code": _singleton_or_none([row.get("instrument_code") for row in candidate_rows]),
            "order_person_spoken_to": _singleton_or_none(candidate_client_names),
            "product_risk_profile": _singleton_or_none([row.get("product_risk_profile") for row in candidate_rows]),
            "product_risk_profile_num": _singleton_or_none([row.get("product_risk_profile_num") for row in candidate_rows]),
            "client_risk_profile": _singleton_or_none([row.get("client_risk_profile") for row in candidate_rows]),
            "client_risk_profile_num": _singleton_or_none([row.get("client_risk_profile_num") for row in candidate_rows]),
            "ptcc_description": _distinct_non_empty(candidate_ptcc_descriptions),
            "ground_truth": manual_row["ground_truth"],
            "manual_op_codes": manual_op_codes,
            "transaction_candidates": transaction_candidates,
            "candidate_client_names": _distinct_non_empty(candidate_client_names),
            "candidate_rm_names": _distinct_non_empty(candidate_rm_names),
            "candidate_main_cif_nos": _distinct_non_empty(candidate_main_cif_nos),
            "candidate_trade_dates": _distinct_non_empty(candidate_trade_dates),
            "candidate_client_phone_numbers": candidate_client_phone_numbers,
            "contact_candidates": contact_candidates,
            "resolved_via_manual_match": True,
            "manual_match_count": len(manual_op_codes),
            "metadata_ambiguous": any(
                len(values) > 1
                for values in [
                    _distinct_non_empty(candidate_client_names),
                    _distinct_non_empty(candidate_rm_names),
                    _distinct_non_empty(candidate_main_cif_nos),
                    candidate_client_phone_numbers,
                    resolved_op_codes,
                ]
            ),
        }
        metadata_lookup[transcript_id] = {**metadata_lookup.get(transcript_id, {}), **enriched_entry}
        enriched_count += 1

    if details_rows:
        details_df = pd.DataFrame(details_rows)
        DETAILS_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        details_df.to_csv(DETAILS_OUTPUT_PATH, index=False, encoding="utf-8-sig")
        print(f"Saved manual enrichment details to: {DETAILS_OUTPUT_PATH}")

    print(
        "Manual metadata enrichment summary: "
        f"manual_transcripts={len(manual_by_transcript)} "
        f"enriched_transcripts={enriched_count} "
        f"total_metadata={len(metadata_lookup)}"
    )
    return metadata_lookup


def main() -> None:
    metadata_lookup = load_metadata_with_enrichment()
    base.save_client_metadata_to_file(metadata_lookup, METADATA_OUTPUT_PATH)


if __name__ == "__main__":
    main()
