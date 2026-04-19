import re
from pathlib import Path
from typing import Dict, Optional, Set

import pandas as pd
from pyspark.sql import DataFrame, SparkSession, functions as F
from pyspark.sql.types import StringType, StructField, StructType
from pyspark.sql.window import Window

from ..core.config import load_config
from ..core.paths import resolve_project_path
from . import joiner_hive as base


CONFIG = load_config()
METADATA_SETTINGS = CONFIG.get("metadata", {})
OUTPUT_SETTINGS = CONFIG.get("outputs", {})


SPARK_APP_NAME = "metadata_vl_excel_hive"
EXCEL_PATH = resolve_project_path(METADATA_SETTINGS.get("excel_path", "SG VL metadata (updated).xlsx"))
EXCEL_SHEET_NAME = METADATA_SETTINGS.get("excel_sheet_name", "500 VL Samples")
METADATA_OUTPUT_PATH = resolve_project_path(
    OUTPUT_SETTINGS.get("metadata_vl_excel_output_path", "data/results/merged_client_metadata_vl_excel_hive.json")
)
ENRICHED_EXCEL_OUTPUT_PATH = resolve_project_path(
    OUTPUT_SETTINGS.get("metadata_vl_excel_csv_path", "data/results/SG_VL_metadata_vl_excel_hive.csv")
)

EXT_COL = "Extension (True after BAU)"
DT_COL = "Inputted Client Order/ Callback Date/Time"
TRADE_DT_COL = "Trade Date"
VL_REVIEWED_COL = "VL reviewed  (True after BAU)"
INPUTTED_PHONE_EXT_COL = "Inputted Phone/Ext"
EXCEL_CIF_COL = "Order Person CIF / Entity Code"

DATE_RE = re.compile(r"\b(\d{2}/\d{2}/\d{4})\b")

MATCHED_EXCEL_SCHEMA = StructType(
    [
        StructField("excel_row_id", StringType(), True),
        StructField("transcript_id", StringType(), True),
        StructField("call_id", StringType(), True),
        StructField("call_date_vl", StringType(), True),
        StructField("phone_number_vl", StringType(), True),
        StructField("dialed_in_number", StringType(), True),
        StructField("direction", StringType(), True),
        StructField("source_mapping_file", StringType(), True),
        StructField("excel_op_code", StringType(), True),
        StructField("excel_cif_no", StringType(), True),
        StructField("excel_trade_date_raw", StringType(), True),
        StructField("excel_input_date_raw", StringType(), True),
    ]
)


def normalize_digits(value) -> str | None:
    """Return a digits-only key without introducing CSV-style .0 artifacts."""
    if pd.isna(value):
        return None
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        value = str(value)

    text = str(value).strip()
    if re.fullmatch(r"\d+\.0", text):
        return text.split(".")[0]

    digits = re.sub(r"\D+", "", text)
    return digits if digits else None


def normalize_ddmmyyyy(value) -> str | None:
    """Parse date-like values and return dd/mm/yyyy."""
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.strftime("%d/%m/%Y")

    text = str(value).strip()
    match = DATE_RE.search(text)
    if match:
        return match.group(1)

    parsed = pd.to_datetime(text, errors="coerce", dayfirst=True)
    if pd.isna(parsed):
        return None
    return parsed.strftime("%d/%m/%Y")


def normalize_vl_reviewed_date_fuzzy(value) -> str | None:
    """Extract a dd/mm/yyyy date from the VL reviewed column, including compact digit forms."""
    if pd.isna(value):
        return None

    text = str(value).strip()
    match = DATE_RE.search(text)
    if match:
        return match.group(1)

    parsed = pd.to_datetime(text, errors="coerce", dayfirst=True)
    if pd.notna(parsed):
        return parsed.strftime("%d/%m/%Y")

    digits = re.sub(r"\D+", "", text)
    if not digits:
        return None

    def try_parse(raw_digits: str, fmt: str) -> str | None:
        dt = pd.to_datetime(raw_digits, format=fmt, errors="coerce")
        if pd.isna(dt):
            return None
        return dt.strftime("%d/%m/%Y")

    for index in range(0, len(digits) - 8 + 1):
        block = digits[index : index + 8]
        for fmt in ("%d%m%Y", "%Y%m%d"):
            out = try_parse(block, fmt)
            if out:
                return out

    for index in range(0, len(digits) - 6 + 1):
        block = digits[index : index + 6]
        day = int(block[0:2])
        month = int(block[2:4])
        year_two = int(block[4:6])
        if not (1 <= day <= 31 and 1 <= month <= 12):
            continue
        year = 1900 + year_two if year_two >= 70 else 2000 + year_two
        out = try_parse(f"{day:02d}{month:02d}{year:04d}", "%d%m%Y")
        if out:
            return out

    return None


def iter_lookup_candidates(row) -> list[tuple[str, str]]:
    """Match Excel rows to voice-log keys in the same priority order used by mapping.py."""
    specs = [
        ("extension_key", "date_key"),
        ("extension_key", "trade_date_key"),
        ("extension_key", "vl_date_key"),
        ("inputted_phone_ext_key", "vl_date_fuzzy"),
    ]
    seen: set[tuple[str, str]] = set()
    candidates: list[tuple[str, str]] = []

    for ext_col, date_col in specs:
        extension = row.get(ext_col)
        date_value = row.get(date_col)
        if pd.isna(extension) or pd.isna(date_value):
            continue
        extension = str(extension).strip()
        date_value = str(date_value).strip()
        if not extension or not date_value:
            continue
        key = (extension, date_value)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(key)

    return candidates


def _load_voice_log_match_rows(transcript_ids: Optional[Set[str]] = None) -> pd.DataFrame:
    """Parse local voice-log txt files into the transcript-level fields used by mapping.py."""
    if not base.VOICE_LOG_METADATA_DIR.exists():
        raise RuntimeError(f"Voice log metadata directory not found: {base.VOICE_LOG_METADATA_DIR}")

    rows = []
    wanted_ids = transcript_ids if transcript_ids is not None else None
    mapping_files = sorted(path for path in base.VOICE_LOG_METADATA_DIR.glob("*.txt") if path.is_file())
    if not mapping_files:
        raise RuntimeError(f"No voice log txt files found in: {base.VOICE_LOG_METADATA_DIR}")

    for path in mapping_files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        blocks = re.split(base.BLOCK_SPLIT_PATTERN, text)
        for block in (block.strip() for block in blocks if block.strip()):
            transcript_id = base._extract_transcript_id(block)
            if not transcript_id or (wanted_ids is not None and transcript_id not in wanted_ids):
                continue

            voice_log_fields = base._extract_voice_log_fields_from_last_line(block)
            if not voice_log_fields:
                continue

            lines = [line.strip() for line in block.splitlines() if line.strip()]
            last_line_tokens = lines[-1].split() if lines else []
            call_id = last_line_tokens[0] if last_line_tokens else None

            rows.append(
                {
                    "transcript_id": transcript_id,
                    "call_id": call_id,
                    "extension": voice_log_fields["extension_raw"],
                    "call_date": voice_log_fields["voice_trade_date_raw"],
                    "phone_number_vl": voice_log_fields["phone_number_vl"],
                    "dialed_in_number": voice_log_fields["dialed_in_number"],
                    "direction": voice_log_fields["direction"],
                    "source_mapping_file": path.name,
                }
            )

    return pd.DataFrame(rows)


def _load_excel_matches(transcript_ids: Optional[Set[str]] = None) -> pd.DataFrame:
    """Load all Excel rows, append matched VL fields where available, and keep Excel op_code/CIF columns."""
    if not EXCEL_PATH.exists():
        raise FileNotFoundError(f"Excel metadata file not found: {EXCEL_PATH}")

    df_map = _load_voice_log_match_rows(transcript_ids)
    if df_map.empty:
        raise RuntimeError(f"No voice log metadata rows found in: {base.VOICE_LOG_METADATA_DIR}")
    df_map = df_map.dropna(subset=["transcript_id", "extension", "call_date"]).copy()

    key_cols = ["extension", "call_date"]
    key_counts = df_map.groupby(key_cols)["transcript_id"].nunique()
    unique_keys = key_counts[key_counts == 1].reset_index()[key_cols]
    df_unique = (
        df_map.merge(unique_keys, on=key_cols, how="inner")
        .sort_values(["extension", "call_date", "transcript_id"])
        .drop_duplicates(subset=key_cols, keep="first")
    )

    df_excel = pd.read_excel(EXCEL_PATH, dtype=str, sheet_name=EXCEL_SHEET_NAME)
    op_code_col = next((column_name for column_name in df_excel.columns if "Op Code" in str(column_name)), None)
    if op_code_col is None:
        raise ValueError("Excel metadata missing an op code column containing 'Op Code' in the header.")

    required_columns = [EXT_COL, DT_COL, TRADE_DT_COL, VL_REVIEWED_COL, INPUTTED_PHONE_EXT_COL, EXCEL_CIF_COL]
    missing_columns = [column_name for column_name in required_columns if column_name not in df_excel.columns]
    if missing_columns:
        raise ValueError(f"Excel metadata missing required columns: {missing_columns}")

    df_excel_out = df_excel.copy()
    df_excel_out["excel_row_id"] = [str(index) for index in range(len(df_excel_out))]
    df_excel_out["extension_key"] = df_excel_out[EXT_COL].map(normalize_digits)
    df_excel_out["date_key"] = df_excel_out[DT_COL].map(normalize_ddmmyyyy)
    df_excel_out["trade_date_key"] = df_excel_out[TRADE_DT_COL].map(normalize_ddmmyyyy)
    df_excel_out["vl_date_key"] = df_excel_out[VL_REVIEWED_COL].map(normalize_ddmmyyyy)
    df_excel_out["inputted_phone_ext_key"] = df_excel_out[INPUTTED_PHONE_EXT_COL].map(normalize_digits)
    df_excel_out["vl_date_fuzzy"] = df_excel_out[VL_REVIEWED_COL].map(normalize_vl_reviewed_date_fuzzy)
    df_excel_out["excel_op_code"] = df_excel_out[op_code_col].map(
        lambda value: str(value).strip() if pd.notna(value) and str(value).strip() else None
    )
    df_excel_out["excel_cif_no"] = df_excel_out[EXCEL_CIF_COL].map(normalize_digits)

    key_to_row = df_unique.set_index(["extension", "call_date"])[
        ["call_id", "transcript_id", "direction", "dialed_in_number", "phone_number_vl", "source_mapping_file"]
    ].to_dict(orient="index")

    def lookup_transcript(row):
        for extension, date_value in iter_lookup_candidates(row):
            record = key_to_row.get((extension, date_value))
            if record is not None:
                return (
                    record.get("call_id", pd.NA),
                    record.get("transcript_id", pd.NA),
                    extension,
                    date_value,
                    record.get("direction", pd.NA),
                    record.get("dialed_in_number", pd.NA),
                    record.get("phone_number_vl", pd.NA),
                    record.get("source_mapping_file", pd.NA),
                )
        return pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA

    fields = df_excel_out.apply(lookup_transcript, axis=1, result_type="expand")
    fields.columns = [
        "call_id",
        "transcript_id",
        "matched_extension",
        "matched_date",
        "direction",
        "dialed_in_number",
        "phone_number_vl",
        "source_mapping_file",
    ]
    for column_name in fields.columns:
        df_excel_out[column_name] = fields[column_name]

    duplicate_match_mask = (
        df_excel_out["transcript_id"].notna()
        & df_excel_out["matched_extension"].notna()
        & df_excel_out["matched_date"].notna()
        & df_excel_out.duplicated(subset=["matched_extension", "matched_date"], keep=False)
    )
    if duplicate_match_mask.any():
        duplicate_rows = int(duplicate_match_mask.sum())
        print(
            "Excel fallback mapping skipped rows because one voice-log key matched multiple Excel rows: "
            f"{duplicate_rows} rows"
        )
        df_csv_cols = [
            "call_id",
            "transcript_id",
            "matched_extension",
            "matched_date",
            "direction",
            "dialed_in_number",
            "phone_number_vl",
            "source_mapping_file",
        ]
        df_excel_out.loc[duplicate_match_mask, df_csv_cols] = pd.NA

    matched_rows = df_excel_out[df_excel_out["transcript_id"].notna()].copy()
    transcript_match_counts = matched_rows.groupby("transcript_id").size()
    ambiguous_transcript_ids = transcript_match_counts[transcript_match_counts > 1].index.tolist()
    if ambiguous_transcript_ids:
        sample_ids = ", ".join(sorted(ambiguous_transcript_ids)[:10])
        print(
            "Excel fallback mapping skipped transcript IDs with multiple matched Excel rows: "
            f"{len(ambiguous_transcript_ids)} conflicts (examples: {sample_ids})"
        )
        df_excel_out.loc[df_excel_out["transcript_id"].isin(ambiguous_transcript_ids), [
            "call_id",
            "transcript_id",
            "matched_extension",
            "matched_date",
            "direction",
            "dialed_in_number",
            "phone_number_vl",
            "source_mapping_file",
        ]] = pd.NA

    df_excel_out = df_excel_out.rename(
        columns={
            "matched_date": "call_date_vl",
            TRADE_DT_COL: "excel_trade_date_raw",
            DT_COL: "excel_input_date_raw",
        }
    )
    matched_count = int(df_excel_out["transcript_id"].notna().sum())
    print(f"Loaded {len(df_excel_out)} Excel rows, with {matched_count} matched to voice logs, from: {EXCEL_PATH}")
    return df_excel_out


def _excel_matches_to_spark(spark: SparkSession, matches: pd.DataFrame) -> DataFrame:
    """Convert matched Excel rows into a small Spark DataFrame for Hive enrichment."""
    if matches.empty:
        return spark.createDataFrame([], schema=MATCHED_EXCEL_SCHEMA)

    records = [
        {
            "excel_row_id": str(row["excel_row_id"]).strip(),
            "transcript_id": str(row["transcript_id"]).strip() if pd.notna(row["transcript_id"]) else None,
            "call_id": str(row["call_id"]).strip() if pd.notna(row["call_id"]) else None,
            "call_date_vl": str(row["call_date_vl"]).strip()
            if pd.notna(row["call_date_vl"])
            else None,
            "phone_number_vl": str(row["phone_number_vl"]).strip() if pd.notna(row["phone_number_vl"]) else None,
            "dialed_in_number": str(row["dialed_in_number"]).strip()
            if pd.notna(row["dialed_in_number"])
            else None,
            "direction": str(row["direction"]).strip() if pd.notna(row["direction"]) else None,
            "source_mapping_file": str(row["source_mapping_file"]).strip()
            if pd.notna(row["source_mapping_file"])
            else None,
            "excel_op_code": str(row["excel_op_code"]).strip() if pd.notna(row["excel_op_code"]) else None,
            "excel_cif_no": str(row["excel_cif_no"]).strip() if pd.notna(row["excel_cif_no"]) else None,
            "excel_trade_date_raw": str(row["excel_trade_date_raw"]).strip()
            if pd.notna(row["excel_trade_date_raw"])
            else None,
            "excel_input_date_raw": str(row["excel_input_date_raw"]).strip()
            if pd.notna(row["excel_input_date_raw"])
            else None,
        }
        for _, row in matches.iterrows()
        if pd.notna(row["excel_row_id"])
    ]
    return spark.createDataFrame(records, schema=MATCHED_EXCEL_SCHEMA)


def _load_transaction_metadata_for_excel_rows(spark: SparkSession, excel_df: DataFrame) -> DataFrame:
    """Load transaction metadata for matched Excel rows by op_code, using nearest Excel input datetime to break ties."""
    normalized_join_keys = excel_df.select(F.col("excel_op_code").alias("op_code")).dropna().dropDuplicates()
    if not normalized_join_keys.head(1):
        return excel_df.select("excel_row_id").dropDuplicates().select(
            "excel_row_id",
            *[F.lit(None).cast("string").alias(field_name) for field_name in base.TRANSACTION_METADATA_FIELDS],
            F.lit(0).cast("int").alias("txn_match_count"),
        ).limit(0)

    df = spark.table(base.TRANSACTION_HIVE_TABLE)
    base._require_columns(df, base.TRANSACTION_HIVE_TABLE, [source for source, _ in base.TRANSACTION_COLUMN_ALIASES])

    op_code_filters = [row["op_code"] for row in normalized_join_keys.select("op_code").collect()]
    if op_code_filters:
        df = df.filter(F.col("op_code").cast("string").isin(op_code_filters))

    txn_selected = (
        df.select(*[base._string_col(source).alias(alias) for source, alias in base.TRANSACTION_COLUMN_ALIASES])
        .withColumn("trade_date_key", base._date_key_col("trade_date_raw"))
        .withColumn("trade_datetime_ts", base._timestamp_col("trade_datetime"))
        .dropDuplicates()
    )

    joined = excel_df.alias("excel").join(
        txn_selected.alias("txn"),
        on=F.col("excel.excel_op_code") == F.col("txn.op_code"),
        how="left",
    )

    has_transaction_candidate = F.when(F.col("txn.op_code").isNotNull(), F.lit(1)).otherwise(F.lit(0))
    trade_time_diff_seconds = F.when(
        F.col("excel.excel_input_datetime_ts").isNotNull() & F.col("txn.trade_datetime_ts").isNotNull(),
        F.abs(
            F.unix_timestamp(F.col("txn.trade_datetime_ts"))
            - F.unix_timestamp(F.col("excel.excel_input_datetime_ts"))
        ),
    )
    has_time_comparison = F.when(trade_time_diff_seconds.isNotNull(), F.lit(1)).otherwise(F.lit(0))

    transcript_window = Window.partitionBy("excel.excel_row_id")
    ranking_window = transcript_window.orderBy(
        has_transaction_candidate.desc(),
        has_time_comparison.desc(),
        trade_time_diff_seconds.asc_nulls_last(),
        F.col("txn.trade_datetime_ts").asc_nulls_last(),
    )
    txn_candidate_details = F.concat_ws(
        " | ",
        F.sort_array(
            F.collect_set(
                F.when(
                    F.col("txn.op_code").isNotNull(),
                    F.concat(
                        F.lit("op_code="),
                        F.coalesce(F.col("txn.op_code"), F.lit("<NULL>")),
                        F.lit(", trade_datetime="),
                        F.coalesce(F.col("txn.trade_datetime"), F.lit("<NULL>")),
                        F.lit(", main_cif_no="),
                        F.coalesce(F.col("txn.main_cif_no"), F.lit("<NULL>")),
                        F.lit(", client_name="),
                        F.coalesce(F.col("txn.order_person_spoken_to"), F.lit("<NULL>")),
                        F.lit(", rm_name="),
                        F.coalesce(F.col("txn.rm_name"), F.lit("<NULL>")),
                    ),
                )
            ).over(transcript_window)
        ),
    )

    ranked = (
        joined.withColumn("txn_candidate_count", F.sum(has_transaction_candidate).over(transcript_window))
        .withColumn("trade_time_diff_seconds", trade_time_diff_seconds)
        .withColumn("txn_candidate_details", txn_candidate_details)
        .withColumn(
            "comparable_candidate_count",
            F.sum(F.when(F.col("trade_time_diff_seconds").isNotNull(), F.lit(1)).otherwise(F.lit(0))).over(
                transcript_window
            ),
        )
        .withColumn("min_trade_time_diff_seconds", F.min("trade_time_diff_seconds").over(transcript_window))
        .withColumn(
            "best_time_candidate_count",
            F.sum(
                F.when(
                    F.col("trade_time_diff_seconds") == F.col("min_trade_time_diff_seconds"),
                    F.lit(1),
                ).otherwise(F.lit(0))
            ).over(transcript_window),
        )
        .withColumn(
            "txn_match_count",
            F.when(F.col("txn_candidate_count") <= 1, F.col("txn_candidate_count"))
            .when(F.col("comparable_candidate_count") == 0, F.col("txn_candidate_count"))
            .otherwise(F.col("best_time_candidate_count")),
        )
        .withColumn("candidate_rank", F.row_number().over(ranking_window))
    )

    ambiguous_rows = ranked.filter((F.col("candidate_rank") == 1) & (F.col("txn_match_count") > 1)).select(
        F.col("excel.excel_row_id").alias("excel_row_id"),
        F.col("excel.transcript_id").alias("transcript_id"),
        F.col("excel.excel_op_code").alias("excel_op_code"),
        F.col("excel.excel_input_date_raw").alias("excel_input_date_raw"),
        "txn_match_count",
    ).collect()
    if ambiguous_rows:
        print("Transaction metadata kept multiple nearest-datetime candidates for these Excel op_code values:")
        for row in ambiguous_rows:
            print(
                f"  excel_row_id={row['excel_row_id']}, "
                f"  transcript_id={row['transcript_id']}, "
                f"op_code={row['excel_op_code']}, "
                f"excel_input_date={row['excel_input_date_raw']}, "
                f"txn_match_count={row['txn_match_count']}"
            )

    return ranked.filter(F.col("candidate_rank") == 1).select(
        F.col("excel.excel_row_id").alias("excel_row_id"),
        *[
            F.when(F.col("txn_match_count") == 1, F.col(f"txn.{field_name}")).alias(field_name)
            for field_name in base.TRANSACTION_METADATA_FIELDS
        ],
        "txn_match_count",
        "txn_candidate_count",
        "txn_candidate_details",
    )


def _attach_latest_contact_metadata_for_excel_rows(base_df: DataFrame, contact_df: DataFrame) -> DataFrame:
    """Join contacts on CIF and keep the latest biz date per Excel row."""
    joined = base_df.alias("base").join(
        contact_df.alias("contact"),
        on=F.col("base.main_cif_no") == F.col("contact.contact_cif_number"),
        how="left",
    )

    ranking_window = Window.partitionBy("base.excel_row_id").orderBy(
        F.col("contact.contact_date_key").desc_nulls_last()
    )
    ranked = joined.withColumn("contact_row_rank", F.row_number().over(ranking_window))
    base_columns = [F.col(f"base.{column_name}").alias(column_name) for column_name in base_df.columns]

    return ranked.filter(F.col("contact_row_rank") == 1).select(
        *base_columns,
        F.col("contact.biz_dt").alias("biz_dt"),
        F.col("contact.client_phone_number").alias("client_phone_number"),
    )


def _save_enriched_excel_rows(
    excel_matches: pd.DataFrame,
    rows: list,
    output_path: Path = ENRICHED_EXCEL_OUTPUT_PATH,
) -> None:
    """Save matched Excel rows plus VL and Hive enrichment columns to a local CSV file."""
    if excel_matches.empty:
        return

    helper_columns = {
        "extension_key",
        "date_key",
        "trade_date_key",
        "vl_date_key",
        "inputted_phone_ext_key",
        "vl_date_fuzzy",
        "matched_extension",
    }
    excel_columns = [column_name for column_name in excel_matches.columns if column_name not in helper_columns]
    enriched_excel = excel_matches[excel_columns].copy()

    hive_rows = [row.asDict(recursive=True) for row in rows]
    if hive_rows:
        hive_df = pd.DataFrame(hive_rows)
        hive_columns = [
            "excel_row_id",
            "txn_match_count",
            "txn_candidate_count",
            "txn_candidate_details",
            "client_phone_number",
            "biz_dt",
            "main_cif_no",
            "op_code",
            "trade_date_raw",
            "input_date_raw",
            "trade_datetime",
            "creation_date",
            "order_expiry_date",
            "order_initiation",
            "portfolio_begin_date",
            "portfolio_end_date",
            "portfolio_code",
            "portfolio_status",
            "instrument_ud_type",
            "instrument_ud_sub_type",
            "instrument_code",
            "order_person_spoken_to",
            "rm_name",
            "product_risk_profile",
            "product_risk_profile_num",
            "client_risk_profile",
            "client_risk_profile_num",
            "ptcc_description",
        ]
        available_hive_columns = [column_name for column_name in hive_columns if column_name in hive_df.columns]
        if available_hive_columns:
            enriched_excel = enriched_excel.merge(
                hive_df[available_hive_columns].drop_duplicates(subset=["excel_row_id"]),
                on="excel_row_id",
                how="left",
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    enriched_excel.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Saved enriched Excel rows to: {output_path}")


def load_client_metadata_from_hive(transcript_ids: Optional[Set[str]] = None) -> Dict[str, Dict[str, Any]]:
    """Build transcript metadata from Excel rows matched to voice logs, then enrich them from Hive."""
    print(
        "Loading matched Excel rows from "
        f"{EXCEL_PATH} using voice-log metadata in {base.VOICE_LOG_METADATA_DIR}, then enriching from Hive tables: "
        f"{base.TRANSACTION_HIVE_TABLE}, {base.CONTACT_HIVE_TABLE}, {base.DESCRIPTION_HIVE_TABLE}, {EXCEL_PATH}"
    )
    print("Hive access mode is read-only: this module only reads with spark.table(...) and never writes.")

    excel_matches = _load_excel_matches(transcript_ids)
    if excel_matches.empty:
        raise RuntimeError(f"No Excel rows could be matched to local voice logs from: {EXCEL_PATH}")
    scenario_columns = {}
    for column_name in excel_matches.columns:
        column_text = str(column_name).strip()
        if column_text.isdigit() and 1 <= int(column_text) <= 14:
            scenario_columns[column_text] = column_name

    ground_truth_by_excel_row_id = {}
    for _, row in excel_matches.iterrows():
        ground_truth_by_excel_row_id[str(row["excel_row_id"]).strip()] = {
            str(scenario_id): (
                None
                if str(scenario_id) not in scenario_columns
                or pd.isna(row[scenario_columns[str(scenario_id)]])
                or not str(row[scenario_columns[str(scenario_id)]]).strip()
                else str(row[scenario_columns[str(scenario_id)]]).strip()
            )
            for scenario_id in range(1, 15)
        }

    spark = SparkSession.builder.appName(SPARK_APP_NAME).enableHiveSupport().getOrCreate()
    try:
        excel_df = _excel_matches_to_spark(spark, excel_matches).withColumn(
            "excel_input_datetime_ts", base._timestamp_col("excel_input_date_raw")
        )
        txn_df = _load_transaction_metadata_for_excel_rows(spark, excel_df)
        base_df = excel_df.join(F.broadcast(txn_df), on="excel_row_id", how="left")

        contact_lookup_df = (
            base_df.withColumn(
                "contact_lookup_cif",
                F.coalesce(F.col("main_cif_no"), F.col("excel_cif_no")),
            )
            .withColumn("main_cif_no_original", F.col("main_cif_no"))
            .withColumn("main_cif_no", F.col("contact_lookup_cif"))
        )
        contact_cif_keys = contact_lookup_df.select("main_cif_no").dropna().dropDuplicates()
        contact_df = base._load_contact_metadata(spark, contact_cif_keys)
        base_with_contact_df = (
            _attach_latest_contact_metadata_for_excel_rows(contact_lookup_df, contact_df)
            .withColumn("main_cif_no", F.col("main_cif_no_original"))
            .drop("main_cif_no_original", "contact_lookup_cif")
        )
        description_join_keys = base_df.select("op_code").dropna().dropDuplicates()
        description_df = base._load_description_metadata(spark, description_join_keys)
        joined = base._attach_description_metadata(base_with_contact_df, description_df)
        rows = joined.collect()
    finally:
        spark.stop()

    _save_enriched_excel_rows(excel_matches, rows)

    lookup: Dict[str, Dict[str, Any]] = {}
    no_transaction_match = 0
    phone_from_excel_cif = 0

    for row in rows:
        row_dict = row.asDict(recursive=True)
        transcript_id = str(row_dict["transcript_id"]).strip() if row_dict["transcript_id"] is not None else ""
        if not transcript_id or transcript_id.lower() == "nan":
            continue

        if row_dict.get("op_code") is None:
            no_transaction_match += 1
        if row_dict.get("op_code") is None and row_dict.get("excel_cif_no") and row_dict.get("client_phone_number"):
            phone_from_excel_cif += 1

        trade_date = pd.to_datetime(
            row_dict.get("trade_date_raw") or row_dict.get("excel_trade_date_raw") or row_dict.get("call_date_vl"),
            dayfirst=True,
            errors="coerce",
        )
        input_date_only = pd.to_datetime(
            row_dict.get("input_date_raw") or row_dict.get("excel_input_date_raw"),
            errors="coerce",
            dayfirst=True,
        )
        excel_row_id = str(row_dict["excel_row_id"]).strip()

        lookup[transcript_id] = {
            "client_name": str(row_dict["order_person_spoken_to"]).strip()
            if row_dict["order_person_spoken_to"] is not None
            else None,
            "rm_name": str(row_dict["rm_name"]).strip() if row_dict["rm_name"] is not None else None,
            "trade_date": trade_date.normalize() if pd.notna(trade_date) else None,
            "input_date_only": input_date_only.normalize() if pd.notna(input_date_only) else None,
            "client_phone_number": row_dict["client_phone_number"],
            "biz_dt": row_dict["biz_dt"],
            "call_id": row_dict["call_id"],
            "call_date_vl": row_dict["call_date_vl"],
            "phone_number_vl": row_dict["phone_number_vl"],
            "dialed_in_number": row_dict["dialed_in_number"],
            "direction": row_dict["direction"],
            "source_mapping_file": row_dict["source_mapping_file"],
            "main_cif_no": row_dict["main_cif_no"],
            "op_code": row_dict["op_code"],
            "excel_op_code": row_dict.get("excel_op_code"),
            "excel_cif_no": row_dict.get("excel_cif_no"),
            "trade_date_raw": row_dict["trade_date_raw"] or row_dict.get("excel_trade_date_raw"),
            "input_date": row_dict["input_date_raw"] or row_dict.get("excel_input_date_raw"),
            "trade_datetime": row_dict["trade_datetime"],
            "creation_date": row_dict["creation_date"],
            "order_expiry_date": row_dict["order_expiry_date"],
            "order_initiation": row_dict["order_initiation"],
            "portfolio_begin_date": row_dict["portfolio_begin_date"],
            "portfolio_end_date": row_dict["portfolio_end_date"],
            "portfolio_code": row_dict["portfolio_code"],
            "portfolio_status": row_dict["portfolio_status"],
            "instrument_ud_type": row_dict["instrument_ud_type"],
            "instrument_ud_sub_type": row_dict["instrument_ud_sub_type"],
            "instrument_code": row_dict["instrument_code"],
            "order_person_spoken_to": row_dict["order_person_spoken_to"],
            "product_risk_profile": row_dict["product_risk_profile"],
            "product_risk_profile_num": row_dict["product_risk_profile_num"],
            "client_risk_profile": row_dict["client_risk_profile"],
            "client_risk_profile_num": row_dict["client_risk_profile_num"],
            "ptcc_description": row_dict["ptcc_description"],
            "ground_truth": ground_truth_by_excel_row_id.get(
                excel_row_id, {str(scenario_id): None for scenario_id in range(1, 15)}
            ),
        }

    print(
        "Excel+Hive metadata summary: "
        f"loaded={len(lookup)} "
        f"transaction_unmatched={no_transaction_match} "
        f"phone_from_excel_cif={phone_from_excel_cif}"
    )
    return lookup


def main() -> None:
    """Run the Hive join, then backfill missing op_code rows through Excel mapping, and save the merged JSON."""
    metadata_lookup = load_client_metadata_from_hive()
    base.save_client_metadata_to_file(metadata_lookup, METADATA_OUTPUT_PATH)


if __name__ == "__main__":
    main()
