import json
import ntpath
from pathlib import Path
import re
from typing import Any, Dict, Iterable, Optional, Set

import pandas as pd
from pyspark.sql import DataFrame, SparkSession, functions as F
from pyspark.sql.types import StringType, StructField, StructType
from pyspark.sql.window import Window

from ..core.config import load_config
from ..core.paths import resolve_project_path


CONFIG = load_config()
METADATA_SETTINGS = CONFIG.get("metadata", {})
OUTPUT_SETTINGS = CONFIG.get("outputs", {})

VOICE_LOG_METADATA_DIR = resolve_project_path(METADATA_SETTINGS.get("voice_log_metadata_dir", "SG_voice_logs"))
TRANSACTION_HIVE_TABLE = METADATA_SETTINGS.get("transaction_hive_table", "p01_rpb_so_s.magnet_base_transaction_v2_prod")
CONTACT_HIVE_TABLE = METADATA_SETTINGS.get("contact_hive_table", "p01_rpb_so_s.magnet_vtt_contact_temp")
DESCRIPTION_HIVE_TABLE = METADATA_SETTINGS.get("description_hive_table", "p01_rpb_so_s.magnet_r2w_ptcc")
SPARK_APP_NAME = "metadata_joiner"
METADATA_OUTPUT_PATH = resolve_project_path(OUTPUT_SETTINGS.get("metadata_output_path", "data/results/merged_client_metadata.json"))

TRANSACTION_COLUMN_ALIASES = [
    ("main_cif_no", "main_cif_no"),
    ("op_code", "op_code"),
    ("phone_ext", "extension_raw"),
    ("trade_date", "trade_date_raw"),
    ("input_date", "input_date_raw"),
    ("trade_datetime", "trade_datetime"),
    ("creation_date", "creation_date"),
    ("order_expiry_date", "order_expiry_date"),
    ("order_initiation", "order_initiation"),
    ("portfolio_begin_date", "portfolio_begin_date"),
    ("portfolio_end_date", "portfolio_end_date"),
    ("portfolio_code", "portfolio_code"),
    ("portfolio_status", "portfolio_status"),
    ("instrument_ud_type", "instrument_ud_type"),
    ("instrument_ud_sub_type", "instrument_ud_sub_type"),
    ("instrument_code", "instrument_code"),
    ("order_person_spoken_to", "order_person_spoken_to"),
    ("rm_name", "rm_name"),
    ("product_risk_profile", "product_risk_profile"),
    ("product_risk_profile_num", "product_risk_profile_num"),
    ("client_risk_profile", "client_risk_profile"),
    ("client_risk_profile_num", "client_risk_profile_num"),
]
TRANSACTION_METADATA_FIELDS = [alias for _, alias in TRANSACTION_COLUMN_ALIASES if alias != "extension_raw"]

BLOCK_SPLIT_PATTERN = r"\n\s*-{100,}\s*(?:\n|$)"
DATE_RE = re.compile(r"\b(\d{2}/\d{2}/\d{4})\b")
EXT_TOKEN_RE = re.compile(r"(\+\d{7,}|\b\d{5}\b)")
CALLID_RE = re.compile(r"^\d{15,}\b")
DATE_TOKEN_RE = re.compile(r"^\d{2}/\d{2}/\d{4}$")
AMPM_TOKEN_RE = re.compile(r"^(AM|PM)$", flags=re.IGNORECASE)
DURATION_RE = re.compile(r"^\d{2}:\d{2}:\d{2}$")

VOICE_LOG_SCHEMA = StructType(
    [
        StructField("transcript_id", StringType(), True),
        StructField("extension_raw", StringType(), True),
        StructField("voice_trade_date_raw", StringType(), True),
        StructField("phone_number_vl", StringType(), True),
        StructField("dialed_in_number", StringType(), True),
        StructField("direction", StringType(), True),
        StructField("voice_start_time_raw", StringType(), True),
        StructField("voice_stop_time_raw", StringType(), True),
        StructField("voice_duration_raw", StringType(), True),
    ]
)


def _require_columns(df: DataFrame, table_name: str, required: Iterable[str]) -> None:
    """Fail early if a Hive table does not contain the columns this module expects."""
    missing = [column_name for column_name in required if column_name not in df.columns]
    if missing:
        raise ValueError(f"Table {table_name} missing required columns: {missing}")


def _string_col(column_name: str):
    """Trim a Spark string column and convert empty strings to null."""
    return F.when(
        F.trim(F.col(column_name).cast("string")) == "", F.lit(None)
    ).otherwise(F.trim(F.col(column_name).cast("string")))


def _digits_only_col(column_name: str):
    """Keep digits only from a Spark column so phone and extension matching ignores punctuation."""
    digits = F.regexp_replace(F.coalesce(_string_col(column_name), F.lit("")), r"\D+", "")
    return F.when(F.length(digits) == 0, F.lit(None)).otherwise(digits)


def _extension_key_col(column_name: str):
    """Normalize an extension-like column to its last 5 digits for cross-source joins."""
    digits = _digits_only_col(column_name)
    return F.when(digits.isNull(), F.lit(None)).otherwise(
        F.expr(
            f"right(regexp_replace(coalesce(trim(cast(`{column_name}` as string)), ''), '\\\\D+', ''), 5)"
        )
    )


def _date_key_col(column_name: str):
    """Parse common date and datetime string formats and return a Spark date column."""
    raw = F.trim(F.col(column_name).cast("string"))
    date_exprs = [
        F.to_date(raw),
        F.to_date(raw, "dd/MM/yyyy"),
        F.to_date(raw, "d/M/yyyy"),
        F.to_date(raw, "yyyy-MM-dd"),
        F.to_date(raw, "yyyy/MM/dd"),
        F.to_date(raw, "dd-MM-yyyy"),
        F.to_date(raw, "d-M-yyyy"),
        F.to_date(raw, "MM/dd/yyyy"),
        F.to_date(raw, "M/d/yyyy"),
        F.to_date(raw, "yyyyMMdd"),
        F.to_date(F.to_timestamp(raw, "yyyy-MM-dd HH:mm:ss")),
        F.to_date(F.to_timestamp(raw, "yyyy-MM-dd HH:mm")),
        F.to_date(F.to_timestamp(raw, "yyyy/MM/dd HH:mm:ss")),
        F.to_date(F.to_timestamp(raw, "yyyy/MM/dd HH:mm")),
        F.to_date(F.to_timestamp(raw, "dd/MM/yyyy HH:mm:ss")),
        F.to_date(F.to_timestamp(raw, "dd/MM/yyyy HH:mm")),
        F.to_date(F.to_timestamp(raw, "MM/dd/yyyy HH:mm:ss")),
        F.to_date(F.to_timestamp(raw, "MM/dd/yyyy HH:mm")),
    ]
    return F.when(raw == "", F.lit(None)).otherwise(F.coalesce(*date_exprs))


def _timestamp_col(column_name: str):
    """Parse common timestamp strings and return a Spark timestamp column."""
    raw = F.trim(F.col(column_name).cast("string"))
    timestamp_exprs = [
        F.to_timestamp(raw),
        F.to_timestamp(raw, "yyyy-MM-dd HH:mm:ss"),
        F.to_timestamp(raw, "yyyy/MM/dd HH:mm:ss"),
        F.to_timestamp(raw, "yyyy-MM-dd HH:mm"),
        F.to_timestamp(raw, "yyyy/MM/dd HH:mm"),
        F.to_timestamp(raw, "dd/MM/yyyy HH:mm:ss"),
        F.to_timestamp(raw, "d/M/yyyy HH:mm:ss"),
        F.to_timestamp(raw, "dd/MM/yyyy hh:mm:ss a"),
        F.to_timestamp(raw, "d/M/yyyy hh:mm:ss a"),
        F.to_timestamp(raw, "MM/dd/yyyy hh:mm:ss a"),
        F.to_timestamp(raw, "M/d/yyyy hh:mm:ss a"),
    ]
    return F.when(raw == "", F.lit(None)).otherwise(F.coalesce(*timestamp_exprs))


def _extract_transcript_id(block: str) -> Optional[str]:
    """Extract the transcript ID from the 'File name:' line inside one block."""
    for line in block.splitlines():
        line = line.strip()
        if line.lower().startswith("file name"):
            match = re.search(r"(\S+?\.wav)\b", line, flags=re.IGNORECASE)
            if not match:
                return None
            wav_base = ntpath.basename(match.group(1))
            return re.sub(r"\.wav$", "", wav_base, flags=re.IGNORECASE)
    return None


def _normalize_extension_token(token: Optional[str]) -> Optional[str]:
    """Return the last 5 digits of the extension field, or fewer digits if the field is shorter."""
    if not token:
        return None
    digits = re.sub(r"\D+", "", token)
    if not digits:
        return None
    return digits[-5:]


def _build_voice_log_timestamp(
    date_token: Optional[str],
    time_token: Optional[str],
    ampm_token: Optional[str],
) -> Optional[str]:
    """Join VL date, clock token, and AM/PM token into one timestamp string when date and AM/PM are valid."""
    if not date_token or not time_token or not ampm_token:
        return None
    if not DATE_TOKEN_RE.fullmatch(date_token):
        return None
    if not AMPM_TOKEN_RE.fullmatch(ampm_token):
        return None
    return f"{date_token} {time_token} {ampm_token.upper()}"


def _extract_voice_log_fields_from_last_line(block: str) -> Optional[Dict[str, Optional[str]]]:
    """Read the last line of a block and recover join keys plus call timing metadata."""
    lines = [line.strip() for line in block.splitlines() if line.strip()]
    if not lines:
        return None

    last_line = lines[-1]
    tokens = last_line.split()
    if len(tokens) < 6 or not CALLID_RE.search(last_line):
        return None

    start_date = tokens[1] if len(tokens) > 1 and DATE_TOKEN_RE.fullmatch(tokens[1]) else None
    start_time = _build_voice_log_timestamp(
        start_date,
        tokens[2] if len(tokens) > 2 else None,
        tokens[3] if len(tokens) > 3 else None,
    )

    stop_time = None
    duration = None
    if len(tokens) > 7 and DATE_TOKEN_RE.fullmatch(tokens[4]):
        stop_time = _build_voice_log_timestamp(tokens[4], tokens[5], tokens[6])
        if len(tokens) > 7 and DURATION_RE.fullmatch(tokens[7]):
            duration = tokens[7]
    elif len(tokens) > 6:
        stop_time = _build_voice_log_timestamp(start_date, tokens[4], tokens[5])
        if DURATION_RE.fullmatch(tokens[6]):
            duration = tokens[6]

    date_match = DATE_RE.search(last_line)
    voice_trade_date_raw = start_date or (date_match.group(1) if date_match else None)

    extension = _normalize_extension_token(tokens[-6] if len(tokens) >= 6 else None)
    if extension is None:
        ext_match = EXT_TOKEN_RE.search(last_line)
        extension = _normalize_extension_token(ext_match.group(1)) if ext_match else None
    if not voice_trade_date_raw or not extension:
        return None

    direction = tokens[-3]
    phone_number_vl = None if direction.lower() == "internal" else tokens[-5]

    return {
        "extension_raw": extension,
        "voice_trade_date_raw": voice_trade_date_raw,
        "phone_number_vl": phone_number_vl,
        "dialed_in_number": tokens[-4],
        "direction": direction,
        "voice_start_time_raw": start_time,
        "voice_stop_time_raw": stop_time,
        "voice_duration_raw": duration,
    }


def _load_local_voice_log_rows(transcript_ids: Optional[Set[str]]) -> list[Dict[str, Any]]:
    """Read top-level voice-log txt files, keep the first row per transcript, and ignore exact duplicate copies."""
    if not VOICE_LOG_METADATA_DIR.exists():
        raise RuntimeError(f"Voice log metadata directory not found: {VOICE_LOG_METADATA_DIR}")

    rows: list[Dict[str, Any]] = []
    rows_by_transcript_id: dict[str, Dict[str, Any]] = {}
    conflicting_transcript_ids: set[str] = set()
    wanted_ids = transcript_ids if transcript_ids is not None else None
    mapping_files = sorted(path for path in VOICE_LOG_METADATA_DIR.glob("*.txt") if path.is_file())

    if not mapping_files:
        raise RuntimeError(f"No voice log txt files found in: {VOICE_LOG_METADATA_DIR}")

    for path in mapping_files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        blocks = re.split(BLOCK_SPLIT_PATTERN, text)
        for block in (block.strip() for block in blocks if block.strip()):
            transcript_id = _extract_transcript_id(block)
            if not transcript_id or (wanted_ids is not None and transcript_id not in wanted_ids):
                continue

            voice_log_fields = _extract_voice_log_fields_from_last_line(block)
            if not voice_log_fields:
                continue
            if not voice_log_fields["extension_raw"] or not voice_log_fields["voice_trade_date_raw"]:
                continue

            row = {
                "transcript_id": transcript_id,
                **voice_log_fields,
            }

            existing_row = rows_by_transcript_id.get(transcript_id)
            if existing_row is not None:
                if existing_row != row:
                    conflicting_transcript_ids.add(transcript_id)
                continue

            rows_by_transcript_id[transcript_id] = row
            rows.append(row)

    if conflicting_transcript_ids:
        sample_ids = ", ".join(sorted(conflicting_transcript_ids)[:10])
        print(
            "Skipped conflicting transcript IDs in local voice logs: "
            f"{len(conflicting_transcript_ids)} conflicts "
            f"(examples: {sample_ids})"
        )

    return rows


def _load_voice_log_metadata(spark: SparkSession, transcript_ids: Optional[Set[str]]) -> DataFrame:
    """Convert local voice-log rows into a Spark DataFrame with normalized join keys."""
    rows = _load_local_voice_log_rows(transcript_ids)
    if not rows:
        raise RuntimeError(f"No voice log metadata rows found in: {VOICE_LOG_METADATA_DIR}")
    df = (
        spark.createDataFrame(rows, schema=VOICE_LOG_SCHEMA)
        .withColumn("extension_key", _extension_key_col("extension_raw"))
        .withColumn("trade_date_key", _date_key_col("voice_trade_date_raw"))
        .withColumn("voice_start_time_ts", _timestamp_col("voice_start_time_raw"))
        .dropna(subset=["transcript_id"]) # drop unmatched rows that can't be joined to transactions
        .select(
            "transcript_id",
            "extension_key",
            "voice_trade_date_raw",
            "trade_date_key",
            "phone_number_vl",
            "dialed_in_number",
            "direction",
            "voice_start_time_raw",
            "voice_stop_time_raw",
            "voice_duration_raw",
            "voice_start_time_ts",
        )
    )
    print(f"Loaded {df.count()} unique voice log metadata rows from local files in: {VOICE_LOG_METADATA_DIR}")
    return df


def _load_transaction_candidates(spark: SparkSession, join_keys_df: DataFrame) -> DataFrame:
    """Load candidate transaction rows for the voice-log (extension,date) keys so time matching can resolve them."""
    normalized_join_keys = join_keys_df.dropna().dropDuplicates()
    if not normalized_join_keys.head(1):
        return normalized_join_keys.select(
            "extension_key",
            "trade_date_key",
            *[F.lit(None).cast("string").alias(field_name) for field_name in TRANSACTION_METADATA_FIELDS],
            F.lit(None).cast("timestamp").alias("trade_datetime_ts"),
        ).limit(0)

    df = spark.table(TRANSACTION_HIVE_TABLE)
    _require_columns(df, TRANSACTION_HIVE_TABLE, [source for source, _ in TRANSACTION_COLUMN_ALIASES])

    # narrow the df to trade_date in voice logs
    trade_date_filters = [
        row["trade_date_str"]
        for row in (
            normalized_join_keys.select(F.date_format("trade_date_key", "yyyy-MM-dd").alias("trade_date_str"))
            .dropna()
            .dropDuplicates()
            .collect()
        )
    ]
    if trade_date_filters:
        df = df.filter(F.col("trade_date").cast("string").isin(trade_date_filters))

    selected = (
        df.select(*[_string_col(source).alias(alias) for source, alias in TRANSACTION_COLUMN_ALIASES])
        .withColumn("extension_key", _extension_key_col("extension_raw")) # extension_raw is phone_ext in transaction table
        .withColumn("trade_date_key", _date_key_col("trade_date_raw"))
        .withColumn("trade_datetime_ts", _timestamp_col("trade_datetime"))
        .join(F.broadcast(normalized_join_keys), on=["extension_key", "trade_date_key"], how="inner")
    )

    return selected.select(
        "extension_key",
        "trade_date_key",
        *TRANSACTION_METADATA_FIELDS,
        "trade_datetime_ts",
    ).dropDuplicates()


def _attach_best_transaction_metadata(voice_df: DataFrame, txn_df: DataFrame) -> DataFrame:
    """Join transaction candidates to each transcript and use start-time proximity to resolve collisions."""
    # create all possible matches between voice_df and txn_df
    joined = voice_df.alias("voice").join(
        txn_df.alias("txn"),
        on=(
            (F.col("voice.extension_key") == F.col("txn.extension_key"))
            & (F.col("voice.trade_date_key") == F.col("txn.trade_date_key"))
        ),
        how="left",
    )
    # filter on those extension_key which are not null: 1= available, 0= not available
    has_transaction_candidate = F.when(F.col("txn.extension_key").isNotNull(), F.lit(1)).otherwise(F.lit(0))
    # compare time difference between vl start time and transaction time
    trade_time_diff_seconds = F.when(
        F.col("voice.voice_start_time_ts").isNotNull() & F.col("txn.trade_datetime_ts").isNotNull(),
        F.abs(F.unix_timestamp(F.col("txn.trade_datetime_ts")) - F.unix_timestamp(F.col("voice.voice_start_time_ts"))),
    )
    has_time_comparison = F.when(trade_time_diff_seconds.isNotNull(), F.lit(1)).otherwise(F.lit(0))

    transcript_window = Window.partitionBy("voice.transcript_id")
    # rank the rows
    ranking_window = transcript_window.orderBy(
        has_transaction_candidate.desc(),
        has_time_comparison.desc(),
        trade_time_diff_seconds.asc_nulls_last(),
        F.col("txn.trade_datetime_ts").asc_nulls_last(),
    )

    ranked = (
        joined.withColumn("txn_candidate_count", F.sum(has_transaction_candidate).over(transcript_window))
        .withColumn("trade_time_diff_seconds", trade_time_diff_seconds)
        .withColumn("has_time_comparison", has_time_comparison)
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
        .withColumn(
            "tied_candidate_details",
            F.concat_ws(
                " | ",
                F.sort_array(
                    F.collect_set(
                        F.when(
                            (F.col("best_time_candidate_count") > 1)
                            & F.col("trade_time_diff_seconds").isNotNull()
                            & (F.col("trade_time_diff_seconds") == F.col("min_trade_time_diff_seconds")),
                            F.concat(
                                F.lit("op_code="),
                                F.coalesce(F.col("txn.op_code"), F.lit("<NULL>")),
                                F.lit(", trade_datetime="),
                                F.coalesce(F.col("txn.trade_datetime"), F.lit("<NULL>")),
                            ),
                        )
                    ).over(transcript_window)
                ),
            ),
        )
        .withColumn("candidate_rank", F.row_number().over(ranking_window))
    )

    return ranked.filter(F.col("candidate_rank") == 1).select(
        F.col("voice.transcript_id").alias("transcript_id"),
        F.col("voice.extension_key").alias("extension_key"),
        F.col("voice.voice_trade_date_raw").alias("voice_trade_date_raw"),
        F.col("voice.trade_date_key").alias("trade_date_key"),
        F.col("voice.phone_number_vl").alias("phone_number_vl"),
        F.col("voice.dialed_in_number").alias("dialed_in_number"),
        F.col("voice.direction").alias("direction"),
        F.col("voice.voice_start_time_raw").alias("voice_start_time_raw"),
        F.col("voice.voice_stop_time_raw").alias("voice_stop_time_raw"),
        F.col("voice.voice_duration_raw").alias("voice_duration_raw"),
        *[
            F.when(F.col("txn_match_count") == 1, F.col(f"txn.{field_name}")).alias(field_name)
            for field_name in TRANSACTION_METADATA_FIELDS
        ],
        "txn_match_count",
        "txn_candidate_count",
        "comparable_candidate_count",
        "best_time_candidate_count",
        "trade_time_diff_seconds",
        "tied_candidate_details",
    )


def _load_contact_metadata(spark: SparkSession, cif_keys_df: DataFrame) -> DataFrame:
    """Load contact rows for matched CIFs and pick one phone per CIF and biz date by priority."""
    normalized_cif_keys = (
        cif_keys_df.select(F.col("main_cif_no").alias("cif_number")).dropna().dropDuplicates()
    )
    if not normalized_cif_keys.head(1):
        return normalized_cif_keys.select(
            F.col("cif_number").alias("contact_cif_number"),
            F.lit(None).cast("string").alias("biz_dt"),
            F.lit(None).cast("date").alias("contact_date_key"),
            F.lit(None).cast("string").alias("client_phone_number"),
        )

    df = spark.table(CONTACT_HIVE_TABLE)
    _require_columns(df, CONTACT_HIVE_TABLE, ["cif_no", "phone_number", "biz_dt"])

    cif_filters = [row["cif_number"] for row in normalized_cif_keys.select("cif_number").collect()]
    if cif_filters:
        df = df.filter(F.col("cif_no").cast("string").isin(cif_filters))

    selected = (
        df.select(
            _string_col("cif_no").alias("cif_number"),
            _string_col("phone_number").alias("client_phone_number"),
            _string_col("biz_dt").alias("contact_date_raw"),
        )
        .withColumn("contact_date_key", _date_key_col("contact_date_raw"))
        .withColumn("client_phone_digits", _digits_only_col("client_phone_number"))
        .withColumn("normalized_phone_length", F.length(F.coalesce(F.col("client_phone_digits"), F.lit(""))))
        .withColumn(
            "starts_with_plus",
            F.when(
                F.col("client_phone_number").isNotNull() & F.col("client_phone_number").startswith("+"),
                F.lit(1),
            ).otherwise(F.lit(0)),
        )
        .join(F.broadcast(normalized_cif_keys), on="cif_number", how="inner")
    )

    ranking_window = Window.partitionBy("cif_number", "contact_date_key").orderBy(
        F.col("normalized_phone_length").desc(),
        F.col("starts_with_plus").desc(),
    )

    chosen = selected.withColumn("phone_priority_rank", F.row_number().over(ranking_window)).filter(
        F.col("phone_priority_rank") == 1
    )

    return chosen.select(
        F.col("cif_number").alias("contact_cif_number"),
        F.col("contact_date_raw").alias("biz_dt"),
        "contact_date_key",
        "client_phone_number",
    )


def _attach_latest_contact_metadata(base_df: DataFrame, contact_df: DataFrame) -> DataFrame:
    """Join contacts on CIF and keep the latest biz date per transcript."""
    joined = base_df.alias("base").join(
        contact_df.alias("contact"),
        on=F.col("base.main_cif_no") == F.col("contact.contact_cif_number"),
        how="left",
    )

    ranking_window = Window.partitionBy("base.transcript_id").orderBy(
        F.col("contact.contact_date_key").desc_nulls_last()
    )
    ranked = joined.withColumn("contact_row_rank", F.row_number().over(ranking_window))
    base_columns = [F.col(f"base.{column_name}").alias(column_name) for column_name in base_df.columns]

    return ranked.filter(F.col("contact_row_rank") == 1).select(
        *base_columns,
        F.col("contact.biz_dt").alias("biz_dt"),
        F.col("contact.client_phone_number").alias("client_phone_number"),
    )


def _load_description_metadata(spark: SparkSession, join_keys_df: DataFrame) -> DataFrame:
    """Load PTCC descriptions for matched transaction codes and keep code-bearing rows for multi-match cases."""
    normalized_join_keys = join_keys_df.select("op_code").dropna().dropDuplicates()
    if not normalized_join_keys.head(1):
        return normalized_join_keys.select(
            "op_code",
            F.lit(None).cast("string").alias("ptcc_description"),
        ).limit(0)

    df = spark.table(DESCRIPTION_HIVE_TABLE)
    _require_columns(df, DESCRIPTION_HIVE_TABLE, ["transaction_code", "description"])

    op_code_filters = [row["op_code"] for row in normalized_join_keys.select("op_code").collect()]
    if op_code_filters:
        df = df.filter(F.col("transaction_code").cast("string").isin(op_code_filters))

    selected = (
        df.select(
            _string_col("transaction_code").alias("op_code"),
            _string_col("description").alias("ptcc_description"),
        )
        .join(F.broadcast(normalized_join_keys), on=["op_code"], how="inner")
        .withColumn("description_has_code", F.lower(F.coalesce(F.col("ptcc_description"), F.lit(""))).rlike("6010|6011|6020"))
    )

    grouped = selected.groupBy("op_code").agg(
        F.count("*").alias("ptcc_row_count"),
        F.sum(F.when(F.col("description_has_code"), F.lit(1)).otherwise(F.lit(0))).alias("coded_row_count"),
    )

    filtered = (
        selected.join(F.broadcast(grouped), on="op_code", how="left")
        .filter((F.col("ptcc_row_count") == 1) | F.col("description_has_code"))
    )

    aggregated = filtered.groupBy("op_code").agg(
        F.count("*").alias("ptcc_match_count"),
        F.sort_array(F.collect_list("ptcc_description")).alias("ptcc_description"),
    )

    ambiguous_rows = aggregated.filter(F.col("ptcc_match_count") > 1).select(
        "op_code",
        "ptcc_match_count",
        "ptcc_description",
    ).collect()
    if ambiguous_rows:
        print(f"Total op_code values with ptcc_match_count >= 2: {len(ambiguous_rows)}")
        print("PTCC description metadata kept multiple rows for these op_code values:")
        for row in ambiguous_rows:
            print(
                f"  op_code={row['op_code']}, "
                f"ptcc_match_count={row['ptcc_match_count']}, "
                f"ptcc_description={row['ptcc_description']}"
            )

    return aggregated.select(
        "op_code",
        "ptcc_description",
    )


def _attach_description_metadata(base_df: DataFrame, description_df: DataFrame) -> DataFrame:
    """Join PTCC descriptions onto each transcript row by op_code."""
    joined = base_df.alias("base").join(
        description_df.alias("ptcc"),
        on=F.col("base.op_code") == F.col("ptcc.op_code"),
        how="left",
    )
    base_columns = [F.col(f"base.{column_name}").alias(column_name) for column_name in base_df.columns]
    return joined.select(
        *base_columns,
        F.col("ptcc.ptcc_description").alias("ptcc_description"),
    )


def load_client_metadata_from_hive(transcript_ids: Optional[Set[str]] = None) -> Dict[str, Dict[str, Any]]:
    """Build transcript metadata by combining local voice-log rows with transaction and contact Hive tables."""
    print(
        "Loading voice-log metadata from local directory "
        f"{VOICE_LOG_METADATA_DIR} and enriching from Hive tables: "
        f"{TRANSACTION_HIVE_TABLE}, {CONTACT_HIVE_TABLE}, {DESCRIPTION_HIVE_TABLE}"
    )
    print("Hive access mode is read-only: this module only reads with spark.table(...) and never writes.")

    spark = SparkSession.builder.appName(SPARK_APP_NAME).enableHiveSupport().getOrCreate()
    try:
        voice_df = _load_voice_log_metadata(spark, transcript_ids)
        txn_join_keys = voice_df.select("extension_key", "trade_date_key").dropna().dropDuplicates()
        txn_df = _load_transaction_candidates(spark, txn_join_keys)
        base_df = _attach_best_transaction_metadata(voice_df, txn_df)
        contact_cif_keys = base_df.select("main_cif_no").dropna().dropDuplicates()
        contact_df = _load_contact_metadata(spark, contact_cif_keys)
        base_with_contact_df = _attach_latest_contact_metadata(base_df, contact_df)
        description_join_keys = base_df.select("op_code").dropna().dropDuplicates()
        description_df = _load_description_metadata(spark, description_join_keys)
        joined = _attach_description_metadata(base_with_contact_df, description_df)
        rows = joined.collect()
    finally:
        spark.stop()

    lookup: Dict[str, Dict[str, Any]] = {}
    transaction_ambiguous = 0
    no_transaction_match = 0
    ambiguous_transcripts: list[tuple[str, Any, Any, Any, str, Any, Any]] = []

    for row in rows:
        row_dict = row.asDict(recursive=True)
        transcript_id = str(row_dict["transcript_id"]).strip() if row_dict["transcript_id"] is not None else ""
        if not transcript_id:
            continue

        if (row_dict["txn_candidate_count"] or 0) == 0:
            no_transaction_match += 1

        if (row_dict["txn_match_count"] or 0) > 1:
            transaction_ambiguous += 1
            if (row_dict["comparable_candidate_count"] or 0) == 0:
                reason = "multiple extension/date matches, but no candidate trade_datetime available"
            elif (row_dict["best_time_candidate_count"] or 0) > 1:
                reason = "multiple candidates tied on closest trade_datetime"
            else:
                reason = "multiple candidates remain after time comparison"
            ambiguous_transcripts.append(
                (
                    transcript_id,
                    row_dict.get("extension_key"),
                    row_dict.get("voice_trade_date_raw") or row_dict.get("trade_date_key"),
                    row_dict.get("voice_start_time_raw"),
                    reason,
                    row_dict.get("best_time_candidate_count"),
                    row_dict.get("tied_candidate_details"),
                )
            )

        trade_date = pd.to_datetime(
            row_dict.get("trade_date_raw") or row_dict.get("voice_trade_date_raw"),
            dayfirst=True,
            errors="coerce",
        )
        input_date_only = pd.to_datetime(row_dict.get("input_date_raw"), errors="coerce")

        lookup[transcript_id] = {
            "client_name": str(row_dict["order_person_spoken_to"]).strip()
            if row_dict["order_person_spoken_to"] is not None
            else None,
            "rm_name": str(row_dict["rm_name"]).strip() if row_dict["rm_name"] is not None else None,
            "trade_date": trade_date.normalize() if pd.notna(trade_date) else None,
            "input_date_only": input_date_only.normalize() if pd.notna(input_date_only) else None,
            "client_phone_number": row_dict["client_phone_number"],
            "biz_dt": row_dict["biz_dt"],
            "phone_number_vl": row_dict["phone_number_vl"],
            "dialed_in_number": row_dict["dialed_in_number"],
            "direction": row_dict["direction"],
            "voice_start_time": row_dict["voice_start_time_raw"],
            "voice_stop_time": row_dict["voice_stop_time_raw"],
            "voice_duration": row_dict["voice_duration_raw"],
            "main_cif_no": row_dict["main_cif_no"],
            "op_code": row_dict["op_code"],
            "trade_date_raw": row_dict["trade_date_raw"],
            "input_date": row_dict["input_date_raw"],
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
        }

    if ambiguous_transcripts:
        print("Skipped transcript transaction metadata due to unresolved time-based transaction ambiguity:")
        for (
            transcript_id,
            extension_key,
            trade_date_value,
            start_time_value,
            reason,
            best_time_candidate_count,
            tied_candidate_details,
        ) in ambiguous_transcripts:
            print(
                f"  transcript_id={transcript_id}, "
                f"extension={extension_key}, "
                f"trade_date={trade_date_value}, "
                f"start_time={start_time_value}, "
                f"reason={reason}, "
                f"best_time_candidate_count={best_time_candidate_count}, "
                f"tied_candidates={tied_candidate_details}"
            )

    print(
        "Hive metadata summary: "
        f"loaded={len(lookup)} "
        f"transaction_no_match={no_transaction_match} "
        f"transaction_ambiguous={transaction_ambiguous}"
    )
    return lookup


def save_client_metadata_to_file(
    metadata_lookup: Dict[str, Dict[str, Any]],
    output_path: Path = METADATA_OUTPUT_PATH,
) -> Path:
    """Write the merged metadata lookup to a local JSON file for later inference runs."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        transcript_id: {
            key: value.isoformat() if isinstance(value, pd.Timestamp) else value
            for key, value in metadata.items()
        }
        for transcript_id, metadata in metadata_lookup.items()
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
    print(f"Saved merged metadata to: {output_path}")
    return output_path


def _parse_normalized_timestamp(value: Any) -> Optional[pd.Timestamp]:
    """Parse a serialized timestamp field back into a normalized pandas Timestamp."""
    if value in (None, ""):
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.normalize()


def load_client_metadata_from_file(
    transcript_ids: Optional[Set[str]] = None,
    input_path: Path = METADATA_OUTPUT_PATH,
) -> Dict[str, Dict[str, Any]]:
    """Load the pre-joined local metadata file produced by this script."""
    if not input_path.exists():
        raise FileNotFoundError(
            f"Merged metadata file not found: {input_path}. Run metadata_joiner_hive.py first."
        )

    with input_path.open("r", encoding="utf-8") as handle:
        raw_lookup = json.load(handle)

    wanted_ids = None
    if transcript_ids is not None:
        wanted_ids = {
            str(transcript_id).strip() for transcript_id in transcript_ids if str(transcript_id).strip()
        }

    lookup: Dict[str, Dict[str, Any]] = {}
    for transcript_id, metadata in raw_lookup.items():
        transcript_id_str = str(transcript_id).strip()
        if not transcript_id_str:
            continue
        if wanted_ids is not None and transcript_id_str not in wanted_ids:
            continue

        entry = dict(metadata)
        entry["trade_date"] = _parse_normalized_timestamp(entry.get("trade_date"))
        entry["input_date_only"] = _parse_normalized_timestamp(entry.get("input_date_only"))
        lookup[transcript_id_str] = entry

    print(f"Loaded merged metadata from: {input_path} ({len(lookup)} transcripts)")
    return lookup


def main() -> None:
    """Run the Hive join once and save the merged transcript metadata locally."""
    metadata_lookup = load_client_metadata_from_hive()
    save_client_metadata_to_file(metadata_lookup)


if __name__ == "__main__":
    main()
