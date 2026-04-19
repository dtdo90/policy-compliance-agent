import re
import ntpath
from pathlib import Path
import pandas as pd

from ..core.config import load_config
from ..core.paths import resolve_project_path


CONFIG = load_config()
METADATA_SETTINGS = CONFIG.get("metadata", {})

# =========================
# CONFIG
# =========================
VOICE_LOGS_DIR = resolve_project_path(METADATA_SETTINGS.get("voice_log_metadata_dir", "SG_voice_logs"))
EXCEL_PATH = resolve_project_path(METADATA_SETTINGS.get("excel_path", "SG VL metadata (updated).xlsx"))

# Primary metadata columns
EXT_COL = "Extension (True after BAU)"
DT_COL = "Inputted Client Order/ Callback Date/Time"
TRADE_DT_COL = "Trade Date"
VL_REVIEWED_COL = "VL reviewed  (True after BAU)"

# Fallback metadata column for unmatched logic
INPUTTED_PHONE_EXT_COL = "Inputted Phone/Ext"

OUTPUT_CSV_WITH_FIELDS = EXCEL_PATH.with_name(EXCEL_PATH.stem + "_merged_excel.csv")
OUTPUT_AMBIGUOUS_CSV = resolve_project_path("data/results/ambiguous_transcript_ids.csv")
OUTPUT_UNMATCHED_CSV = resolve_project_path("data/results/unmatched_transcript_ids.csv")
OUTPUT_AMBIGUOUS_MULTIPLE_MATCHING_ROWS_CSV = resolve_project_path("data/results/ambiguous_multile_matching_rows.csv")
OUTPUT_AMBIGUOUS_MULTIPLE_IDS_CSV = resolve_project_path("data/results/ambiguous_multiple_rows_ids.csv")

# Separator must be >= 100 dashes
BLOCK_SPLIT_PATTERN = r"\n\s*-{100,}\s*\n"

# Regex
DATE_RE = re.compile(r"\b(\d{2}/\d{2}/\d{4})\b")
EXT_TOKEN_RE = re.compile(r"(\+\d{7,}|\b\d{5}\b)")
CALLID_RE = re.compile(r"^\d{15,}\b")


# =========================
# NORMALIZERS (CSV-safe)
# =========================
def normalize_digits(x) -> str | None:
    """
    Digits-only key.
    CSV-safe: avoids turning "83011.0" into "830110".
    """
    if pd.isna(x):
        return None

    if isinstance(x, int):
        return str(x)

    if isinstance(x, float):
        if x.is_integer():
            return str(int(x))
        x = str(x)

    s = str(x).strip()

    # common CSV artifact as string
    if re.fullmatch(r"\d+\.0", s):
        return s.split(".")[0]

    digits = re.sub(r"\D+", "", s)
    return digits if digits else None


def normalize_ddmmyyyy(x) -> str | None:
    """
    Parse date-like values and return dd/mm/yyyy.
    Handles dd/mm/yyyy, dd-mm-yyyy, yyyy-mm-dd, etc.
    """
    if pd.isna(x):
        return None

    if isinstance(x, pd.Timestamp):
        return x.strftime("%d/%m/%Y")

    s = str(x).strip()

    # fast path dd/mm/yyyy
    m = DATE_RE.search(s)
    if m:
        return m.group(1)

    # fallback parse
    dt = pd.to_datetime(s, errors="coerce", dayfirst=True)
    if pd.isna(dt):
        return None
    return dt.strftime("%d/%m/%Y")


def normalize_vl_reviewed_date_fuzzy(x) -> str | None:
    """
    Extract a date from VL reviewed column and return dd/mm/yyyy, including compact forms.

    Examples:
      - "x83054, 02/08/2024 5:55:00" -> 02/08/2024
      - "x83017 300622 1711" -> 30/06/2022   (ddmmyy)
      - "30062022" -> 30/06/2022            (ddmmyyyy)
      - "20220630" -> 30/06/2022            (yyyymmdd)
    """
    if pd.isna(x):
        return None
    s = str(x).strip()

    # 1) direct dd/mm/yyyy
    m = DATE_RE.search(s)
    if m:
        return m.group(1)

    # 2) parseable by pandas (dd-mm-yyyy, yyyy-mm-dd, etc.)
    dt = pd.to_datetime(s, errors="coerce", dayfirst=True)
    if pd.notna(dt):
        return dt.strftime("%d/%m/%Y")

    # 3) compact digit patterns inside messy strings
    digits = re.sub(r"\D+", "", s)
    if not digits:
        return None

    def try_parse(ds: str, fmt: str) -> str | None:
        dtx = pd.to_datetime(ds, format=fmt, errors="coerce")
        if pd.isna(dtx):
            return None
        return dtx.strftime("%d/%m/%Y")

    # Prefer 8-digit candidates first (sliding window)
    for i in range(0, len(digits) - 8 + 1):
        sub8 = digits[i : i + 8]
        # ddmmyyyy
        out = try_parse(sub8, "%d%m%Y")
        if out:
            return out
        # yyyymmdd
        out = try_parse(sub8, "%Y%m%d")
        if out:
            return out

    # Then 6-digit ddmmyy candidates
    for i in range(0, len(digits) - 6 + 1):
        sub6 = digits[i : i + 6]
        dd = int(sub6[0:2])
        mm = int(sub6[2:4])
        yy = int(sub6[4:6])
        if not (1 <= dd <= 31 and 1 <= mm <= 12):
            continue
        yyyy = 1900 + yy if yy >= 70 else 2000 + yy
        out = try_parse(f"{dd:02d}{mm:02d}{yyyy:04d}", "%d%m%Y")
        if out:
            return out

    return None


def split_blocks(text: str) -> list[str]:
    blocks = re.split(BLOCK_SPLIT_PATTERN, text)
    return [b.strip() for b in blocks if b.strip()]


# =========================
# PARSERS (VOICE LOGS)
# =========================
def extract_transcript_id(block: str) -> str | None:
    """
    Extract transcript id from line like:
      File name:  C:\\...\\magnet_pool\\_1_1_235.wav
    Return: _1_1_235 (no .wav)
    """
    for ln in block.splitlines():
        ln_s = ln.strip()
        if ln_s.lower().startswith("file name"):
            m = re.search(r"(\S+?\.wav)\b", ln_s, flags=re.IGNORECASE)
            if not m:
                return None
            wav_path = m.group(1)
            wav_base = ntpath.basename(wav_path)  # windows path safe on linux/mac
            return re.sub(r"\.wav$", "", wav_base, flags=re.IGNORECASE)
    return None


def extract_call_fields(last: str) -> tuple[str | None, str | None, str | None, str | None]:
    """
    Tail tokens of last line:
    <call_id>  ... <phone_number> <dialed_in_number> <direction> 83 -1
    We want: call_id= 1st token, phone_number = last 5th, dialed = last 4th, direction = last 3rd.
    """
    toks = last.split()
    if len(toks) < 6:
        return None, None, None, None
    call_id= toks[0]
    direction= toks[-3]
    dialed_in_number = toks[-4]
    phone_number_vl = toks[-5]
    if direction.lower()=="internal":
        phone_number_vl = None
    return call_id, phone_number_vl, dialed_in_number, direction


def extract_callid_extension_date_and_call_fields_from_last_line(
    block: str, *, debug_id: str = ""
) -> tuple[str | None, str | None, str | None, str | None, str | None, str | None]:
    """
    From last non-empty line:
      - starts with call_id (>=15 digits)
      - date is first dd/mm/yyyy
      - extension is first token matching +digits OR 5 digits
      - call fields from last tokens (phone/dialed/direction)
    """
    lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
    if not lines:
        print(f"[ERROR] Empty block. {debug_id}")
        return None, None, None, None, None

    last = lines[-1]

    if not CALLID_RE.search(last):
        print(f"[ERROR] Last line does not start with call-id (>=15 digits). {debug_id}\nLast line: {last}")
        return None, None, None, None, None, None

    m_date = DATE_RE.search(last)
    if not m_date:
        print(f"[ERROR] No dd/mm/yyyy found in last line. {debug_id}\nLast line: {last}")
        return None, None, None, None, None, None
    date_ddmmyyyy = m_date.group(1)

    m_ext = EXT_TOKEN_RE.search(last)
    if not m_ext:
        print(f"[ERROR] No extension token found in last line. {debug_id}\nLast line: {last}")
        return None, None, None, None, None, None

    token = m_ext.group(1)
    if token.startswith("+"):
        digits = re.sub(r"\D+", "", token)
        if len(digits) < 5:
            print(f"[ERROR] Phone token too short for extension. {debug_id}\nToken: {token}")
            return None,None, None, None, None, None
        extension = digits[-5:] # extension is last 5 digits 
    else:
        extension = token

    call_id, phone_number_vl, dialed_in_number, direction = extract_call_fields(last)
    return call_id, extension, date_ddmmyyyy, phone_number_vl, dialed_in_number, direction


LOOKUP_CANDIDATE_SPECS = [
    ("extension_key", "date_key"),
    ("extension_key", "trade_date_key"),
    ("extension_key", "vl_date_key"),
    ("inputted_phone_ext_key", "vl_date_fuzzy"),
]


def iter_lookup_candidates(row) -> list[tuple[str, str]]:
    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str]] = []

    for ext_col, date_col in LOOKUP_CANDIDATE_SPECS:
        ext = row.get(ext_col)
        date_value = row.get(date_col)
        if pd.isna(ext) or pd.isna(date_value):
            continue
        ext = str(ext).strip()
        date_value = str(date_value).strip()
        if not ext or not date_value:
            continue
        key = (ext, date_value)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)

    return out


def parse_voice_logs(dir_path: Path) -> pd.DataFrame:
    rows = []
    total_blocks = 0

    for path in sorted(dir_path.glob("*.txt")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        blocks = split_blocks(text)
        print(f"[INFO] {path.name}: {len(blocks)} blocks")
        total_blocks += len(blocks)

        for i, block in enumerate(blocks, start=1):
            transcript_id = extract_transcript_id(block)
            debug_id = f"(mapping_file={path.name}, block={i}, transcript_id={transcript_id})"

            if not transcript_id:
                print(f"[ERROR] Missing transcript_id. {debug_id}")
                continue

            call_id, extension, date_ddmmyyyy, phone_number, dialed_in_number, direction = \
                extract_callid_extension_date_and_call_fields_from_last_line(block, debug_id=debug_id)

            if not extension or not date_ddmmyyyy:
                continue

            rows.append({
                "call_id": call_id,
                "transcript_id": transcript_id,
                "extension": extension,
                "call_date": date_ddmmyyyy,
                "phone_number_vl": phone_number,
                "dialed_in_number": dialed_in_number,
                "direction": direction,
                "source_mapping_file": path.name,
            })

    print(f"[INFO] Total blocks across all files: {total_blocks}")
    return pd.DataFrame(rows)


# =========================
# MAIN
# =========================
def main():
    # 1) Parse mapping files
    df_map = parse_voice_logs(VOICE_LOGS_DIR)
    if df_map.empty:
        raise RuntimeError(f"No mappings found in: {VOICE_LOGS_DIR}")

    # 2) Ambiguous keys (extension,call_date)
    key_cols = ["extension", "call_date"]
    key_counts = df_map.groupby(key_cols)["transcript_id"].nunique()

    ambiguous_keys = key_counts[key_counts > 1].reset_index()[key_cols]
    df_ambiguous = df_map.merge(ambiguous_keys, on=key_cols, how="inner")

    if not df_ambiguous.empty:
        df_ambiguous.sort_values(["call_id", "transcript_id", "extension", "call_date", "source_mapping_file"]).to_csv(
            OUTPUT_AMBIGUOUS_CSV, index=False, encoding="utf-8-sig"
        )
        print(f"[WARN] Saved ambiguous transcript_ids: {OUTPUT_AMBIGUOUS_CSV} (rows={len(df_ambiguous)})")
    else:
        print("[INFO] No ambiguous (extension, call_date) keys found.")

    # Unique-only mapping
    unique_keys = key_counts[key_counts == 1].reset_index()[key_cols]
    df_unique = df_map.merge(unique_keys, on=key_cols, how="inner").copy()
    print(f"[INFO] Unique (extension, call_date) keys for mapping: {len(df_unique)} rows")
    df_unique = df_unique.sort_values(["extension", "call_date", "source_mapping_file", "transcript_id"]) \
                         .drop_duplicates(subset=key_cols, keep="first")
    print(f"[INFO] After deduplication of unique keys: {len(df_unique)} rows")

    # 3) Read metadata excel (dtype=str avoids float artifacts)
    df_csv = pd.read_excel(EXCEL_PATH, dtype=str, sheet_name="500 VL Samples")

    required_cols = [EXT_COL, DT_COL, TRADE_DT_COL, VL_REVIEWED_COL, INPUTTED_PHONE_EXT_COL]
    for col in required_cols:
        if col not in df_csv.columns:
            raise ValueError(f"Metadata missing column: {col}")

    df_csv_out = df_csv.copy()

    # Primary keys used for writeback
    df_csv_out["extension_key"] = df_csv_out[EXT_COL].map(normalize_digits)
    df_csv_out["date_key"] = df_csv_out[DT_COL].map(normalize_ddmmyyyy)
    df_csv_out["trade_date_key"] = df_csv_out[TRADE_DT_COL].map(normalize_ddmmyyyy)
    df_csv_out["vl_date_key"] = df_csv_out[VL_REVIEWED_COL].map(normalize_ddmmyyyy)

    # Extra keys for UNMATCHED fallback only
    df_csv_out["inputted_phone_ext_key"] = df_csv_out[INPUTTED_PHONE_EXT_COL].map(normalize_digits)
    df_csv_out["vl_date_fuzzy"] = df_csv_out[VL_REVIEWED_COL].map(normalize_vl_reviewed_date_fuzzy)

    # Build dict for unique mapping: (extension,call_date) -> fields
    key_to_row = df_unique.set_index(["extension", "call_date"])[
        ["call_id", "transcript_id", "direction", "dialed_in_number", "phone_number_vl", "source_mapping_file"]
    ].to_dict(orient="index")

    def lookup_fields(row):
        """
        Writeback logic:
          Try the same candidate keys used by unmatched detection, in priority order.
        matched call_date = first date that succeeded by that priority.
        """
        for ext, d in iter_lookup_candidates(row):
            rec = key_to_row.get((ext, d))
            if rec is not None:
                return (
                    rec.get("call_id", pd.NA),
                    rec.get("transcript_id", pd.NA),
                    ext,
                    d,
                    rec.get("direction", pd.NA),
                    rec.get("dialed_in_number", pd.NA),
                    rec.get("phone_number_vl", pd.NA),
                    rec.get("source_mapping_file", pd.NA),
                )

        return pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA

    # Apply writeback
    fields = df_csv_out.apply(lookup_fields, axis=1, result_type="expand")
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
    fields = fields.rename(columns={"matched_date": "call_date"})

    for c in fields.columns:
        df_csv_out[c] = fields[c]

    # If one unique voice-log mapping key (extension, call_date) is assigned to multiple CSV rows,
    # treat those rows as ambiguous and remove the mapping from the final writeback output.
    mapped_cols = [
        "call_id",
        "transcript_id",
        "matched_extension",
        "call_date",
        "direction",
        "dialed_in_number",
        "phone_number_vl",
        "source_mapping_file",
    ]
    candidate_match_mask = (
        df_csv_out["transcript_id"].notna()
        & df_csv_out["matched_extension"].notna()
        & df_csv_out["call_date"].notna()
    )
    duplicate_match_mask = candidate_match_mask & df_csv_out.duplicated(
        subset=["matched_extension", "call_date"], keep=False
    )
    ambiguous_multiple_matching_rows = df_csv_out[duplicate_match_mask].copy()

    if not ambiguous_multiple_matching_rows.empty:
        ambiguous_multiple_keys = (
            ambiguous_multiple_matching_rows[["matched_extension", "call_date"]]
            .rename(columns={"matched_extension": "extension"})
            .drop_duplicates()
        )
        ambiguous_multiple_ids = df_unique.merge(
            ambiguous_multiple_keys,
            on=["extension", "call_date"],
            how="inner",
        )
        cols = ["call_id", "transcript_id", "extension", "call_date", "source_mapping_file"]
        cols = [c for c in cols if c in ambiguous_multiple_ids.columns]
        ambiguous_multiple_ids.sort_values(
            ["extension", "call_date", "transcript_id"]
        ).to_csv(
            OUTPUT_AMBIGUOUS_MULTIPLE_IDS_CSV,
            index=False,
            encoding="utf-8-sig",
            columns=cols,
        )

        ambiguous_multiple_matching_rows.sort_values(
            ["transcript_id", "call_id"]
        ).to_csv(
            OUTPUT_AMBIGUOUS_MULTIPLE_MATCHING_ROWS_CSV, index=False, encoding="utf-8-sig"
        )
        df_csv_out.loc[duplicate_match_mask, mapped_cols] = pd.NA
        print(
            f"[WARN] Saved ambiguous multiple-matching rows: "
            f"{OUTPUT_AMBIGUOUS_MULTIPLE_MATCHING_ROWS_CSV} "
            f"(rows={len(ambiguous_multiple_matching_rows)})"
        )
        print(
            f"[WARN] Saved ambiguous multiple-matching transcript_ids: "
            f"{OUTPUT_AMBIGUOUS_MULTIPLE_IDS_CSV} "
            f"(rows={len(ambiguous_multiple_ids)})"
        )
    else:
        print("[INFO] No ambiguous CSV rows with duplicate voice-log key assignments found.")

    # =========================
    # Save UNMATCHED transcript_ids (voice logs side)
    # Fallback logic for determining "matched":
    #   - primary keys: (extension_key, date_key/trade/vl)
    #   - extra fallback: (inputted_phone_ext_key, vl_date_fuzzy)
    # =========================
    final_matched_mask = (
        df_csv_out["transcript_id"].notna()
        & df_csv_out["matched_extension"].notna()
        & df_csv_out["call_date"].notna()
    )
    accepted_matches = df_csv_out[final_matched_mask].copy()
    meta_keys = (
        accepted_matches[["matched_extension", "call_date"]]
        .rename(columns={"matched_extension": "extension"})
        .dropna()
        .drop_duplicates()
    )

    unmatched = df_unique.merge(meta_keys, on=["extension", "call_date"], how="left", indicator=True)
    unmatched = unmatched[unmatched["_merge"] == "left_only"].drop(columns=["_merge"])

    if not unmatched.empty:
        # Save found extension, call_date, and mapping file + other useful fields
        cols = [ "call_id", "transcript_id", "extension", "source_mapping_file" ]
        cols = [c for c in cols if c in unmatched.columns]
        unmatched.sort_values(["extension", "call_date", "transcript_id"]).to_csv(
            OUTPUT_UNMATCHED_CSV, index=False, encoding="utf-8-sig", columns=cols
        )
        print(f"[WARN] Saved unmatched transcript_ids: {OUTPUT_UNMATCHED_CSV} (rows={len(unmatched)})")
    else:
        print("[INFO] No unmatched transcript_ids (unique keys) found.")

    matched_output_rows = int(df_csv_out["transcript_id"].notna().sum())
    matched_unique_transcript_ids = int(len(meta_keys))
    print(
        "[INFO] Match summary: "
        f"unique_mapping_rows={len(df_unique)} "
        f"matched_unique_transcript_ids={matched_unique_transcript_ids} "
        f"matched_output_rows={matched_output_rows}"
    )

    # Drop helper columns before saving updated CSV
    df_csv_out = df_csv_out.drop(columns=[
        "extension_key", "date_key", "trade_date_key", "vl_date_key",
        "inputted_phone_ext_key", "vl_date_fuzzy", "matched_extension"
    ])

    df_csv_out.to_csv(OUTPUT_CSV_WITH_FIELDS, index=False, encoding="utf-8-sig")
    print(f"[INFO] Updated CSV written to: {OUTPUT_CSV_WITH_FIELDS}")

if __name__ == "__main__":
    main()
