"""Metadata loading helpers kept separate from semantic-only inference."""

from __future__ import annotations

from typing import Any


def _fuzzy_col(columns, keywords):
    for column in columns:
        column_text = str(column).lower()
        if all(keyword.lower() in column_text for keyword in keywords):
            return column
    return None


def load_client_metadata(filepath: str) -> dict[str, dict[str, Any]]:
    import pandas as pd

    df_raw = pd.read_csv(filepath)
    col_id = _fuzzy_col(df_raw.columns, ["transcript", "id"]) or "transcript_ids"
    col_client = _fuzzy_col(df_raw.columns, ["order_person", "spoken_to"]) or "order_person_spoken_to"
    col_rm = _fuzzy_col(df_raw.columns, ["rm", "name"]) or "rm name"
    col_trade = "excel_trade_date_raw" if "excel_trade_date_raw" in df_raw.columns else _fuzzy_col(df_raw.columns, ["trade", "date"])
    col_call = "call_date_vl" if "call_date_vl" in df_raw.columns else ("trade_date_vl" if "trade_date_vl" in df_raw.columns else None)
    col_client_phone = _fuzzy_col(df_raw.columns, ["client", "phone"]) or "client_phone_number"
    col_vl_phone = "phone_number_vl" if "phone_number_vl" in df_raw.columns else _fuzzy_col(df_raw.columns, ["phone", "vl"])

    lookup: dict[str, dict[str, Any]] = {}
    for _, row in df_raw.iterrows():
        transcript_id = str(row.get(col_id, "")).strip()
        if not transcript_id or transcript_id.lower() == "nan":
            continue

        clean_id = transcript_id.rsplit(".", 1)[0]
        trade_ts = pd.to_datetime(row.get(col_trade), dayfirst=True, errors="coerce")
        call_ts = pd.to_datetime(row.get(col_call), dayfirst=True, errors="coerce") if col_call else None
        lookup[clean_id] = {
            "client_name": str(row.get(col_client)).strip() if pd.notna(row.get(col_client)) else None,
            "rm_name": str(row.get(col_rm)).strip() if pd.notna(row.get(col_rm)) else None,
            "trade_date": trade_ts.normalize() if pd.notna(trade_ts) else None,
            "call_date_vl": call_ts.normalize() if call_ts is not None and pd.notna(call_ts) else None,
            "client_phone_number": row.get(col_client_phone) if pd.notna(row.get(col_client_phone)) else None,
            "phone_number_vl": row.get(col_vl_phone) if col_vl_phone and pd.notna(row.get(col_vl_phone)) else None,
            "direction": row.get("direction") if pd.notna(row.get("direction")) else None,
        }
    return lookup
