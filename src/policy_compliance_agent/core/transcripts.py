"""Transcript parsing and chunking helpers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .paths import resolve_project_path

SPEAKER_LINE_RE = re.compile(r"SPEAKER_\d+\s+\[[\d.]+s\s*-\s*[\d.]+s\]:\s*(.*)")
STRUCTURED_SPEAKER_RE = re.compile(r"SPEAKER_(\d+)\s+\[([\d.]+)s\s*-\s*([\d.]+)s\]:\s*(.*)")
COLON_SPEAKER_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9 _/\-]{0,40})\s*:\s*(.*)$")


def chunk_text(text: str, max_chunk_words: int, chunk_overlap: int) -> list[str]:
    words = text.split()
    if len(words) <= max_chunk_words:
        return [text] if text else []

    chunks = []
    stride = max_chunk_words - chunk_overlap
    for index in range(0, len(words), stride):
        chunk = words[index : index + max_chunk_words]
        if len(chunk) >= 2 or index == 0:
            chunks.append(" ".join(chunk))
    return chunks


def _normalize_speaker_label(label: str) -> str:
    return " ".join(label.strip().rstrip(":").lower().split())


def extract_speaker_text(text: str, speaker_labels: list[str] | tuple[str, ...] | set[str]) -> str:
    targets = {_normalize_speaker_label(label) for label in speaker_labels if str(label).strip()}
    if not targets:
        return text.strip()

    parts: list[str] = []
    current_is_target = False
    current_buffer: list[str] = []

    def flush_current() -> None:
        nonlocal current_is_target, current_buffer
        if current_is_target and current_buffer:
            merged = " ".join(chunk.strip() for chunk in current_buffer if chunk.strip()).strip()
            if merged:
                parts.append(merged)
        current_is_target = False
        current_buffer = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        structured_match = STRUCTURED_SPEAKER_RE.match(line)
        if structured_match:
            flush_current()
            current_is_target = _normalize_speaker_label(f"speaker_{structured_match.group(1)}") in targets
            text_part = structured_match.group(4).strip()
            current_buffer = [text_part] if text_part else []
            continue

        speaker_match = COLON_SPEAKER_RE.match(line)
        if speaker_match:
            flush_current()
            current_is_target = _normalize_speaker_label(speaker_match.group(1)) in targets
            text_part = speaker_match.group(2).strip()
            current_buffer = [text_part] if text_part else []
            continue

        if current_buffer:
            current_buffer.append(line)

    flush_current()
    return " ".join(parts).strip()


def _load_plain_text_transcript(txt_file: Path) -> str:
    parts: list[str] = []
    with txt_file.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            match = SPEAKER_LINE_RE.match(line)
            if match:
                text = match.group(1).strip()
                if text:
                    parts.append(text)
            elif parts:
                parts.append(line)
            else:
                parts.append(line)
    # Keep speaker-turn boundaries so downstream speaker extraction can remove
    # customer/client turns before chunking.
    return "\n".join(parts).strip()


def load_transcripts_from_folder(voice_logs_dir: str | Path) -> list[dict[str, Any]]:
    folder = resolve_project_path(voice_logs_dir)
    if not folder.exists():
        raise FileNotFoundError(f"Directory not found: {folder}")

    results: list[dict[str, Any]] = []
    for txt_file in sorted(folder.glob("*.txt")):
        transcript = _load_plain_text_transcript(txt_file)
        if transcript:
            results.append({"transcript_id": txt_file.stem, "transcript": transcript})
    return results


def load_transcripts_structured_from_txt(voice_logs_dir: str | Path) -> list[dict[str, Any]]:
    folder = resolve_project_path(voice_logs_dir)
    if not folder.exists():
        raise FileNotFoundError(f"Directory not found: {folder}")

    transcripts: list[dict[str, Any]] = []
    for txt_file in sorted(folder.glob("*.txt")):
        conversation: dict[str, list[str]] = {"sca": [], "client": []}
        current_speaker: str | None = None
        current_buffer: list[str] = []

        def flush_current() -> None:
            nonlocal current_speaker, current_buffer
            if current_speaker and current_buffer:
                conversation.setdefault(current_speaker, []).append(" ".join(current_buffer).strip())
            current_speaker = None
            current_buffer = []

        with txt_file.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                match = STRUCTURED_SPEAKER_RE.match(line)
                if not match:
                    if current_speaker and current_buffer:
                        current_buffer.append(line)
                    continue

                speaker_num = int(match.group(1))
                text = match.group(4).strip()
                normalized = "sca" if speaker_num % 2 == 0 else "client"

                if current_speaker is None:
                    current_speaker = normalized
                    current_buffer = [text]
                elif current_speaker == normalized:
                    current_buffer.append(text)
                else:
                    flush_current()
                    current_speaker = normalized
                    current_buffer = [text]

        flush_current()
        if sum(len(turns) for turns in conversation.values()) > 0:
            transcripts.append({"transcript_id": txt_file.stem, "transcript": conversation})

    return transcripts
