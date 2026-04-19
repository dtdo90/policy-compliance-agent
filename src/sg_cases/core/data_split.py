import csv
import random
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple

from .config import load_config
from .paths import resolve_project_path

CONFIG = load_config()
OUTPUT_SETTINGS = CONFIG.get("outputs", {})
DATA_SETTINGS = CONFIG.get("data", {})

CHINESE_CHAR_RE = re.compile(r"[\u3400-\u4DBF\u4E00-\u9FFF]")
EXCLUDE_CHINESE_GE=100

CSV_PATH = resolve_project_path(OUTPUT_SETTINGS.get("report_csv_path", "data/results/semantic_compliance_report.csv"))
VOICE_LOGS_DIR = resolve_project_path(DATA_SETTINGS.get("transcripts_txt_dir", "data/voice_logs"))

# Take 140 transcripts for training
TRAIN_SIZE=140
SEED=42


def _parse_pass_labels(value: str) -> Set[str]:
    raw = str(value or "").strip()
    if not raw or raw.upper() == "NONE":
        return set()
    return {part.strip() for part in raw.split(",") if part.strip() and part.strip().upper() != "NONE"}


def _scenario_sort_key(x: str):
    parts = str(x).split(".")
    if all(p.isdigit() for p in parts):
        return (0, tuple(int(p) for p in parts))
    return (1, (str(x),))


def _read_csv_entries(csv_path: Path) -> Tuple[List[Dict], str]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "transcript_id" not in reader.fieldnames:
            raise ValueError(f"'transcript_id' column not found in CSV: {csv_path}")
        compliant_col = "COMPLIANT" if "COMPLIANT" in reader.fieldnames else reader.fieldnames[-1]

        entries: List[Dict] = []
        seen = set()
        for row in reader:
            t_id = str(row.get("transcript_id", "")).strip()
            if not t_id or t_id in seen:
                continue
            seen.add(t_id)
            entries.append(
                {
                    "transcript_id": t_id,
                    "pass_labels": _parse_pass_labels(row.get(compliant_col, "")),
                }
            )
    return entries, compliant_col


def _count_chinese_chars(text: str) -> int:
    return len(CHINESE_CHAR_RE.findall(text))


def _copy_group(items: List[Dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for item in items:
        src_path = item["path"]
        shutil.copy2(src_path, out_dir / src_path.name)


def _score_counts(
    train_counts: Dict[str, int], target_counts: Dict[str, float], all_labels: List[str]
) -> float:
    score = 0.0
    for label in all_labels:
        expected = target_counts.get(label, 0.0)
        actual = float(train_counts.get(label, 0))
        denom = max(1.0, expected)  # normalize to avoid frequent labels dominating
        diff = (actual - expected) / denom
        score += diff * diff
    return score


def _stratified_split_indices(records: List[Dict], train_size: int, seed: int) -> Tuple[List[int], List[int], Dict]:
    n = len(records)
    train_n = max(0, min(train_size, n))
    idxs = list(range(n))
    if train_n == 0 or train_n == n:
        rng = random.Random(seed)
        rng.shuffle(idxs)
        return idxs[:train_n], idxs[train_n:], {"restarts": 0, "swaps": 0, "score": 0.0}

    all_labels = sorted({lab for rec in records for lab in rec["pass_labels"]}, key=_scenario_sort_key)
    if not all_labels:
        rng = random.Random(seed)
        rng.shuffle(idxs)
        return idxs[:train_n], idxs[train_n:], {"restarts": 0, "swaps": 0, "score": 0.0}

    total_counts = Counter(lab for rec in records for lab in rec["pass_labels"])
    target_counts = {lab: total_counts[lab] * train_n / n for lab in all_labels}

    def _optimize_once(rng: random.Random):
        order = idxs[:]
        rng.shuffle(order)
        train_idx = order[:train_n]
        test_idx = order[train_n:]
        train_counts = Counter()
        for i in train_idx:
            train_counts.update(records[i]["pass_labels"])

        current_score = _score_counts(train_counts, target_counts, all_labels)
        accepted_swaps = 0
        attempts = max(20000, 80 * n)

        for _ in range(attempts):
            if not train_idx or not test_idx:
                break
            ti_pos = rng.randrange(len(train_idx))
            te_pos = rng.randrange(len(test_idx))
            i = train_idx[ti_pos]
            j = test_idx[te_pos]
            rec_i_labels = records[i]["pass_labels"]
            rec_j_labels = records[j]["pass_labels"]
            changed_labels = rec_i_labels | rec_j_labels
            if not changed_labels:
                continue

            delta = 0.0
            for lab in changed_labels:
                before = train_counts.get(lab, 0)
                after = before - (1 if lab in rec_i_labels else 0) + (1 if lab in rec_j_labels else 0)
                expected = target_counts.get(lab, 0.0)
                denom = max(1.0, expected)
                delta += ((after - expected) / denom) ** 2 - ((before - expected) / denom) ** 2

            if delta < -1e-12:
                for lab in rec_i_labels:
                    train_counts[lab] -= 1
                    if train_counts[lab] <= 0:
                        train_counts.pop(lab, None)
                for lab in rec_j_labels:
                    train_counts[lab] += 1
                train_idx[ti_pos], test_idx[te_pos] = j, i
                current_score += delta
                accepted_swaps += 1

        return train_idx, test_idx, train_counts, current_score, accepted_swaps

    best = None
    restarts = 8
    for r in range(restarts):
        rng = random.Random(seed + r)
        candidate = _optimize_once(rng)
        if best is None or candidate[3] < best[3]:
            best = candidate

    assert best is not None
    train_idx, test_idx, train_counts, best_score, accepted_swaps = best
    diag = {
        "restarts": restarts,
        "swaps": accepted_swaps,
        "score": float(best_score),
        "train_counts": dict(train_counts),
        "target_counts": target_counts,
        "all_labels": all_labels,
    }
    return train_idx, test_idx, diag


def main() -> None:

    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV output file not found: {CSV_PATH}")
    if not VOICE_LOGS_DIR.exists() or not VOICE_LOGS_DIR.is_dir():
        raise FileNotFoundError(f"Voice logs folder not found: {VOICE_LOGS_DIR}")

    csv_entries, compliant_col = _read_csv_entries(CSV_PATH)
    eligible: List[Dict] = []
    excluded_chinese: List[Tuple[str, int]] = []
    missing_files: List[str] = []

    for entry in csv_entries:
        t_id = entry["transcript_id"]
        txt_path = VOICE_LOGS_DIR / f"{t_id}.txt"
        if not txt_path.exists():
            missing_files.append(t_id)
            continue

        text = txt_path.read_text(encoding="utf-8", errors="ignore")
        chinese_count = _count_chinese_chars(text)
        if chinese_count >= EXCLUDE_CHINESE_GE:
            excluded_chinese.append((t_id, chinese_count))
            continue

        eligible.append(
            {
                "transcript_id": t_id,
                "path": txt_path,
                "pass_labels": set(entry["pass_labels"]),
                "chinese_count": chinese_count,
            }
        )

    train_n = min(TRAIN_SIZE, len(eligible))
    train_idx, test_idx, strat_diag = _stratified_split_indices(eligible, train_n, SEED)
    train_items = [eligible[i] for i in train_idx]
    test_items = [eligible[i] for i in test_idx]

    train_dir = VOICE_LOGS_DIR / "train_data"
    test_dir = VOICE_LOGS_DIR / "test_data"

    _copy_group(train_items, train_dir)
    _copy_group(test_items, test_dir)

    all_labels = sorted({lab for rec in eligible for lab in rec["pass_labels"]}, key=_scenario_sort_key)
    total_counts = Counter(lab for rec in eligible for lab in rec["pass_labels"])
    train_counts = Counter(lab for rec in train_items for lab in rec["pass_labels"])
    test_counts = Counter(lab for rec in test_items for lab in rec["pass_labels"])

    print(f"CSV source: {CSV_PATH}")
    print(f"PASS label column used: {compliant_col}")
    print(f"Voice logs: {VOICE_LOGS_DIR}")
    print(f"Total transcript IDs in CSV: {len(csv_entries)}")
    print(f"Missing txt files: {len(missing_files)}")
    print(
        f"Excluded (Chinese chars >= {EXCLUDE_CHINESE_GE}): {len(excluded_chinese)}"
    )
    print(f"Eligible after filter: {len(eligible)}")
    print(f"Copied to train_data: {len(train_items)} -> {train_dir}")
    print(f"Copied to test_data: {len(test_items)} -> {test_dir}")
    if len(eligible) < TRAIN_SIZE:
        print(
            f"Warning: only {len(eligible)} eligible files available; "
            f"requested train_size={TRAIN_SIZE}."
        )
    if missing_files:
        print(f"Example missing transcript_ids: {missing_files[:10]}")
    if excluded_chinese:
        preview = ", ".join(f"{t}:{n}" for t, n in excluded_chinese[:10])
        print(f"Example excluded (id:count): {preview}")
    if all_labels and len(eligible) > 0 and len(train_items) > 0 and len(test_items) > 0:
        print(
            f"Stratification optimizer: restarts={strat_diag.get('restarts', 0)}, "
            f"accepted_swaps(best run)={strat_diag.get('swaps', 0)}, "
            f"score={strat_diag.get('score', 0.0):.4f}"
        )
        print("Per-scenario PASS rates (eligible/train/test):")
        for lab in all_labels:
            overall = total_counts[lab] / len(eligible)
            train_rate = train_counts[lab] / len(train_items)
            test_rate = test_counts[lab] / len(test_items)
            print(
                f"  {lab}: {overall:.3f} / {train_rate:.3f} / {test_rate:.3f} "
                f"(counts {total_counts[lab]}/{train_counts[lab]}/{test_counts[lab]})"
            )
    print(
        "Note: existing files in train_data/test_data are not deleted; copies overwrite same names."
    )


if __name__ == "__main__":
    main()
