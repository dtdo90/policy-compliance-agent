"""Ollama-backed synthetic data generation for SG compliance prompts.

This module follows the same generator -> judge pattern as the existing Qwen
generator, but calls Ollama's local HTTP API directly instead of loading a
model through Transformers.

The output is a dataset-preparation JSON list with rows shaped like:
{
  "disclaimer_id": "...",
  "scenario": "...",
  "scenario_name": "...",
  "prompt_index": 0,
  "type": "compliant" | "non-compliant",
  "dialogue": "<lowercased RM snippet>",
  "source_batch_size": 30,
  "kept_after_filter": 24,
  "supervisor_score": 9
}
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROMPTS_PATH = PROJECT_ROOT / "data" / "prompts" / "prompts_2_12_sg.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "results" / "SG_synthetic_data" / "dataset_2_12_ollama.json"
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "qwen3:8b"

GEN_WRAPPER_SYSTEM = """You must follow the user's instructions exactly.
Return ONLY the JSON requested. Do not add markdown, backticks, or any extra commentary outside the JSON.
"""

JUDGE_SYSTEM_TEMPLATE = """You are a strict banking compliance supervisor scoring a GENERATED JSON ARRAY of RM snippets.

You will be given:
- system_instruction (scenario instruction)
- user_prompt (task instruction)
- generated_output (raw model output, expected to be a JSON array)

Evaluate:
1) Output is a valid JSON array ONLY (no extra text).
2) Schema: each element is an object with exactly keys ["label","text"].
   - label is exactly "Compliant" or "Non-Compliant"
   - text is a single RM turn (no multi-speaker / no "Client:" etc.)
3) Length: "around 20 words": hard minimum {min_words} words per snippet.
   - Too-short snippets are a major issue.
   - Too-long snippets (> {max_words_soft} words) are minor but should be reduced.
4) Alignment to user_prompt intent (e.g., compliant vs hard negative near-miss patterns).

Return ONLY one JSON object with EXACT keys:
pass, score, violations, notes

Schema:
{{
  "pass": true/false,
  "score": 0-10,
  "violations": ["..."],
  "notes": "short fix-oriented notes"
}}

PASS requires: score >= 9 and no hard failures.
"""

THINK_RE = re.compile(r"<think>.*?</think>", flags=re.IGNORECASE | re.DOTALL)


def strip_think_blocks(text: str) -> str:
    if not text:
        return ""
    return THINK_RE.sub("", text).strip()


def safe_json_load(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def prompt_list_from_cfg(user_prompt: Any) -> list[str]:
    if isinstance(user_prompt, str) and user_prompt.strip():
        return [user_prompt]
    if isinstance(user_prompt, list):
        return [prompt for prompt in user_prompt if isinstance(prompt, str) and prompt.strip()]
    return []


def norm_label(label: Any) -> str:
    if not isinstance(label, str):
        return ""
    text = label.strip().lower().replace("_", "-").replace(" ", "-")
    if text == "compliant":
        return "compliant"
    if text in {"non-compliant", "noncompliant", "non-compliance", "noncompliance"} or text.startswith("non"):
        return "non-compliant"
    return text


def enforce_single_turn(text: str) -> str:
    if not isinstance(text, str):
        return ""
    output = text.strip()
    for prefix in ["rm:", "relationship manager:", "agent:", "advisor:"]:
        if output.lower().startswith(prefix):
            output = output[len(prefix) :].strip()
    return " ".join(output.splitlines()).strip()


def word_count(text: str) -> int:
    return len([word for word in str(text).strip().split() if word.strip()])


def extract_first_json_array(text: str) -> list[Any] | None:
    text = strip_think_blocks(text)
    if not text:
        return None
    text = re.sub(r"```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```", "", text).strip()

    if text.startswith("[") and text.endswith("]"):
        try:
            obj = json.loads(text)
            return obj if isinstance(obj, list) else None
        except Exception:
            pass

    start = text.find("[")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escaped = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    candidate = text[start : idx + 1]
                    try:
                        obj = json.loads(candidate)
                        return obj if isinstance(obj, list) else None
                    except Exception:
                        return None
    return None


def extract_judge_json(text: str) -> dict[str, Any] | None:
    text = strip_think_blocks(text)
    if not text:
        return None
    text = re.sub(r"```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```", "", text).strip()

    if text.startswith("{") and text.endswith("}"):
        try:
            obj = json.loads(text)
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass

    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escaped = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : idx + 1]
                    try:
                        obj = json.loads(candidate)
                        return obj if isinstance(obj, dict) else None
                    except Exception:
                        return None
    return None


def filter_snippet_array(
    arr: Any,
    *,
    min_words: int,
    max_words_soft: int,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    stats = {
        "input_is_array": isinstance(arr, list),
        "input_len": len(arr) if isinstance(arr, list) else 0,
        "kept": 0,
        "dropped_schema": 0,
        "dropped_label": 0,
        "dropped_multi_speaker": 0,
        "dropped_too_short": 0,
        "kept_too_long": 0,
        "min_words": min_words,
        "max_words_soft": max_words_soft,
    }

    kept: list[dict[str, str]] = []
    if not isinstance(arr, list):
        return kept, stats

    for item in arr:
        if not isinstance(item, dict) or set(item.keys()) != {"label", "text"}:
            stats["dropped_schema"] += 1
            continue

        label_raw = item.get("label")
        text_raw = item.get("text")
        if not isinstance(label_raw, str) or not isinstance(text_raw, str):
            stats["dropped_schema"] += 1
            continue

        if label_raw.strip() not in {"Compliant", "Non-Compliant"}:
            stats["dropped_label"] += 1
            continue

        text = enforce_single_turn(text_raw)
        lower = text.lower()
        if any(tag in lower for tag in ["client:", "rm:", "advisor:", "agent:"]):
            stats["dropped_multi_speaker"] += 1
            continue

        if word_count(text) < min_words:
            stats["dropped_too_short"] += 1
            continue

        if word_count(text) > max_words_soft:
            stats["kept_too_long"] += 1

        kept.append({"label": label_raw.strip(), "text": text})
        stats["kept"] += 1

    return kept, stats


def make_chat_input(system: str, user: str) -> dict[str, Any]:
    return {
        "model": "",
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "stream": False,
    }


def call_ollama_chat(
    *,
    model: str,
    messages: list[dict[str, str]],
    base_url: str,
    timeout: float,
    options: dict[str, Any] | None = None,
) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
    }
    if options:
        payload["options"] = options

    data = json.dumps(payload).encode("utf-8")
    request = Request(
        url=base_url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except HTTPError as exc:
        raise RuntimeError(f"Ollama HTTP error {exc.code}: {exc.read().decode('utf-8', errors='ignore')}") from exc
    except URLError as exc:
        raise RuntimeError(f"Failed to reach Ollama at {base_url}: {exc.reason}") from exc

    parsed = json.loads(raw)
    if isinstance(parsed, dict):
        if isinstance(parsed.get("message"), dict) and isinstance(parsed["message"].get("content"), str):
            return parsed["message"]["content"]
        if isinstance(parsed.get("response"), str):
            return parsed["response"]
    raise RuntimeError(f"Unexpected Ollama response shape: {raw[:500]}")


def build_generator_system(scenario_instruction: str) -> str:
    if isinstance(scenario_instruction, str) and scenario_instruction.strip():
        return GEN_WRAPPER_SYSTEM.strip() + "\n\n" + scenario_instruction.strip()
    return GEN_WRAPPER_SYSTEM.strip()


def build_generator_user(user_prompt: str, fix_hint: str | None, batch_size: int) -> str:
    prompt = (user_prompt or "").strip()
    prompt += (
        "\n\nReturn ONLY a JSON array with exactly "
        f"{int(batch_size)} items."
    )
    if fix_hint:
        prompt += "\n\nFIX INSTRUCTION (must follow): " + fix_hint.strip()
    return prompt


def build_judge_system(min_words: int, max_words_soft: int) -> str:
    return JUDGE_SYSTEM_TEMPLATE.format(min_words=min_words, max_words_soft=max_words_soft)


def build_judge_user(scenario_instruction: str, user_prompt: str, raw_output: str) -> str:
    return json.dumps(
        {
            "system_instruction": scenario_instruction,
            "user_prompt": user_prompt,
            "generated_output": raw_output,
        },
        ensure_ascii=False,
        indent=2,
    )


def judge_accept(judge: dict[str, Any]) -> bool:
    try:
        return bool(judge.get("pass")) and int(judge.get("score", 0)) >= 9
    except Exception:
        return False


def done_prompt_keys(dataset_rows: list[dict[str, Any]]) -> set[tuple[str, str, int]]:
    done: set[tuple[str, str, int]] = set()
    for row in dataset_rows:
        if not isinstance(row, dict):
            continue
        disclaimer_id = str(row.get("disclaimer_id", "")).strip().lower()
        scenario_name = str(row.get("scenario_name", "")).strip().lower()
        prompt_index = row.get("prompt_index")
        if disclaimer_id and scenario_name and isinstance(prompt_index, int):
            done.add((disclaimer_id, scenario_name, prompt_index))
    return done


def load_prompt_configs(prompts_path: Path) -> dict[str, Any]:
    data = safe_json_load(prompts_path, {})
    if not isinstance(data, dict) or not data:
        raise ValueError(f"Invalid or empty prompt config at {prompts_path}")
    return data


def generate_synthetic_data(
    *,
    prompts_path: Path = DEFAULT_PROMPTS_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    generator_model: str = DEFAULT_MODEL,
    judge_model: str | None = None,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    min_words: int = 18,
    max_words_soft: int = 90,
    min_items_to_accept_batch: int = 10,
    max_tries_per_prompt: int = 3,
    sleep_between_calls_sec: float = 0.2,
    seed: int = 42,
    batch_size: int = 30,
    generator_options: dict[str, Any] | None = None,
    judge_options: dict[str, Any] | None = None,
) -> Path:
    random.seed(seed)
    prompts = load_prompt_configs(prompts_path)
    existing = safe_json_load(output_path, [])
    dataset: list[dict[str, Any]] = existing if isinstance(existing, list) else []
    completed_keys = done_prompt_keys(dataset)
    judge_model = judge_model or generator_model
    judge_system = build_judge_system(min_words=min_words, max_words_soft=max_words_soft)

    for disclaimer_id, scenarios in prompts.items():
        if not isinstance(scenarios, dict):
            continue

        for scenario_name, scenario_cfg in scenarios.items():
            if scenario_name == "_qa" or not isinstance(scenario_cfg, dict):
                continue

            system_instruction = scenario_cfg.get("system_instruction", "")
            user_prompts = prompt_list_from_cfg(scenario_cfg.get("user_prompt", ""))
            if not isinstance(system_instruction, str) or not system_instruction.strip() or not user_prompts:
                continue

            disclaimer_key = str(disclaimer_id).strip().lower()
            scenario_key = str(scenario_name).strip().lower()

            for prompt_index, user_prompt in enumerate(user_prompts):
                key = (disclaimer_key, scenario_key, prompt_index)
                if key in completed_keys:
                    continue

                fix_hint = None
                best_score = -1
                best_stats: dict[str, Any] | None = None

                for attempt in range(1, max_tries_per_prompt + 1):
                    generator_messages = [
                        {"role": "system", "content": build_generator_system(system_instruction)},
                        {
                            "role": "user",
                            "content": build_generator_user(user_prompt, fix_hint, batch_size),
                        },
                    ]
                    raw_output = call_ollama_chat(
                        model=generator_model,
                        messages=generator_messages,
                        base_url=ollama_url,
                        timeout=600.0,
                        options=generator_options,
                    )

                    generated = extract_first_json_array(raw_output)
                    kept, stats = filter_snippet_array(
                        generated,
                        min_words=min_words,
                        max_words_soft=max_words_soft,
                    )

                    judge_messages = [
                        {"role": "system", "content": judge_system},
                        {
                            "role": "user",
                            "content": build_judge_user(system_instruction, user_prompt, raw_output),
                        },
                    ]
                    judge_output = call_ollama_chat(
                        model=judge_model,
                        messages=judge_messages,
                        base_url=ollama_url,
                        timeout=600.0,
                        options=judge_options,
                    )

                    judge = extract_judge_json(judge_output) or {
                        "pass": False,
                        "score": 0,
                        "violations": ["invalid_judge_json"],
                        "notes": "Supervisor output invalid JSON.",
                    }

                    try:
                        score = int(judge.get("score", 0))
                    except Exception:
                        score = 0
                    judge["score"] = score

                    if score > best_score:
                        best_score = score
                        best_stats = stats

                    accepted = judge_accept(judge) and len(kept) >= min_items_to_accept_batch
                    print(
                        f"did={disclaimer_id} scenario={scenario_name} prompt={prompt_index} "
                        f"attempt={attempt}/{max_tries_per_prompt} score={score} kept={len(kept)}/{stats['input_len']}"
                    )

                    if accepted:
                        added = 0
                        for item in kept:
                            label = norm_label(item.get("label"))
                            text = enforce_single_turn(item.get("text", ""))
                            if label not in {"compliant", "non-compliant"}:
                                continue
                            if word_count(text) < min_words:
                                continue
                            dataset.append(
                                {
                                    "disclaimer_id": disclaimer_key,
                                    "scenario": scenario_key,
                                    "scenario_name": scenario_name,
                                    "prompt_index": prompt_index,
                                    "type": label,
                                    "dialogue": text.lower().strip(),
                                    "source_batch_size": stats["input_len"],
                                    "kept_after_filter": len(kept),
                                    "supervisor_score": score,
                                }
                            )
                            added += 1

                        save_json(output_path, dataset)
                        completed_keys.add(key)
                        print(f"accepted did={disclaimer_id} scenario={scenario_name} prompt={prompt_index} added={added}")
                        break

                    notes = str(judge.get("notes", "")).strip()
                    if not stats["input_is_array"]:
                        fix_hint = "Output must be ONLY a valid JSON array (no extra text)."
                    elif stats["dropped_schema"] > 0:
                        fix_hint = "Fix schema: each item must have exactly keys {label, text} and both must be strings."
                    elif stats["dropped_too_short"] > 0:
                        fix_hint = f"Make snippets longer: each RM turn must be at least {min_words} words."
                    elif stats["dropped_multi_speaker"] > 0:
                        fix_hint = "Ensure each item is RM-only with no role prefixes or multi-speaker text."
                    else:
                        fix_hint = "Follow the prompt constraints more clearly; ensure compliant vs near-miss patterns are obvious."
                    if notes:
                        fix_hint += f" Supervisor notes: {notes}"

                    time.sleep(sleep_between_calls_sec)

                if key not in completed_keys:
                    kept_count = 0 if best_stats is None else int(best_stats["kept"])
                    print(
                        f"failed did={disclaimer_id} scenario={scenario_name} prompt={prompt_index} "
                        f"best_score={best_score} best_kept={kept_count}"
                    )

    save_json(output_path, dataset)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate synthetic SG compliance rows using Ollama Qwen3.")
    parser.add_argument("--prompts", default=str(DEFAULT_PROMPTS_PATH), help="Path to prompts_2_12_sg.json")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help="Output JSON path")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name for generation")
    parser.add_argument("--judge-model", default=None, help="Optional separate model for judging")
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL, help="Ollama chat endpoint URL")
    parser.add_argument("--min-words", type=int, default=18, help="Minimum words per snippet")
    parser.add_argument("--max-words-soft", type=int, default=90, help="Soft maximum words per snippet")
    parser.add_argument("--min-items-to-accept-batch", type=int, default=10, help="Minimum kept items needed to accept a batch")
    parser.add_argument("--max-tries-per-prompt", type=int, default=3, help="Generator/judge retries per prompt")
    parser.add_argument("--sleep-between-calls-sec", type=float, default=0.2, help="Delay between retries")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--batch-size", type=int, default=30, help="Requested batch size in the prompt")
    return parser


def main(argv: list[str] | None = None) -> Path:
    parser = build_parser()
    args = parser.parse_args(argv)
    return generate_synthetic_data(
        prompts_path=Path(args.prompts),
        output_path=Path(args.output),
        generator_model=args.model,
        judge_model=args.judge_model,
        ollama_url=args.ollama_url,
        min_words=args.min_words,
        max_words_soft=args.max_words_soft,
        min_items_to_accept_batch=args.min_items_to_accept_batch,
        max_tries_per_prompt=args.max_tries_per_prompt,
        sleep_between_calls_sec=args.sleep_between_calls_sec,
        seed=args.seed,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
