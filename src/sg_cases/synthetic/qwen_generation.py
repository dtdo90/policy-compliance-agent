"""Synthetic policy-control snippet generation helpers."""

from __future__ import annotations

import json
import random
import re
import time
from typing import Any

from ..core.config import load_config
from ..core.json_utils import safe_json_load, save_json
from ..core.paths import resolve_project_path

GEN_WRAPPER_SYSTEM = """You must follow the user's instructions exactly.
Return ONLY the JSON requested. Do not add markdown, backticks, or any extra commentary outside the JSON.
"""

_THINK_RE = re.compile(r"<think>.*?</think>", flags=re.IGNORECASE | re.DOTALL)


def strip_think_blocks(text: str) -> str:
    if not text:
        return ""
    return _THINK_RE.sub("", text).strip()


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
    out = text.strip()
    for prefix in ["rm:", "relationship manager:", "agent:", "advisor:"]:
        if out.lower().startswith(prefix):
            out = out[len(prefix) :].strip()
    return " ".join(out.splitlines()).strip()


def word_count(text: str) -> int:
    return len([word for word in str(text).strip().split() if word.strip()])


def prompt_list_from_cfg(user_prompt: Any) -> list[str]:
    if isinstance(user_prompt, str) and user_prompt.strip():
        return [user_prompt]
    if isinstance(user_prompt, list):
        return [prompt for prompt in user_prompt if isinstance(prompt, str) and prompt.strip()]
    return []


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
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        else:
            if char == '"':
                in_string = True
            elif char == "[":
                depth += 1
            elif char == "]":
                depth -= 1
                if depth == 0:
                    candidate = text[start : index + 1]
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
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        else:
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : index + 1]
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


def make_chat_input(tokenizer, system: str, user: str) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"SYSTEM:\n{system}\n\nUSER:\n{user}\n\nASSISTANT:\n"


def generate_text(model, tokenizer, prompt: str, *, max_new_tokens: int, temperature: float, top_p: float, do_sample: bool) -> str:
    import torch

    with torch.inference_mode():
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        kwargs = dict(
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=do_sample,
        )
        if do_sample:
            kwargs.update({"temperature": temperature, "top_p": top_p})
        out = model.generate(**inputs, **kwargs)
        text = tokenizer.decode(out[0], skip_special_tokens=True)

    if text.startswith(prompt):
        text = text[len(prompt) :]
    return text.strip()


def build_generator_system(scenario_instruction: str) -> str:
    if isinstance(scenario_instruction, str) and scenario_instruction.strip():
        return GEN_WRAPPER_SYSTEM.strip() + "\n\n" + scenario_instruction.strip()
    return GEN_WRAPPER_SYSTEM.strip()


def build_generator_user(user_prompt: str, fix_hint: str | None) -> str:
    prompt = (user_prompt or "").strip()
    if fix_hint:
        prompt += "\n\nFIX INSTRUCTION (must follow): " + fix_hint.strip()
    return prompt


def build_judge_system(min_words: int, max_words_soft: int) -> str:
    return f"""You are a strict policy compliance supervisor scoring a GENERATED JSON ARRAY of agent snippets.

You will be given:
- system_instruction (scenario instruction)
- user_prompt (task instruction)
- generated_output (raw model output, expected to be a JSON array)

Evaluate:
1) Output is a valid JSON array ONLY (no extra text).
2) Schema: each element is an object with exactly keys ["label","text"].
   - label is exactly "Compliant" or "Non-Compliant"
   - text is a single agent turn (no multi-speaker / no "Client:" etc.)
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
    out: set[tuple[str, str, int]] = set()
    for row in dataset_rows:
        if not isinstance(row, dict):
            continue
        disclaimer_id = str(row.get("disclaimer_id", "")).strip().lower()
        scenario_name = str(row.get("scenario_name", "")).strip().lower()
        prompt_index = row.get("prompt_index")
        if disclaimer_id and scenario_name and isinstance(prompt_index, int):
            out.add((disclaimer_id, scenario_name, prompt_index))
    return out


def generate(config: dict[str, Any] | None = None, config_path: str | None = None) -> str:
    config = config or load_config(config_path)
    data_settings = config.get("data", {})
    model_settings = config.get("models", {})
    output_settings = config.get("outputs", {})
    synthetic_settings = config.get("synthetic", {})

    prompts_path = resolve_project_path(data_settings["prompts_file"])
    output_path = str(output_settings["synthetic_output_path"])
    configs = safe_json_load(str(prompts_path), {})
    if not isinstance(configs, dict) or not configs:
        raise ValueError(f"Invalid or empty prompt config at {prompts_path}")

    existing = safe_json_load(output_path, [])
    dataset: list[dict[str, Any]] = existing if isinstance(existing, list) else []
    completed_keys = done_prompt_keys(dataset)

    seed = int(synthetic_settings.get("seed", 42))
    min_words = int(synthetic_settings.get("min_words", 18))
    max_words_soft = int(synthetic_settings.get("max_words_soft", 90))
    min_items_to_accept_batch = int(synthetic_settings.get("min_items_to_accept_batch", 10))
    max_tries_per_prompt = int(synthetic_settings.get("max_tries_per_prompt", 3))
    sleep_between_calls = float(synthetic_settings.get("sleep_between_calls_sec", 0.2))

    random.seed(seed)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained(
        model_settings["qwen_model_path"],
        use_fast=True,
        trust_remote_code=bool(synthetic_settings.get("trust_remote_code", True)),
    )
    dtype_name = str(model_settings.get("qwen_dtype", "float16"))
    dtype = torch.float16 if dtype_name == "float16" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_settings["qwen_model_path"],
        torch_dtype=dtype,
        device_map=model_settings.get("qwen_device_map", "auto"),
        low_cpu_mem_usage=True,
        trust_remote_code=bool(synthetic_settings.get("trust_remote_code", True)),
    )
    model.eval()

    judge_system = build_judge_system(min_words=min_words, max_words_soft=max_words_soft)

    for disclaimer_id, scenarios in configs.items():
        if not isinstance(scenarios, dict):
            continue
        for scenario_name, scenario_cfg in scenarios.items():
            if scenario_name == "_qa" or not isinstance(scenario_cfg, dict):
                continue

            system_instruction = scenario_cfg.get("system_instruction", "")
            user_prompts = prompt_list_from_cfg(scenario_cfg.get("user_prompt", ""))
            if not isinstance(system_instruction, str) or not system_instruction.strip() or not user_prompts:
                continue

            scenario_key = str(scenario_name).strip().lower()
            disclaimer_key = str(disclaimer_id).strip().lower()
            for prompt_index, user_prompt in enumerate(user_prompts):
                key = (disclaimer_key, scenario_key, prompt_index)
                if key in completed_keys:
                    continue

                fix_hint = None
                best_score = -1
                best_stats = None

                for attempt in range(1, max_tries_per_prompt + 1):
                    generator_prompt = make_chat_input(
                        tokenizer,
                        build_generator_system(system_instruction),
                        build_generator_user(user_prompt, fix_hint),
                    )
                    raw_output = generate_text(
                        model,
                        tokenizer,
                        generator_prompt,
                        max_new_tokens=int(synthetic_settings.get("gen_max_new_tokens", 4096)),
                        temperature=float(synthetic_settings.get("gen_temperature", 0.7)),
                        top_p=float(synthetic_settings.get("gen_top_p", 0.9)),
                        do_sample=True,
                    )

                    snippets = extract_first_json_array(raw_output)
                    kept, stats = filter_snippet_array(
                        snippets,
                        min_words=min_words,
                        max_words_soft=max_words_soft,
                    )

                    judge_prompt = make_chat_input(
                        tokenizer,
                        judge_system,
                        build_judge_user(system_instruction, user_prompt, raw_output),
                    )
                    judge_output = generate_text(
                        model,
                        tokenizer,
                        judge_prompt,
                        max_new_tokens=int(synthetic_settings.get("judge_max_new_tokens", 4096)),
                        temperature=0.0,
                        top_p=1.0,
                        do_sample=False,
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
                    accepted = judge_accept(judge) and len(kept) >= min_items_to_accept_batch

                    if score > best_score:
                        best_score = score
                        best_stats = stats

                    print(
                        f"did={disclaimer_id} scenario={scenario_name} prompt={prompt_index} "
                        f"attempt={attempt}/{max_tries_per_prompt} score={score} kept={len(kept)}/{stats['input_len']}"
                    )

                    if accepted:
                        for item in kept:
                            label_norm = norm_label(item.get("label"))
                            text = enforce_single_turn(item.get("text", ""))
                            if label_norm not in {"compliant", "non-compliant"}:
                                continue
                            if word_count(text) < min_words:
                                continue
                            dataset.append(
                                {
                                    "disclaimer_id": disclaimer_key,
                                    "scenario": scenario_key,
                                    "scenario_name": scenario_name,
                                    "prompt_index": prompt_index,
                                    "type": label_norm,
                                    "dialogue": text.lower().strip(),
                                    "source_batch_size": stats["input_len"],
                                    "kept_after_filter": len(kept),
                                    "supervisor_score": score,
                                }
                            )
                        save_json(output_path, dataset)
                        completed_keys.add(key)
                        break

                    notes = str(judge.get("notes", "")).strip()
                    if not stats["input_is_array"]:
                        fix_hint = "Output must be ONLY a valid JSON array (no extra text)."
                    elif stats["dropped_schema"] > 0:
                        fix_hint = "Fix schema: each item must have exactly keys {label, text} and both must be strings."
                    elif stats["dropped_too_short"] > 0:
                        fix_hint = f"Make snippets longer: each agent turn must be at least {min_words} words."
                    elif stats["dropped_multi_speaker"] > 0:
                        fix_hint = "Ensure each item is agent-only with no role prefixes or multi-speaker text."
                    else:
                        fix_hint = "Follow the prompt constraints more clearly; ensure compliant vs near-miss patterns are obvious."
                    if notes:
                        fix_hint += f" Supervisor notes: {notes}"
                    time.sleep(sleep_between_calls)

                if key not in completed_keys:
                    kept_count = 0 if best_stats is None else int(best_stats["kept"])
                    print(f"Failed did={disclaimer_id} scenario={scenario_name} prompt={prompt_index} best_score={best_score} kept={kept_count}")

    save_json(output_path, dataset)
    return output_path


def main(config_path: str | None = None) -> str:
    return generate(config_path=config_path)
