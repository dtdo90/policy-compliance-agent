"""External API-backed synthetic data generation for configurable demo workflows."""

from __future__ import annotations

import json
import os
import random
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ..core.config import load_config
from ..core.paths import resolve_project_path
from .qwen_generation import (
    build_generator_system,
    build_generator_user,
    build_judge_system,
    build_judge_user,
    done_prompt_keys,
    enforce_single_turn,
    extract_first_json_array,
    extract_judge_json,
    filter_snippet_array,
    judge_accept,
    norm_label,
    prompt_list_from_cfg,
    word_count,
)


def _response_format(value: str | None) -> dict[str, str] | None:
    text = str(value or "").strip().lower()
    if text == "json_object":
        return {"type": "json_object"}
    return None


def _extract_content(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(f"Unexpected API response payload: {payload}")
    message = choices[0].get("message", {})
    content = message.get("content", "")

    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str):
                text_parts.append(item["text"])
        content = "\n".join(text_parts)

    if not isinstance(content, str):
        raise RuntimeError(f"Unexpected API content payload: {payload}")
    return content.strip()


def _chat_completion(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    timeout_sec: int,
    temperature: float,
    top_p: float,
    response_format: dict[str, str] | None = None,
) -> str:
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
    }
    if response_format is not None:
        body["response_format"] = response_format

    request = Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=timeout_sec) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore") if hasattr(exc, "read") else ""
        raise RuntimeError(f"External API request failed with HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not reach external API at {base_url}: {exc}") from exc

    return _extract_content(payload)


def generate(config: dict[str, Any] | None = None, config_path: str | None = None) -> str:
    config = config or load_config(config_path)
    data_settings = config.get("data", {})
    output_settings = config.get("outputs", {})
    synthetic_settings = config.get("synthetic", {})
    api_settings = synthetic_settings.get("external_api", {})

    prompts_path = resolve_project_path(data_settings["prompts_file"])
    output_path = str(output_settings["synthetic_output_path"])
    configs = json.loads(prompts_path.read_text(encoding="utf-8"))
    if not isinstance(configs, dict) or not configs:
        raise ValueError(f"Invalid or empty prompt config at {prompts_path}")

    base_url = str(api_settings.get("base_url", "")).strip()
    api_key_env = str(api_settings.get("api_key_env", "OPENAI_API_KEY")).strip()
    generator_model = str(api_settings.get("generation_model", "")).strip()
    judge_model = str(api_settings.get("judge_model") or generator_model).strip()
    api_key = os.environ.get(api_key_env, "").strip()
    if not base_url or not generator_model or not api_key:
        raise ValueError("External API generation requires synthetic.external_api base_url, generation_model, and API key env.")

    response_format = _response_format(api_settings.get("response_format"))
    timeout_sec = int(api_settings.get("timeout_sec", 120))

    existing_path = resolve_project_path(output_path)
    try:
        existing = json.loads(existing_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        existing = []
    dataset: list[dict[str, Any]] = existing if isinstance(existing, list) else []
    completed_keys = done_prompt_keys(dataset)

    seed = int(synthetic_settings.get("seed", 42))
    min_words = int(synthetic_settings.get("min_words", 18))
    max_words_soft = int(synthetic_settings.get("max_words_soft", 90))
    min_items_to_accept_batch = int(synthetic_settings.get("min_items_to_accept_batch", 10))
    max_tries_per_prompt = int(synthetic_settings.get("max_tries_per_prompt", 3))
    sleep_between_calls = float(synthetic_settings.get("sleep_between_calls_sec", 0.2))
    random.seed(seed)

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
                    raw_output = _chat_completion(
                        base_url=base_url,
                        api_key=api_key,
                        model=generator_model,
                        messages=[
                            {"role": "system", "content": build_generator_system(system_instruction)},
                            {"role": "user", "content": build_generator_user(user_prompt, fix_hint)},
                        ],
                        timeout_sec=timeout_sec,
                        temperature=float(api_settings.get("temperature", synthetic_settings.get("gen_temperature", 0.7))),
                        top_p=float(api_settings.get("top_p", synthetic_settings.get("gen_top_p", 0.9))),
                        response_format=response_format,
                    )

                    snippets = extract_first_json_array(raw_output)
                    kept, stats = filter_snippet_array(
                        snippets,
                        min_words=min_words,
                        max_words_soft=max_words_soft,
                    )

                    judge_output = _chat_completion(
                        base_url=base_url,
                        api_key=api_key,
                        model=judge_model,
                        messages=[
                            {"role": "system", "content": judge_system},
                            {"role": "user", "content": build_judge_user(system_instruction, user_prompt, raw_output)},
                        ],
                        timeout_sec=timeout_sec,
                        temperature=0.0,
                        top_p=1.0,
                        response_format=response_format,
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
                        existing_path.parent.mkdir(parents=True, exist_ok=True)
                        existing_path.write_text(json.dumps(dataset, indent=2, ensure_ascii=False), encoding="utf-8")
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

    existing_path.parent.mkdir(parents=True, exist_ok=True)
    existing_path.write_text(json.dumps(dataset, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path
