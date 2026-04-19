"""Synthetic compliant transcript generation for SG disclosure scenarios 2-12.

This script uses an Ollama-hosted Qwen model in a generator -> judge loop.
It writes accepted conversations as `.txt` files under `synthetic_transcripts/`.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ..core.paths import resolve_project_path

DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:8b")
DEFAULT_BASE_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
DEFAULT_OUTPUT_DIR = resolve_project_path("synthetic_transcripts")
DEFAULT_DISCLOSURES_FILE = resolve_project_path("resources/disclosures_with_anchors_sg.json")
DEFAULT_TARGET_COUNT = 20
DEFAULT_MAX_TRIES = 4
DEFAULT_TIMEOUT_SEC = 600

SYSTEM_PROMPT = """You write synthetic SG banking phone-call transcripts for compliance data generation.

Rules:
- Produce a realistic conversation with 15 to 25 turns total.
- Use only the speakers RM and Client.
- Make the conversation feel natural and coherent, with short client replies and longer RM explanations when needed.
- Keep the language conversational and spoken, but not sloppy or unreadable.
- Output only valid JSON, no markdown, no code fences, no commentary.
- The JSON must be an array of turn objects with exactly these keys: speaker, text.
- speaker must be exactly RM or Client.
- text must be a single spoken turn with no extra speaker labels.
"""

JUDGE_SYSTEM_PROMPT = """You are a strict transcript compliance judge.

You will receive:
- expected_requirements: the disclosure scenarios and anchor meanings the transcript must cover
- transcript: the generated conversation

Decide whether the transcript is compliant with every expected requirement.

Judging rules:
- For scenarios 2-8 and 12, the transcript must clearly express the single anchor meaning for each scenario.
- For scenarios 9 and 10, the transcript must clearly express every mandatory anchor meaning for each scenario.
- For scenario 11, the transcript must clearly express every mandatory anchor meaning and at least one standard anchor meaning.
- Paraphrasing is allowed, but the meaning must be obvious from the conversation.

Return only valid JSON with exactly these keys:
- pass: true or false
- score: 0 to 10
- violations: array of short strings
- covered_scenarios: array of scenario ids that are clearly covered
- missing_scenarios: array of scenario ids that are not clearly covered
- notes: short fix-oriented notes

PASS requires:
- all expected scenarios are covered semantically
- the transcript has between 15 and 25 turns
- the transcript is coherent and readable
"""

STYLE_LIBRARY = [
    "standard business tone with calm, professional pacing",
    "warm and conversational, like a familiar relationship manager call",
    "slightly rushed but still clear and human",
    "patient and explanatory, with a client who asks a few questions",
    "lightly informal, with natural spoken phrasing",
]

TRANSCRIPT_PLAN_CANDIDATES: list[list[str]] = [
    ["2"],
    ["3"],
    ["4"],
    ["5"],
    ["6"],
    ["7"],
    ["8"],
    ["12"],
    ["9"],
    ["10"],
    ["11"],
    ["2", "6", "12"],
    ["2", "4"],
    ["3", "5"],
    ["4", "6"],
    ["5", "7"],
    ["6", "8"],
    ["2", "3", "4"],
    ["2", "5", "12"],
    ["3", "6", "7"],
    ["2", "7", "8"],
    ["3", "4", "12"],
    ["4", "5", "6"],
    ["2", "3", "6"],
    ["5", "6", "7"],
    ["2", "4", "8"],
    ["3", "5", "12"],
]


@dataclass(frozen=True)
class ScenarioRequirement:
    scenario_id: str
    kind: str
    mandatory: list[str]
    standard: list[str]

    @property
    def all_anchors(self) -> list[str]:
        return self.mandatory + self.standard


def _load_json_file(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_disclosures(path: Path = DEFAULT_DISCLOSURES_FILE) -> dict[str, Any]:
    data = _load_json_file(path)
    if not isinstance(data, dict):
        raise ValueError(f"Expected disclosure file to contain an object: {path}")
    return data


def build_scenario_requirement(disclosures: dict[str, Any], scenario_id: str) -> ScenarioRequirement:
    entry = disclosures.get(str(scenario_id))
    if not isinstance(entry, dict):
        raise KeyError(f"Scenario {scenario_id} not found in disclosures")

    anchor = entry.get("anchor", "")
    if str(scenario_id) in {"9", "10", "11"}:
        if not isinstance(anchor, dict):
            raise ValueError(f"Scenario {scenario_id} is expected to have structured anchors")

        mandatory = [
            str(item).strip()
            for item in anchor.get("mandatory", [])
            if isinstance(item, str) and item.strip()
        ]
        standard = [
            str(item).strip()
            for item in anchor.get("standard", [])
            if isinstance(item, str) and item.strip()
        ]
        if str(scenario_id) == "11":
            if not mandatory or not standard:
                raise ValueError("Scenario 11 requires both mandatory and standard anchors")
            return ScenarioRequirement(
                scenario_id=scenario_id,
                kind="mandatory_plus_standard",
                mandatory=mandatory,
                standard=standard,
            )
        if not mandatory:
            raise ValueError(f"Scenario {scenario_id} requires mandatory anchors")
        return ScenarioRequirement(
            scenario_id=scenario_id,
            kind="mandatory_all",
            mandatory=mandatory,
            standard=[],
        )

    if not isinstance(anchor, str) or not anchor.strip():
        fallback = str(entry.get("Criteria", "")).strip()
        if not fallback:
            raise ValueError(f"Scenario {scenario_id} has no usable anchor text")
        anchor = fallback
    return ScenarioRequirement(
        scenario_id=scenario_id,
        kind="single",
        mandatory=[anchor.strip()],
        standard=[],
    )


def build_expected_requirements(disclosures: dict[str, Any], scenario_ids: Iterable[str]) -> list[ScenarioRequirement]:
    return [build_scenario_requirement(disclosures, scenario_id) for scenario_id in scenario_ids]


def requirement_summary(requirement: ScenarioRequirement) -> str:
    if requirement.kind == "single":
        return f"Scenario {requirement.scenario_id}: clearly convey this meaning -> {requirement.mandatory[0]}"
    if requirement.kind == "mandatory_all":
        anchors = "\n".join(f"  - mandatory {index + 1}: {anchor}" for index, anchor in enumerate(requirement.mandatory))
        return (
            f"Scenario {requirement.scenario_id}: cover every mandatory point below. "
            "The RM can paraphrase naturally, but each meaning must be clearly present:\n"
            f"{anchors}"
        )
    mandatory = "\n".join(f"  - mandatory {index + 1}: {anchor}" for index, anchor in enumerate(requirement.mandatory))
    standard = "\n".join(f"  - standard {index + 1}: {anchor}" for index, anchor in enumerate(requirement.standard))
    return (
        f"Scenario {requirement.scenario_id}: cover all mandatory points below and at least one standard point. "
        "Paraphrasing is allowed, but the meaning must be clear:\n"
        f"Mandatory:\n{mandatory}\n\nStandard options:\n{standard}"
    )


def build_generator_system_prompt(style: str) -> str:
    return SYSTEM_PROMPT + f"\nStyle goal: {style}."


def build_generator_user_prompt(requirements: list[ScenarioRequirement], style: str, transcript_name: str) -> str:
    requirement_text = "\n\n".join(requirement_summary(item) for item in requirements)
    return f"""Generate one synthetic transcript named {transcript_name}.

Style:
- {style}

Conversation requirements:
- Exactly one transcript.
- 15 to 25 turns total.
- Alternate RM and Client naturally.
- Each turn should be a single spoken utterance.
- The RM must clearly and naturally cover every requirement below.
- For scenario 11, include every mandatory point and at least one standard point.
- Do not mention the word "anchor" or "scenario" in the transcript.
- Output only valid JSON array of turn objects with keys speaker and text.

Requirements:
{requirement_text}
"""


def build_judge_user_prompt(requirements: list[ScenarioRequirement], transcript_text: str, turn_count: int) -> str:
    payload = {
        "expected_requirements": [
            {
                "scenario_id": requirement.scenario_id,
                "kind": requirement.kind,
                "mandatory": requirement.mandatory,
                "standard": requirement.standard,
                "mandatory_required_count": len(requirement.mandatory) if requirement.kind != "single" else 1,
                "standard_required_count": 1 if requirement.kind == "mandatory_plus_standard" else 0,
            }
            for requirement in requirements
        ],
        "turn_count": turn_count,
        "transcript": transcript_text,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _ollama_chat(
    *,
    model: str,
    base_url: str,
    messages: list[dict[str, str]],
    timeout_sec: int,
    temperature: float,
    top_p: float,
    response_format: str | None = None,
) -> str:
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": top_p,
        },
    }
    if response_format is not None:
        body["format"] = response_format

    request = Request(
        f"{base_url.rstrip('/')}/api/chat",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=timeout_sec) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore") if hasattr(exc, "read") else ""
        raise RuntimeError(f"Ollama request failed with HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not reach Ollama at {base_url}: {exc}") from exc

    message = payload.get("message", {})
    if not isinstance(message, dict):
        raise RuntimeError(f"Unexpected Ollama response payload: {payload}")
    content = message.get("content", "")
    if not isinstance(content, str):
        raise RuntimeError(f"Unexpected Ollama message content: {payload}")
    return content.strip()


def _extract_json_value(text: str) -> Any:
    text = text.strip()
    if text.startswith("```"):
        text = text.replace("```json", "", 1).replace("```", "").strip()

    if text.startswith("[") or text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

    start = min((index for index in (text.find("["), text.find("{")) if index != -1), default=-1)
    if start < 0:
        raise ValueError("No JSON payload found in model output")

    candidate = text[start:]
    for end in range(len(candidate), 0, -1):
        snippet = candidate[:end]
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            continue
    raise ValueError("Could not parse JSON from model output")


def parse_turns(payload: Any) -> list[dict[str, str]]:
    if not isinstance(payload, list):
        raise ValueError("Expected a JSON array of turns")

    turns: list[dict[str, str]] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("Every turn must be an object")
        speaker = str(item.get("speaker", "")).strip()
        text = str(item.get("text", "")).strip()
        if speaker not in {"RM", "Client"}:
            raise ValueError(f"Unexpected speaker label: {speaker}")
        if not text:
            raise ValueError("Empty transcript turn text")
        turns.append({"speaker": speaker, "text": text})
    return turns


def render_turns_as_text(turns: list[dict[str, str]]) -> str:
    return "\n".join(f"{turn['speaker']}: {turn['text']}" for turn in turns).strip() + "\n"


def _judge_accepts(result: dict[str, Any], turn_count: int, expected_ids: set[str]) -> bool:
    if not bool(result.get("pass")):
        return False
    if int(result.get("score", 0)) < 9:
        return False
    missing = {str(item) for item in result.get("missing_scenarios", []) if str(item).strip()}
    covered = {str(item) for item in result.get("covered_scenarios", []) if str(item).strip()}
    if missing:
        return False
    if not expected_ids.issubset(covered):
        return False
    return 15 <= turn_count <= 25


def _cleanup_text(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().splitlines()).strip() + "\n"


def _output_stem(scenario_ids: Iterable[str]) -> str:
    ids = sorted({str(item).strip() for item in scenario_ids if str(item).strip()}, key=lambda value: int(value))
    return "_".join(ids)


def _build_candidate_plans(target_count: int) -> list[list[str]]:
    plans = [plan[:] for plan in TRANSCRIPT_PLAN_CANDIDATES]
    if target_count <= len(plans):
        return plans
    # If the caller wants more than the default list, repeat the combination library.
    while len(plans) < target_count:
        plans.extend(plan[:] for plan in TRANSCRIPT_PLAN_CANDIDATES)
    return plans


def generate_transcripts(
    *,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    target_count: int = DEFAULT_TARGET_COUNT,
    max_tries: int = DEFAULT_MAX_TRIES,
    timeout_sec: int = DEFAULT_TIMEOUT_SEC,
    seed: int = 42,
) -> list[Path]:
    random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    disclosures = load_disclosures()
    candidate_plans = _build_candidate_plans(target_count + 10)
    written_paths: list[Path] = []

    for plan in candidate_plans:
        if len(written_paths) >= target_count:
            break

        requirements = build_expected_requirements(disclosures, plan)
        transcript_name = _output_stem(plan)
        style = random.choice(STYLE_LIBRARY)
        expected_ids = {requirement.scenario_id for requirement in requirements}

        best_error: str | None = None
        for attempt in range(1, max_tries + 1):
            generator_messages = [
                {"role": "system", "content": build_generator_system_prompt(style)},
                {
                    "role": "user",
                    "content": build_generator_user_prompt(requirements, style, transcript_name),
                },
            ]

            try:
                raw_generation = _ollama_chat(
                    model=model,
                    base_url=base_url,
                    messages=generator_messages,
                    timeout_sec=timeout_sec,
                    temperature=0.7,
                    top_p=0.9,
                    response_format="json",
                )
                turns = parse_turns(_extract_json_value(raw_generation))
                transcript_text = _cleanup_text(render_turns_as_text(turns))
            except Exception as exc:
                best_error = str(exc)
                print(
                    f"[generate] plan={transcript_name} attempt={attempt}/{max_tries} "
                    f"generator error: {best_error}"
                )
                continue

            judge_messages = [
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": build_judge_user_prompt(requirements, transcript_text, len(turns)),
                },
            ]

            try:
                raw_judgment = _ollama_chat(
                    model=model,
                    base_url=base_url,
                    messages=judge_messages,
                    timeout_sec=timeout_sec,
                    temperature=0.0,
                    top_p=1.0,
                    response_format="json",
                )
                judgment = _extract_json_value(raw_judgment)
                if not isinstance(judgment, dict):
                    raise ValueError("Judge did not return a JSON object")
            except Exception as exc:
                best_error = str(exc)
                print(
                    f"[generate] plan={transcript_name} attempt={attempt}/{max_tries} "
                    f"judge error: {best_error}"
                )
                continue

            if _judge_accepts(judgment, len(turns), expected_ids):
                file_path = output_dir / f"{transcript_name}.txt"
                file_path.write_text(transcript_text, encoding="utf-8")
                written_paths.append(file_path)
                print(
                    f"[generate] accepted {file_path.name} "
                    f"with {len(turns)} turns and scenarios {transcript_name}"
                )
                break

            best_error = json.dumps(judgment, ensure_ascii=False)
            print(
                f"[generate] plan={transcript_name} attempt={attempt}/{max_tries} "
                f"rejected: {best_error}"
            )

        if transcript_name and (output_dir / f"{transcript_name}.txt").exists():
            continue

        if best_error is not None:
            print(f"[generate] plan={transcript_name} did not pass after {max_tries} tries: {best_error}")

    return written_paths


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate synthetic compliant transcript TXT files via Ollama.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name, default qwen3:8b.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Ollama base URL, default http://127.0.0.1:11434.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output folder for transcript TXT files.")
    parser.add_argument("--count", type=int, default=DEFAULT_TARGET_COUNT, help="Target number of successful transcripts.")
    parser.add_argument("--max-tries", type=int, default=DEFAULT_MAX_TRIES, help="Max attempts per plan.")
    parser.add_argument("--timeout-sec", type=int, default=DEFAULT_TIMEOUT_SEC, help="HTTP timeout in seconds.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for plan/style selection.")
    return parser


def main(argv: list[str] | None = None) -> list[Path]:
    args = build_arg_parser().parse_args(argv)
    return generate_transcripts(
        model=args.model,
        base_url=args.base_url,
        output_dir=resolve_project_path(args.output_dir),
        target_count=args.count,
        max_tries=args.max_tries,
        timeout_sec=args.timeout_sec,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
