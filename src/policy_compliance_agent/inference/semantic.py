"""Semantic compliance inference for SG disclosure rules 2-12."""

from __future__ import annotations

import time
from typing import Any, Callable

import math
import torch

from ..core.config import load_config
from ..core.disclosures import filter_disclaimers, load_disclaimers
from ..core.json_utils import save_json
from ..core.reporting import build_annotation_output, generate_csv_report
from ..core.runtime import configure_cpu_runtime
from ..core.transcripts import chunk_text, extract_speaker_text, load_transcripts_from_folder

DEFAULT_EXCLUDED_RULE_IDS = {"1", "13", "14"}


def _argmax(values: list[float]) -> int:
    return max(range(len(values)), key=lambda index: values[index])


def _argmin(values: list[float]) -> int:
    return min(range(len(values)), key=lambda index: values[index])


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-float(value)))


def _semantic_settings(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("semantic_inference", {})


def _model_settings(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("models", {})


def _output_settings(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("outputs", {})


def _data_settings(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("data", {})


def build_claim_index(disclaimers) -> tuple[list[str], list[dict[str, Any]], dict[str, dict[str, list[int]]]]:
    claim_texts: list[str] = []
    claim_meta: list[dict[str, Any]] = []
    disc_to_claims: dict[str, dict[str, list[int]]] = {}

    for disclaimer in disclaimers:
        groups = {"single": [], "mandatory": [], "standard": []}
        anchor = disclaimer.anchor
        disclaimer_id = str(disclaimer.id)

        if isinstance(anchor, dict):
            for claim_type in ("mandatory", "standard"):
                for claim_idx, claim_text in enumerate(anchor.get(claim_type) or []):
                    if not isinstance(claim_text, str) or not claim_text.strip():
                        continue
                    global_idx = len(claim_texts)
                    claim_texts.append(claim_text.strip())
                    claim_meta.append(
                        {
                            "disclaimer_id": disclaimer_id,
                            "claim_type": claim_type,
                            "claim_idx": claim_idx,
                        }
                    )
                    groups[claim_type].append(global_idx)

            if not groups["mandatory"] and not groups["standard"]:
                fallback = str(disclaimer.criteria or "").strip()
                if fallback:
                    global_idx = len(claim_texts)
                    claim_texts.append(fallback)
                    claim_meta.append(
                        {
                            "disclaimer_id": disclaimer_id,
                            "claim_type": "single",
                            "claim_idx": 0,
                        }
                    )
                    groups["single"].append(global_idx)
        else:
            claim_text = str(anchor or "").strip()
            if not claim_text:
                claim_text = str(disclaimer.criteria or "").strip()
            if claim_text:
                global_idx = len(claim_texts)
                claim_texts.append(claim_text)
                claim_meta.append(
                    {
                        "disclaimer_id": disclaimer_id,
                        "claim_type": "single",
                        "claim_idx": 0,
                    }
                )
                groups["single"].append(global_idx)

        disc_to_claims[disclaimer_id] = groups

    return claim_texts, claim_meta, disc_to_claims


def _fail_result(disclaimer) -> dict[str, Any]:
    return {
        "status": "FAIL",
        "evidence": {
            "description": disclaimer.description,
            "purpose_of_control": disclaimer.purpose_of_control,
            "criteria": disclaimer.criteria,
            "keywords": disclaimer.keywords,
            "match_text": "",
            "retrieval_score": 0.0,
            "verification_score": 0.0,
            "claims": {"single": [], "mandatory": [], "standard": []},
        },
        "for_review": [],
    }


def aggregate_rule_result(
    disclaimer,
    groups: dict[str, list[int]],
    claim_results: list[dict[str, Any]],
    claim_texts: list[str],
    claim_meta: list[dict[str, Any]],
) -> dict[str, Any]:
    single_ids = groups.get("single", [])
    mandatory_ids = groups.get("mandatory", [])
    standard_ids = groups.get("standard", [])

    def claim_evidence(claim_id: int) -> dict[str, Any]:
        return {
            "claim_idx": claim_id,
            "claim_type": claim_meta[claim_id].get("claim_type", ""),
            "passed": bool(claim_results[claim_id]["passed"]),
            "anchor": claim_texts[claim_id],
            "match_text": claim_results[claim_id]["best_text"],
            "retrieval_score": float(claim_results[claim_id]["best_retr"]),
            "verification_score": float(claim_results[claim_id]["best_ver"]),
        }

    if single_ids:
        claim_id = single_ids[0]
        passed = bool(claim_results[claim_id]["passed"])
        overall_score = float(claim_results[claim_id]["best_ver"])
        overall_text = claim_results[claim_id]["best_text"]
        overall_retr = float(claim_results[claim_id]["best_retr"])
        required_for_review = claim_results[claim_id]["for_review"]
        claims_ev = {"single": [claim_evidence(claim_id)], "mandatory": [], "standard": []}
    elif mandatory_ids and not standard_ids:
        mandatory_pass = all(bool(claim_results[claim_id]["passed"]) for claim_id in mandatory_ids)
        passed = mandatory_pass
        mandatory_scores = [float(claim_results[claim_id]["best_ver"]) for claim_id in mandatory_ids] or [0.0]
        overall_score = float(min(mandatory_scores))
        weakest_claim_id = mandatory_ids[_argmin(mandatory_scores)]
        overall_text = claim_results[weakest_claim_id]["best_text"]
        overall_retr = float(claim_results[weakest_claim_id]["best_retr"])
        required_for_review = []
        for claim_id in mandatory_ids:
            if not claim_results[claim_id]["passed"]:
                required_for_review.extend(claim_results[claim_id]["for_review"])
        claims_ev = {
            "single": [],
            "mandatory": [claim_evidence(claim_id) for claim_id in mandatory_ids],
            "standard": [],
        }
    elif not mandatory_ids and standard_ids:
        standard_pass = any(bool(claim_results[claim_id]["passed"]) for claim_id in standard_ids)
        passed = standard_pass
        standard_scores = [float(claim_results[claim_id]["best_ver"]) for claim_id in standard_ids]
        overall_score = float(max(standard_scores)) if standard_scores else 0.0
        best_standard_claim_id = standard_ids[_argmax(standard_scores)] if standard_scores else None
        if best_standard_claim_id is not None:
            overall_text = claim_results[best_standard_claim_id]["best_text"]
            overall_retr = float(claim_results[best_standard_claim_id]["best_retr"])
        else:
            overall_text = ""
            overall_retr = 0.0
        required_for_review = []
        if not passed:
            for claim_id in standard_ids:
                required_for_review.extend(claim_results[claim_id]["for_review"])
        claims_ev = {
            "single": [],
            "mandatory": [],
            "standard": [claim_evidence(claim_id) for claim_id in standard_ids],
        }
    elif mandatory_ids and standard_ids:
        mandatory_pass = all(bool(claim_results[claim_id]["passed"]) for claim_id in mandatory_ids)
        standard_pass = any(bool(claim_results[claim_id]["passed"]) for claim_id in standard_ids)
        passed = mandatory_pass and standard_pass
        mandatory_scores = [float(claim_results[claim_id]["best_ver"]) for claim_id in mandatory_ids] or [0.0]
        standard_scores = [float(claim_results[claim_id]["best_ver"]) for claim_id in standard_ids] or [0.0]
        best_standard_score = float(max(standard_scores)) if standard_scores else 0.0
        overall_score = float(min(min(mandatory_scores), best_standard_score))
        weakest_mandatory_claim_id = mandatory_ids[_argmin(mandatory_scores)]
        best_standard_claim_id = standard_ids[_argmax(standard_scores)] if standard_scores else None

        if not mandatory_pass:
            overall_text = claim_results[weakest_mandatory_claim_id]["best_text"]
            overall_retr = float(claim_results[weakest_mandatory_claim_id]["best_retr"])
        elif best_standard_claim_id is not None:
            overall_text = claim_results[best_standard_claim_id]["best_text"]
            overall_retr = float(claim_results[best_standard_claim_id]["best_retr"])
        else:
            overall_text = ""
            overall_retr = 0.0

        required_for_review = []
        for claim_id in mandatory_ids:
            if not claim_results[claim_id]["passed"]:
                required_for_review.extend(claim_results[claim_id]["for_review"])
        if mandatory_pass and not standard_pass:
            for claim_id in standard_ids:
                required_for_review.extend(claim_results[claim_id]["for_review"])

        claims_ev = {
            "single": [],
            "mandatory": [claim_evidence(claim_id) for claim_id in mandatory_ids],
            "standard": [claim_evidence(claim_id) for claim_id in standard_ids],
        }
    else:
        passed = False
        overall_score = 0.0
        overall_text = ""
        overall_retr = 0.0
        required_for_review = []
        claims_ev = {"single": [], "mandatory": [], "standard": []}

    evidence = {
        "description": disclaimer.description,
        "purpose_of_control": disclaimer.purpose_of_control,
        "criteria": disclaimer.criteria,
        "keywords": disclaimer.keywords,
        "match_text": overall_text,
        "retrieval_score": float(overall_retr),
        "verification_score": float(overall_score),
        "claims": claims_ev,
    }

    if passed:
        return {"status": "PASS", "evidence": evidence}
    return {"status": "FAIL", "evidence": evidence, "for_review": required_for_review}


class SemanticComplianceAnalyzer:
    def __init__(
        self,
        disclaimers,
        config: dict[str, Any],
        retriever=None,
        verifier=None,
        semantic_search_fn: Callable[[Any, Any, int], Any] | None = None,
    ) -> None:
        configure_cpu_runtime(int(config.get("runtime", {}).get("cpu_threads", 4)))

        self.config = config
        self.disclaimers = list(disclaimers)
        self.settings = _semantic_settings(config)
        self.model_settings = _model_settings(config)
        self.retrieval_top_k = int(self.settings.get("retrieval_top_k", 5))
        self.verification_threshold = float(self.settings.get("verification_threshold", 0.7))
        self.borderline_low = float(self.settings.get("borderline_low", 0.4))
        self.borderline_high = float(self.settings.get("borderline_high", 0.7))
        self.max_chunk_words = int(self.settings.get("max_chunk_words", 50))
        self.chunk_overlap = int(self.settings.get("chunk_overlap", 20))
        self.speaker_labels = [
            str(label).strip()
            for label in self.settings.get("speaker_labels", [])
            if str(label).strip()
        ]
        self.bi_encoder_batch_size = int(self.settings.get("bi_encoder_batch_size", 128))
        self.cross_encoder_batch_size = int(self.settings.get("cross_encoder_batch_size", 64))

        self.claim_texts, self.claim_meta, self.disc_to_claims = build_claim_index(self.disclaimers)

        if retriever is None or verifier is None or semantic_search_fn is None:
            from sentence_transformers import CrossEncoder, SentenceTransformer, util

            self.retriever = retriever or SentenceTransformer(self.model_settings["bi_encoder_path"])
            self.verifier = verifier or CrossEncoder(self.model_settings["cross_encoder_path"])
            self.semantic_search_fn = semantic_search_fn or util.semantic_search
        else:
            self.retriever = retriever
            self.verifier = verifier
            self.semantic_search_fn = semantic_search_fn

        if self.claim_texts:
            self.claim_embeddings = self.retriever.encode(
                self.claim_texts,
                convert_to_tensor=True,
                batch_size=self.bi_encoder_batch_size,
                show_progress_bar=False,
            )
        else:
            self.claim_embeddings = None

    def analyze_transcript(self, text: str, transcript_id: str) -> dict[str, dict[str, Any]]:
        chunk_source = text.strip()
        if chunk_source and self.speaker_labels:
            chunk_source = extract_speaker_text(chunk_source, self.speaker_labels)

        chunks = chunk_text(chunk_source, self.max_chunk_words, self.chunk_overlap) if chunk_source else []
        transcript_results: dict[str, dict[str, Any]] = {}

        if chunks and self.claim_embeddings is not None:
            chunk_embeddings = self.retriever.encode(
                chunks,
                convert_to_tensor=True,
                batch_size=self.bi_encoder_batch_size,
                show_progress_bar=False,
            )
            search_results = self.semantic_search_fn(self.claim_embeddings, chunk_embeddings, top_k=self.retrieval_top_k)
        else:
            search_results = []

        claim_results = [
            {"passed": False, "best_ver": 0.0, "best_retr": 0.0, "best_text": "", "for_review": []}
            for _ in range(len(self.claim_texts))
        ]

        if self.claim_texts:
            per_claim_meta: list[list[dict[str, Any]]] = [[] for _ in range(len(self.claim_texts))]
            all_pairs: list[list[str]] = []

            for claim_idx, results in enumerate(search_results):
                claim_text = self.claim_texts[claim_idx]
                for hit in results:
                    chunk_idx = hit["corpus_id"]
                    retrieval_score = float(hit.get("score", 0.0))
                    all_pairs.append([claim_text, chunks[chunk_idx]])
                    per_claim_meta[claim_idx].append(
                        {"text": chunks[chunk_idx], "retrieval_score": retrieval_score}
                    )

            if all_pairs:
                raw_scores = self.verifier.predict(
                    all_pairs,
                    batch_size=self.cross_encoder_batch_size,
                    show_progress_bar=False,
                    activation_fn=torch.nn.Identity(),
                )
                all_verification_scores = [_sigmoid(score) for score in raw_scores]
            else:
                all_verification_scores = []

            cursor = 0
            for claim_idx, claim_hits in enumerate(per_claim_meta):
                hit_count = len(claim_hits)
                if hit_count == 0:
                    continue

                verification_scores = list(all_verification_scores[cursor : cursor + hit_count])
                cursor += hit_count
                best_idx = _argmax(verification_scores)
                best_verification = float(verification_scores[best_idx])

                for_review = []
                for hit_idx, verification_score in enumerate(verification_scores):
                    verification_float = float(verification_score)
                    if self.borderline_low <= verification_float < self.borderline_high:
                        for_review.append(
                            {
                                "text": claim_hits[hit_idx]["text"],
                                "retrieval_score": float(claim_hits[hit_idx]["retrieval_score"]),
                                "verification_score": verification_float,
                                "claim_type": self.claim_meta[claim_idx]["claim_type"],
                                "claim_text": self.claim_texts[claim_idx],
                            }
                        )

                claim_results[claim_idx] = {
                    "passed": best_verification >= self.verification_threshold,
                    "best_ver": best_verification,
                    "best_retr": float(claim_hits[best_idx]["retrieval_score"]),
                    "best_text": claim_hits[best_idx]["text"],
                    "for_review": for_review,
                }

        for disclaimer in self.disclaimers:
            groups = self.disc_to_claims.get(str(disclaimer.id), {"single": [], "mandatory": [], "standard": []})
            if not self.claim_texts:
                transcript_results[str(disclaimer.id)] = _fail_result(disclaimer)
                continue
            transcript_results[str(disclaimer.id)] = aggregate_rule_result(
                disclaimer=disclaimer,
                groups=groups,
                claim_results=claim_results,
                claim_texts=self.claim_texts,
                claim_meta=self.claim_meta,
            )

        return transcript_results


def run_semantic_inference(
    input_path: str | None = None,
    config: dict[str, Any] | None = None,
    config_path: str | None = None,
) -> dict[str, dict[str, Any]]:
    config = config or load_config(config_path)
    data_settings = _data_settings(config)
    output_settings = _output_settings(config)
    semantic_settings = _semantic_settings(config)

    include_rule_ids = semantic_settings.get("include_rule_ids")
    disclosures = load_disclaimers(data_settings["disclosures_file"])
    disclosures = filter_disclaimers(
        disclosures,
        include_rule_ids=include_rule_ids,
        exclude_rule_ids=DEFAULT_EXCLUDED_RULE_IDS,
    )

    transcript_source = input_path or data_settings["transcripts_txt_dir"]
    transcripts_list = load_transcripts_from_folder(transcript_source)
    analyzer = SemanticComplianceAnalyzer(disclosures, config)

    from tqdm import tqdm

    final_report: dict[str, dict[str, Any]] = {}
    passed_cases: dict[str, dict[str, Any]] = {}
    borderline_out: dict[str, dict[str, Any]] = {}

    start_time = time.time()
    for item in tqdm(transcripts_list, desc="Analyzing transcripts"):
        transcript_id = str(item.get("transcript_id", "")).strip()
        transcript = str(item.get("transcript", ""))
        if not transcript_id:
            continue

        result = analyzer.analyze_transcript(transcript, transcript_id)
        final_report[transcript_id] = result
        passed_cases[transcript_id] = {}

        for disclaimer_id, disclaimer_result in result.items():
            if disclaimer_result.get("status") == "PASS":
                passed_cases[transcript_id][disclaimer_id] = disclaimer_result
            if disclaimer_result.get("for_review"):
                borderline_out.setdefault(transcript_id, {})
                borderline_out[transcript_id][disclaimer_id] = disclaimer_result

    elapsed = time.time() - start_time
    print(f"Processed {len(transcripts_list)} transcripts in {elapsed:.2f}s")

    save_json(output_settings["report_json_path"], final_report)
    save_json(output_settings["passed_cases_json_path"], passed_cases)
    save_json(output_settings["borderline_json_path"], borderline_out)
    generate_csv_report(final_report, output_settings["report_csv_path"])

    if bool(semantic_settings.get("annotation", False)):
        save_json(output_settings["annotation_json_path"], build_annotation_output(final_report))

    return final_report


def main(config_path: str | None = None, input_path: str | None = None) -> dict[str, dict[str, Any]]:
    return run_semantic_inference(input_path=input_path, config_path=config_path)
