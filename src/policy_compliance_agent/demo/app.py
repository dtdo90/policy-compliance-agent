"""Gradio app for the privacy-safe demo workflow."""

from __future__ import annotations

import base64
from copy import deepcopy
import html
import json
from pathlib import Path
import re
import shutil
import tempfile
import threading
import time
from typing import Any

import pandas as pd

from ..agentic import answer_agentic_question
from ..agentic.loop import load_incoming_transcripts
from .services import (
    answer_demo_question,
    approve_demo_examples,
    complete_agentic_reinference_cycle,
    diagnose_label_changed_cases,
    investigate_score_regressions_with_ollama,
    investigate_label_changed_cases_with_ollama,
    filter_review_items_for_human_approval,
    label_borderline_items_with_ollama,
    load_demo_config,
    load_demo_samples,
    prepare_agentic_training_cycle,
    reset_demo_to_baseline,
    retrain_demo_verifier,
    run_agentic_review_cycle,
    run_demo_inference,
)


APP_CSS = """
#borderline-review-table table {
  table-layout: auto;
  width: 100%;
}

#borderline-review-table th,
#borderline-review-table td {
  white-space: normal !important;
  word-break: break-word;
  overflow-wrap: anywhere;
}

#borderline-review-table th:nth-child(1),
#borderline-review-table td:nth-child(1) {
  width: 84px;
  max-width: 84px;
}

#borderline-review-table th:nth-child(2),
#borderline-review-table td:nth-child(2) {
  width: 96px;
  max-width: 96px;
}

#borderline-review-table th:nth-child(3),
#borderline-review-table td:nth-child(3) {
  width: 136px;
  max-width: 136px;
}

#borderline-review-table th:nth-child(4),
#borderline-review-table td:nth-child(4) {
  min-width: 360px;
}

#borderline-review-table th:nth-child(5),
#borderline-review-table td:nth-child(5),
#borderline-review-table th:nth-child(6),
#borderline-review-table td:nth-child(6),
#borderline-review-table th:nth-child(7),
#borderline-review-table td:nth-child(7) {
  width: 172px;
  max-width: 172px;
}

#agentic-review-table,
#agentic-review-table .wrap,
#agentic-review-table .table-wrap {
  max-width: 100% !important;
}

#agentic-review-table table {
  table-layout: fixed !important;
  width: 100% !important;
  max-width: 100% !important;
}

#agentic-review-table th,
#agentic-review-table td {
  white-space: normal !important;
  word-break: break-word !important;
  overflow-wrap: anywhere !important;
  vertical-align: top;
}

#agentic-review-table th:nth-child(1),
#agentic-review-table td:nth-child(1) {
  width: 54px !important;
  max-width: 54px !important;
}

#agentic-review-table th:nth-child(2),
#agentic-review-table td:nth-child(2) {
  width: 64px !important;
  max-width: 64px !important;
}

#agentic-review-table th:nth-child(3),
#agentic-review-table td:nth-child(3) {
  width: 82px !important;
  max-width: 82px !important;
}

#agentic-review-table th:nth-child(4),
#agentic-review-table td:nth-child(4) {
  width: 27% !important;
  max-width: 27% !important;
}

#agentic-review-table th:nth-child(5),
#agentic-review-table td:nth-child(5) {
  width: 34% !important;
  max-width: 34% !important;
}

#agentic-review-table th:nth-child(6),
#agentic-review-table td:nth-child(6),
#agentic-review-table th:nth-child(7),
#agentic-review-table td:nth-child(7),
#agentic-review-table th:nth-child(8),
#agentic-review-table td:nth-child(8) {
  width: 72px !important;
  max-width: 72px !important;
}

#agentic-summary-markdown,
#agentic-investigation-markdown,
#agentic-comparison-markdown {
  font-size: 0.98rem;
  line-height: 1.58;
  color: #1f2937;
}

#agentic-summary-markdown h2,
#agentic-investigation-markdown h3,
#agentic-comparison-markdown h3 {
  font-size: 1.28rem;
  font-weight: 900;
  color: #0f2847;
  margin: 0.2rem 0 0.6rem;
  letter-spacing: -0.005em;
}

#agentic-summary-markdown h3,
#agentic-investigation-markdown h4,
#agentic-comparison-markdown h4 {
  font-size: 0.84rem;
  font-weight: 900;
  color: #17423f;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  margin: 1rem 0 0.45rem;
  padding-bottom: 5px;
  border-bottom: 1px solid #e6e1d7;
}

#agentic-summary-markdown p,
#agentic-summary-markdown li,
#agentic-investigation-markdown p,
#agentic-investigation-markdown li,
#agentic-comparison-markdown p,
#agentic-comparison-markdown li {
  margin-bottom: 0.4rem;
}

#agentic-summary-markdown blockquote,
#agentic-investigation-markdown blockquote,
#agentic-comparison-markdown blockquote {
  margin: 0.55rem 0;
  padding: 10px 16px;
  border-left: 4px solid #0a57a8;
  border-radius: 10px;
  background: linear-gradient(90deg, #f4f8ff 0%, #fbfdff 100%);
  color: #132a57;
}

#agentic-summary-markdown blockquote strong,
#agentic-investigation-markdown blockquote strong {
  color: #07204f;
}

#agentic-summary-markdown hr,
#agentic-investigation-markdown hr,
#agentic-comparison-markdown hr {
  border: 0;
  border-top: 1px dashed #d9d4c6;
  margin: 0.9rem 0;
}

#agentic-investigation-markdown h4,
#agentic-comparison-markdown h4 {
  text-transform: none;
  letter-spacing: -0.005em;
  font-size: 1rem;
  color: #0f2847;
  border-bottom: 0;
  padding-bottom: 0;
  margin: 0.9rem 0 0.4rem;
}

#agentic-investigation-markdown h4::before,
#agentic-comparison-markdown h4::before {
  content: "";
  display: inline-block;
  width: 6px;
  height: 6px;
  border-radius: 999px;
  background: #0a57a8;
  margin-right: 10px;
  vertical-align: middle;
  transform: translateY(-2px);
}

.status-pill {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 3px 12px;
  border-radius: 999px;
  font-size: 0.78rem;
  font-weight: 900;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  vertical-align: 2px;
}

.status-pill::before {
  content: "";
  width: 7px;
  height: 7px;
  border-radius: 999px;
  background: currentColor;
  opacity: 0.75;
}

.status-pill.status-running {
  background: #fff4e1;
  color: #8b5a00;
  border: 1px solid #f2c67a;
}

.status-pill.status-completed {
  background: #e6f5eb;
  color: #1e6a43;
  border: 1px solid #a2d3b6;
}

.status-pill.status-awaiting {
  background: #eaf1ff;
  color: #1a3c78;
  border: 1px solid #a8bee2;
}

.status-pill.status-needs-review {
  background: #fdf1de;
  color: #8a4b00;
  border: 1px solid #f0c685;
}

.status-pill.status-error {
  background: #fdeaea;
  color: #8b1d1d;
  border: 1px solid #e8a1a1;
}

.status-pill.status-unknown {
  background: #eef2f7;
  color: #4a5a72;
  border: 1px solid #c9d3e1;
}

#agentic-tab-intro {
  font-size: 1.02rem;
  color: #1f2c46;
  margin: 4px 0 6px;
}

.supervisor-table-wrap {
  overflow-x: auto;
  margin-top: 0.55rem;
}

.supervisor-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.9rem;
}

.supervisor-table th,
.supervisor-table td {
  border: 1px solid #e6dfd2;
  padding: 9px 11px;
  vertical-align: top;
}

.supervisor-table th {
  background: #f7f1e7;
  color: #243b3b;
  font-weight: 800;
  text-align: left;
}

.supervisor-table details summary {
  cursor: pointer;
  color: #17423f;
  font-weight: 800;
}

.supervisor-detail {
  margin-top: 10px;
  padding: 10px 12px;
  border-radius: 10px;
  background: #fffaf1;
  border: 1px solid #eadfcd;
}

.supervisor-detail-rule {
  margin: 0 0 12px;
}

.supervisor-detail-rule ul {
  margin: 6px 0 0 18px;
}

.supervisor-transcript-text {
  max-height: 220px;
  overflow: auto;
  white-space: pre-wrap;
  background: #f8f8f6;
  border: 1px solid #e1ddd3;
  border-radius: 8px;
  padding: 10px;
}

#llm-review-results {
  line-height: 1.6;
}

#llm-review-shell {
  position: fixed;
  right: 20px;
  bottom: 20px;
  width: min(430px, calc(100vw - 32px));
  z-index: 1000;
}

#llm-review-panel {
  border-radius: 18px;
  box-shadow: 0 14px 32px rgba(15, 23, 42, 0.18);
}

#llm-review-chatbot {
  height: 320px;
  max-height: 320px;
  overflow-y: auto;
}

#llm-prompt-buttons button {
  white-space: normal;
}

#agentic-analysis-shell {
  position: fixed;
  right: 20px;
  bottom: 20px;
  width: min(450px, calc(100vw - 32px));
  z-index: 999;
}

#agentic-analysis-panel {
  border-radius: 18px;
  box-shadow: 0 14px 32px rgba(15, 23, 42, 0.18);
}

#agentic-analysis-chatbot {
  height: 340px;
  max-height: 340px;
  overflow-y: auto;
}

#agentic-prompt-buttons button {
  white-space: normal;
}

.app-hero {
  position: relative;
  overflow: hidden;
  display: grid;
  grid-template-columns: minmax(220px, 0.85fr) 1.6fr minmax(160px, 0.55fr);
  align-items: center;
  gap: 28px;
  padding: 26px 34px;
  margin-bottom: 18px;
  border-radius: 26px;
  background:
    linear-gradient(135deg, #03204b 0%, #063774 45%, #0a57a8 100%);
  color: #ffffff;
  box-shadow: 0 22px 48px rgba(3, 32, 75, 0.28);
  isolation: isolate;
}

.app-hero::before {
  content: "";
  position: absolute;
  inset: 0;
  background:
    radial-gradient(circle at 14% 110%, rgba(230, 0, 0, 0.22), transparent 42%),
    radial-gradient(circle at 92% -20%, rgba(255, 255, 255, 0.18), transparent 45%);
  pointer-events: none;
  z-index: -1;
}

.app-hero-brand {
  display: flex;
  align-items: center;
  gap: 14px;
}

.app-hero-logo-frame {
  flex: 0 0 auto;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  padding: 10px 16px;
  border-radius: 14px;
  background: #ffffff;
  box-shadow: 0 10px 22px rgba(0, 18, 48, 0.22);
}

.app-hero-logo-frame img {
  width: 108px;
  height: auto;
  display: block;
}

.app-hero-brand-meta {
  display: flex;
  flex-direction: column;
  gap: 3px;
  line-height: 1.2;
  min-width: 0;
}

.app-hero-eyebrow {
  color: #ffd57a;
  font-size: 0.74rem;
  font-weight: 900;
  letter-spacing: 0.16em;
  text-transform: uppercase;
}

.app-hero-team {
  color: #ffffff;
  font-size: 1.02rem;
  font-weight: 800;
  letter-spacing: 0.01em;
}

.app-hero-tagline {
  color: rgba(255, 255, 255, 0.78);
  font-size: 0.78rem;
  font-weight: 600;
  letter-spacing: 0.02em;
}

.app-hero-main {
  display: flex;
  flex-direction: column;
  gap: 10px;
  padding: 0 6px;
  min-width: 0;
}

.app-hero-kicker {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  align-self: flex-start;
  padding: 5px 12px;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.12);
  color: #ffd57a;
  font-size: 0.72rem;
  font-weight: 800;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  border: 1px solid rgba(255, 255, 255, 0.20);
}

.app-hero-kicker .dot {
  width: 7px;
  height: 7px;
  border-radius: 999px;
  background: #ff3a3a;
  box-shadow: 0 0 0 4px rgba(255, 58, 58, 0.22);
  animation: heroDotPulse 1.6s ease-in-out infinite;
}

.app-hero-title {
  margin: 0;
  color: #ffffff;
  font-size: clamp(1.8rem, 2.6vw, 2.65rem);
  font-weight: 900;
  letter-spacing: -0.025em;
  line-height: 1.05;
}

.app-hero-title .accent {
  background: linear-gradient(90deg, #ffd57a 0%, #ffffff 100%);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
}

.app-hero-subtitle {
  color: rgba(255, 255, 255, 0.88);
  font-size: 0.95rem;
  line-height: 1.55;
  max-width: 640px;
}

.app-hero-subtitle strong {
  color: #ffffff;
  font-weight: 800;
}

.app-hero-badges {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 2px;
}

.app-hero-badge {
  font-size: 0.73rem;
  font-weight: 700;
  letter-spacing: 0.04em;
  padding: 4px 11px;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.12);
  color: #ffffff;
  border: 1px solid rgba(255, 255, 255, 0.20);
}

.app-hero-meta {
  display: flex;
  flex-direction: column;
  gap: 10px;
  align-items: flex-end;
  text-align: right;
}

.app-hero-meta-card {
  min-width: 0;
  padding: 8px 12px;
  border-radius: 12px;
  background: rgba(255, 255, 255, 0.06);
  border: 1px solid rgba(255, 255, 255, 0.14);
  backdrop-filter: blur(4px);
  -webkit-backdrop-filter: blur(4px);
}

.app-hero-meta-label {
  color: rgba(255, 213, 122, 0.78);
  font-size: 0.56rem;
  font-weight: 800;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  margin-bottom: 2px;
}

.app-hero-meta-value {
  color: rgba(255, 255, 255, 0.92);
  font-size: 0.78rem;
  font-weight: 700;
  letter-spacing: 0.01em;
}

.app-hero-meta-note {
  color: rgba(255, 255, 255, 0.62);
  font-size: 0.66rem;
  font-weight: 500;
  line-height: 1.35;
  margin-top: 2px;
}

@keyframes heroDotPulse {
  0%, 100% { box-shadow: 0 0 0 4px rgba(255, 58, 58, 0.22); }
  50% { box-shadow: 0 0 0 7px rgba(255, 58, 58, 0.08); }
}

#transcript-source-row {
  margin-top: 16px;
  margin-bottom: 14px;
}

.transcript-source-card {
  min-height: 210px;
  padding: 18px !important;
  border: 1px solid #e7dfd2 !important;
  border-radius: 18px !important;
  background: linear-gradient(180deg, #fffefd 0%, #fbf7f0 100%) !important;
  box-shadow: 0 14px 34px rgba(79, 55, 26, 0.08);
}

.transcript-source-card > .form,
.transcript-source-card .form {
  border: 0 !important;
  background: transparent !important;
  padding: 0 !important;
}

.source-kicker {
  color: #b45f24;
  font-size: 0.78rem;
  font-weight: 800;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  margin-bottom: 4px;
}

.source-title {
  color: #2f2a26;
  font-size: 1.05rem;
  font-weight: 800;
  margin-bottom: 4px;
}

.source-copy {
  color: #6f665d;
  font-size: 0.9rem;
  line-height: 1.35;
  margin-bottom: 14px;
}

.source-note {
  color: #7a7067;
  font-size: 0.82rem;
  line-height: 1.35;
  margin-top: 10px;
}

.transcript-source-card button {
  width: 100%;
}

#upload-transcript-btn {
  width: 100%;
}

#upload-transcript-btn,
#upload-transcript-btn button {
  min-height: 52px;
  border: 0 !important;
  border-radius: 14px !important;
  background: linear-gradient(135deg, #2c5f5d 0%, #153c3f 100%) !important;
  color: #fff !important;
  font-weight: 800 !important;
  box-shadow: 0 10px 22px rgba(21, 60, 63, 0.18);
}

#upload-transcript-btn:hover,
#upload-transcript-btn button:hover {
  filter: brightness(1.05);
}

#agentic-run-btn,
#agentic-run-btn button {
  min-height: 52px;
  border: 0 !important;
  border-radius: 14px !important;
  background: linear-gradient(135deg, #0a57a8 0%, #003d7a 100%) !important;
  color: #ffffff !important;
  font-weight: 900 !important;
  letter-spacing: 0.02em;
  box-shadow: 0 14px 30px rgba(0, 61, 122, 0.28);
  transition: transform 160ms ease, filter 160ms ease, box-shadow 160ms ease;
}

#agentic-run-btn:hover,
#agentic-run-btn button:hover {
  filter: brightness(1.06);
  transform: translateY(-1px);
  box-shadow: 0 18px 34px rgba(0, 61, 122, 0.34);
}

#agentic-reset-btn,
#agentic-reset-btn button {
  min-height: 52px;
  border: 1.5px solid #b6c2d4 !important;
  border-radius: 14px !important;
  background: #ffffff !important;
  color: #1f2c46 !important;
  font-weight: 800 !important;
  box-shadow: 0 8px 18px rgba(15, 31, 58, 0.08);
  transition: all 160ms ease;
}

#agentic-reset-btn:hover,
#agentic-reset-btn button:hover {
  background: #f5f7fc !important;
  border-color: #8a9bb5 !important;
}

#agentic-select-all-btn,
#agentic-select-all-btn button {
  min-height: 46px;
  border: 0 !important;
  border-radius: 12px !important;
  background: linear-gradient(135deg, #ffd57a 0%, #e9a52c 100%) !important;
  color: #3b2a07 !important;
  font-weight: 900 !important;
  box-shadow: 0 10px 22px rgba(233, 165, 44, 0.26);
  transition: filter 160ms ease, transform 160ms ease;
}

#agentic-select-all-btn:hover,
#agentic-select-all-btn button:hover {
  filter: brightness(1.04);
  transform: translateY(-1px);
}

#agentic-investigate-review-btn,
#agentic-investigate-review-btn button {
  min-height: 46px;
  border: 1.5px solid #b3c7ec !important;
  border-radius: 12px !important;
  background: #f4f8ff !important;
  color: #1a3c78 !important;
  font-weight: 800 !important;
  box-shadow: 0 8px 18px rgba(26, 60, 120, 0.08);
  transition: all 160ms ease;
}

#agentic-investigate-review-btn:hover,
#agentic-investigate-review-btn button:hover {
  background: #eaf1ff !important;
  border-color: #6f95d4 !important;
}

#agentic-investigate-regression-btn,
#agentic-investigate-regression-btn button {
  min-height: 46px;
  border: 1.5px solid #d5c2e8 !important;
  border-radius: 12px !important;
  background: #faf5ff !important;
  color: #5b2c86 !important;
  font-weight: 800 !important;
  box-shadow: 0 8px 18px rgba(91, 44, 134, 0.08);
  transition: all 160ms ease;
}

#agentic-investigate-regression-btn:hover,
#agentic-investigate-regression-btn button:hover {
  background: #f4e9ff !important;
  border-color: #9d74c4 !important;
}

#agentic-continue-btn,
#agentic-continue-btn button {
  min-height: 52px;
  border: 0 !important;
  border-radius: 14px !important;
  background: linear-gradient(135deg, #0a57a8 0%, #003d7a 100%) !important;
  color: #ffffff !important;
  font-weight: 900 !important;
  letter-spacing: 0.02em;
  box-shadow: 0 12px 28px rgba(0, 61, 122, 0.28);
  transition: transform 160ms ease, filter 160ms ease, box-shadow 160ms ease;
}

#agentic-continue-btn:hover,
#agentic-continue-btn button:hover {
  filter: brightness(1.06);
  transform: translateY(-1px);
  box-shadow: 0 18px 34px rgba(0, 61, 122, 0.34);
}

#agentic-continue-btn[disabled],
#agentic-continue-btn button:disabled {
  background: #e5e7eb !important;
  color: #8a8f98 !important;
  box-shadow: none;
  transform: none;
}

.agentic-upload-row {
  margin: 14px 0 12px;
}

.agentic-upload-card {
  padding: 16px !important;
  border: 1px solid #e7dfd2 !important;
  border-radius: 18px !important;
  background: linear-gradient(180deg, #fffefd 0%, #fbf7f0 100%) !important;
  box-shadow: 0 12px 28px rgba(79, 55, 26, 0.07);
}

.agentic-upload-card > .form,
.agentic-upload-card .form {
  border: 0 !important;
  background: transparent !important;
  padding: 0 !important;
}

.agentic-upload-card input,
.agentic-upload-card button {
  min-height: 52px;
}

#agentic-progress {
  margin: 16px 0 20px;
}

@property --progress {
  syntax: '<percentage>';
  inherits: true;
  initial-value: 0%;
}

.agentic-progress-shell {
  --progress: 0%;
  --progress-end: 0%;
  --accent: #005eb8;
  position: relative;
  padding: 18px 22px 16px;
  border-radius: 22px;
  background: linear-gradient(145deg, #061a3a 0%, #0f2e5c 55%, #1a4786 100%);
  color: #ffffff;
  box-shadow: 0 20px 40px rgba(6, 26, 58, 0.28);
  isolation: isolate;
}

.agentic-progress-shell.is-running {
  animation: none;
}

@keyframes agenticAdvanceProgress {
  to { --progress: var(--progress-end); }
}

.agentic-progress-shell::before {
  content: "";
  position: absolute;
  inset: 0;
  border-radius: inherit;
  background:
    radial-gradient(circle at 18% 110%, rgba(230, 0, 0, 0.18), transparent 45%),
    radial-gradient(circle at 100% -10%, rgba(255, 213, 122, 0.16), transparent 42%);
  pointer-events: none;
  z-index: -1;
}

.agentic-progress-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 10px;
  margin-bottom: 12px;
}

.agentic-progress-title {
  font-size: 0.78rem;
  font-weight: 900;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: #ffd57a;
}

.agentic-progress-stage {
  display: inline-flex;
  align-items: center;
  gap: 10px;
  font-size: 0.9rem;
  font-weight: 800;
  color: #ffffff;
}

.agentic-progress-stage .label {
  padding: 4px 12px;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.14);
  border: 1px solid rgba(255, 255, 255, 0.22);
  letter-spacing: 0.02em;
}

.agentic-progress-stage .percent {
  font-variant-numeric: tabular-nums;
  color: #ffd57a;
  font-size: 1rem;
  font-weight: 900;
  min-width: 3.4ch;
  text-align: right;
}

.agentic-progress-rail {
  position: relative;
  height: 10px;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.14);
  overflow: hidden;
  margin-bottom: 14px;
}

.agentic-progress-fill {
  position: relative;
  display: block;
  width: var(--progress);
  height: 100%;
  border-radius: inherit;
  overflow: hidden;
  background: linear-gradient(90deg, #e60000 0%, #ffd57a 55%, #ffffff 100%);
  box-shadow: 0 0 16px rgba(255, 213, 122, 0.38);
  transition: width 460ms ease;
}

.agentic-progress-shell.is-running .agentic-progress-fill {
  transition: none;
}

.agentic-progress-shell.is-running .agentic-progress-fill::before {
  content: "";
  position: absolute;
  inset: 0;
  background: repeating-linear-gradient(
    -45deg,
    rgba(255, 255, 255, 0.32) 0 10px,
    rgba(255, 255, 255, 0) 10px 20px
  );
  animation: agenticStripeMarch 0.9s linear infinite;
  pointer-events: none;
}

.agentic-progress-shell.is-running .agentic-progress-stage::before {
  content: "";
  display: inline-block;
  width: 14px;
  height: 14px;
  border-radius: 999px;
  border: 2.5px solid rgba(255, 213, 122, 0.38);
  border-top-color: #ffd57a;
  animation: agenticSpinner 0.9s linear infinite;
  vertical-align: -2px;
}

@keyframes agenticStripeMarch {
  from { background-position: 0 0; }
  to { background-position: 28px 0; }
}

@keyframes agenticSpinner {
  to { transform: rotate(360deg); }
}

.agentic-progress-rail::after {
  content: "";
  position: absolute;
  inset: 0;
  background: linear-gradient(110deg, transparent 0%, rgba(255, 255, 255, 0.25) 45%, transparent 70%);
  transform: translateX(-100%);
  animation: agenticRailSweep 2.4s ease-in-out infinite;
  pointer-events: none;
}

.agentic-progress-track {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 10px;
}

.agentic-progress-step {
  display: flex;
  align-items: center;
  gap: 11px;
  padding: 12px 14px;
  border-radius: 14px;
  background: rgba(255, 255, 255, 0.06);
  border: 1px solid rgba(255, 255, 255, 0.14);
  color: rgba(255, 255, 255, 0.72);
  font-size: 0.88rem;
  font-weight: 700;
  letter-spacing: 0.01em;
  transition: transform 220ms ease, background 220ms ease, border-color 220ms ease, box-shadow 220ms ease;
}

.agentic-progress-step .step-index {
  flex: 0 0 auto;
  width: 28px;
  height: 28px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.18);
  color: #ffffff;
  font-size: 0.82rem;
  font-weight: 900;
  letter-spacing: 0.02em;
}

.agentic-progress-step .step-label {
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.agentic-progress-step.done {
  background: rgba(47, 140, 91, 0.18);
  border-color: rgba(47, 140, 91, 0.48);
  color: #d7f3e2;
}

.agentic-progress-step.done .step-index {
  background: #2f8c5b;
  color: #ffffff;
}

.agentic-progress-step.running {
  background: rgba(255, 213, 122, 0.18);
  border-color: rgba(255, 213, 122, 0.6);
  color: #ffffff;
  box-shadow: 0 12px 26px rgba(0, 18, 48, 0.38);
  transform: translateY(-2px);
}

.agentic-progress-step.running .step-index {
  background: #ffd57a;
  color: #2a1c00;
  animation: agenticStepPulse 1.25s ease-in-out infinite;
}

.agentic-progress-step.blocked {
  background: rgba(230, 0, 0, 0.18);
  border-color: rgba(230, 0, 0, 0.55);
  color: #ffd4d4;
}

.agentic-progress-step.blocked .step-index {
  background: #e60000;
  color: #ffffff;
}

@keyframes agenticStepPulse {
  0%, 100% {
    box-shadow: 0 0 0 0 rgba(255, 213, 122, 0.55);
  }
  50% {
    box-shadow: 0 0 0 9px rgba(255, 213, 122, 0);
  }
}

@keyframes agenticRailSweep {
  0% { transform: translateX(-100%); }
  100% { transform: translateX(100%); }
}

@media (max-width: 900px) {
  #llm-review-shell {
    right: 12px;
    bottom: 12px;
    width: min(100vw - 24px, 430px);
  }

  #agentic-analysis-shell {
    right: 12px;
    bottom: 12px;
    width: min(100vw - 24px, 450px);
  }

  .app-hero {
    grid-template-columns: 1fr;
    gap: 18px;
    padding: 22px 22px;
  }

  .app-hero-meta {
    align-items: flex-start;
    text-align: left;
  }

  .app-hero-meta-card {
    width: 100%;
    min-width: 0;
  }
}
"""

QUICK_REVIEW_PROMPTS = [
    "Summarize the current inference.",
    "Explain the borderline phrases.",
    "Suggest final labels for borderline phrases.",
]

AGENTIC_QUICK_PROMPTS = [
    "Summarize the current agentic run.",
    "Explain which borderline scores should move up or down.",
    "Recommend whether the retraining improved the result.",
]

DEFAULT_LLM_PANEL_MESSAGE = ""

DEFAULT_AGENTIC_CHAT_MESSAGE = (
    "Ask about the latest agentic run."
)

DEMO_RULE_DISPLAY_NAMES = {
    "101": "Identity verification",
    "102": "Change of Booking",
}

DEMO_CLAIM_DISPLAY_NAMES = {
    ("102", "mandatory", 0): "Disclosure 1: Change fee",
    ("102", "mandatory", 1): "Disclosure 2: Fare difference or travel credit",
}

DEMO_TRANSCRIPT_SCRIPTS_DIR = Path("data/results/demo/transcript_scripts")
AGENTIC_PROGRESS_STEPS = [
    ("review", "Agentic Review"),
    ("approval", "Human Approval"),
    ("retrain", "Retrain"),
    ("reinference", "Re-inference"),
]
AGENTIC_PROGRESS_COLORS = {
    "review": "#2f7fa3",
    "approval": "#c98916",
    "retrain": "#7952b3",
    "reinference": "#2b8a5e",
}
AGENTIC_PROGRESS_BOUNDS: dict[str, tuple[float, float]] = {
    "review": (0.0, 50.0),
    "approval": (50.0, 60.0),
    "retrain": (60.0, 90.0),
    "reinference": (90.0, 100.0),
}
AGENTIC_PROGRESS_RUNNING_FRACTION = 0.5
AGENTIC_REINFERENCE_PROGRESS_DWELL_SEC = 1.2
GRADIO_UPLOAD_PREFIX_RE = re.compile(r"^gradio_[0-9a-f]{16,}[_-]+", flags=re.IGNORECASE)


def _logo_data_uri() -> str:
    logo_path = Path("resources/images/uob_logo.png")
    try:
        encoded = base64.b64encode(logo_path.read_bytes()).decode("ascii")
    except OSError:
        return ""
    return f"data:image/png;base64,{encoded}"


def _hero_html() -> str:
    logo_uri = _logo_data_uri()
    logo_html = (
        f"<div class='app-hero-logo-frame'><img src='{logo_uri}' alt='UOB logo'></div>"
        if logo_uri
        else ""
    )
    return (
        "<div class='app-hero'>"
        "<div class='app-hero-brand'>"
        f"{logo_html}"
        "<div class='app-hero-brand-meta'>"
        "<div class='app-hero-eyebrow'>UOB AI Labs</div>"
        "<div class='app-hero-team'>Policy Compliance</div>"
        "<div class='app-hero-tagline'>Agentic AI Demo</div>"
        "</div>"
        "</div>"
        "<div class='app-hero-main'>"
        "<span class='app-hero-kicker'><span class='dot'></span>"
        "Agentic Workflow &middot; Internal Demo</span>"
        "<h1 class='app-hero-title'>Travel Agency "
        "<span class='accent'>Policy Studio</span></h1>"
        "<div class='app-hero-badges'>"
        "<span class='app-hero-badge'>Synthetic Data</span>"
        "<span class='app-hero-badge'>Human-in-the-loop</span>"
        "<span class='app-hero-badge'>Recursive Self-Improvement</span>"
        "</div>"
        "</div>"
        "<div class='app-hero-meta'>"
        "<div class='app-hero-meta-card'>"
        "<div class='app-hero-meta-label'>Prepared by</div>"
        "<div class='app-hero-meta-value'>Tai Do</div>"
        "<div class='app-hero-meta-note'>UOB AI Labs &middot; PBVA</div>"
        "</div>"
        "</div>"
        "</div>"
    )


def _extend_gradio_server_start_timeout(timeout_sec: float = 30.0) -> None:
    import gradio.http_server as http_server
    from gradio.exceptions import ServerFailedToStartError

    original = getattr(http_server.Server.run_in_thread, "__name__", "")
    if original == "run_in_thread_with_extended_timeout":
        return

    def run_in_thread_with_extended_timeout(self) -> None:
        self.thread = threading.Thread(target=self.run, daemon=True)
        if getattr(self, "reloader", None):
            self.watch_thread = threading.Thread(target=self.watch, daemon=True)
            self.watch_thread.start()
        self.thread.start()
        start = time.time()
        while not self.started:
            time.sleep(1e-3)
            if time.time() - start > timeout_sec:
                raise ServerFailedToStartError("Server failed to start. Please check that the port is available.")

    http_server.Server.run_in_thread = run_in_thread_with_extended_timeout


def _rule_display_name(disclaimer_id: str, evidence: dict[str, Any]) -> str:
    rule_id = str(disclaimer_id).strip()
    return DEMO_RULE_DISPLAY_NAMES.get(rule_id) or str(evidence.get("description", "")).strip()


def _iter_claim_evidence(evidence: dict[str, Any]) -> list[tuple[str, int, dict[str, Any]]]:
    claims = evidence.get("claims", {})
    if not isinstance(claims, dict):
        return []

    output: list[tuple[str, int, dict[str, Any]]] = []
    for claim_type in ("single", "mandatory", "standard"):
        claim_items = claims.get(claim_type, [])
        if not isinstance(claim_items, list):
            continue
        for order, claim in enumerate(claim_items):
            if isinstance(claim, dict):
                output.append((claim_type, order, claim))
    return output


def _format_evidence_lines(disclaimer_id: str, evidence: dict[str, Any]) -> list[str]:
    claim_items = _iter_claim_evidence(evidence)
    if not claim_items:
        score = float(evidence.get("verification_score") or 0.0)
        best_text = str(evidence.get("match_text", "")).strip() or "No matched text available."
        return [
            "",
            "**Evidence**",
            f"- **Best text:** {best_text}",
            f"- **Score:** `{score:.3f}`",
        ]

    lines = ["", "**Evidence**"]
    if len(claim_items) > 1:
        headers: list[str] = []
        anchor_cells: list[str] = []
        text_cells: list[str] = []
        score_cells: list[str] = []
        for claim_type, order, claim in claim_items:
            display_name = DEMO_CLAIM_DISPLAY_NAMES.get((str(disclaimer_id).strip(), claim_type, order), f"Disclosure {order + 1}")
            anchor = str(claim.get("anchor", "")).strip() or "No anchor available."
            best_text = str(claim.get("match_text", "")).strip() or "No matched text available."
            score = float(claim.get("verification_score") or 0.0)
            headers.append(html.escape(display_name))
            anchor_cells.append(f"**Anchor:** {html.escape(anchor)}")
            text_cells.append(f"**Best text:** {html.escape(best_text)}")
            score_cells.append(f"**Score:** `{score:.3f}`")

        lines.extend(
            [
                "| " + " | ".join(headers) + " |",
                "| " + " | ".join(["---"] * len(headers)) + " |",
                "| " + " | ".join(anchor_cells) + " |",
                "| " + " | ".join(text_cells) + " |",
                "| " + " | ".join(score_cells) + " |",
            ]
        )
        return lines

    for claim_type, order, claim in claim_items:
        display_name = DEMO_CLAIM_DISPLAY_NAMES.get((str(disclaimer_id).strip(), claim_type, order))
        anchor = str(claim.get("anchor", "")).strip() or "No anchor available."
        best_text = str(claim.get("match_text", "")).strip() or "No matched text available."
        score = float(claim.get("verification_score") or 0.0)

        if display_name:
            lines.extend(["", f"**{display_name}**"])
        lines.extend(
            [
                f"- **Anchor:** {anchor}",
                f"- **Best text:** {best_text}",
                f"- **Score:** `{score:.3f}`",
            ]
        )
    return lines


def _format_results(payload: dict[str, Any]) -> str:
    results = payload.get("results", {})
    lines = ["## Inference Results"]
    for disclaimer_id, result in results.items():
        status = result.get("status", "FAIL")
        evidence = result.get("evidence", {})
        display_name = _rule_display_name(str(disclaimer_id), evidence)
        heading = f"### Rule {disclaimer_id}: {display_name}" if display_name else f"### Rule {disclaimer_id}"
        lines.extend(["", heading, f"**Status:** `{status}`"])
        lines.extend(_format_evidence_lines(str(disclaimer_id), evidence))
        lines.extend(["", "---"])
    if len(lines) == 1:
        lines.append("- No inference output available.")
    return "\n".join(lines)


def _verifier_label(score: float) -> str:
    return "Pass" if float(score) >= 0.5 else "Fail"


def _label_to_display(value: Any, *, default: str = "") -> str:
    label = str(value or "").strip().lower()
    if not label:
        return default
    normalized = label.replace("_", "-")
    if normalized in {"pass", "passed", "positive", "compliant", "yes", "y", "true", "1"}:
        return "Pass"
    if normalized in {"fail", "failed", "negative", "non-compliant", "non compliant", "noncompliant", "no", "n", "false", "0"}:
        return "Fail"
    if normalized in {"skip", "skipped", "ambiguous", "unclear"}:
        return "Skip"
    if "non-compliant" in normalized or "non compliant" in normalized or "noncompliant" in normalized:
        return "Fail"
    if "compliant" in normalized:
        return "Pass"
    return str(value or "").strip()


def _display_to_training_label(value: Any) -> str:
    display = _label_to_display(value)
    if display == "Pass":
        return "Compliant"
    if display == "Fail":
        return "Non-Compliant"
    return ""


def _text_preview(text: Any, max_words: int = 12) -> str:
    words = str(text or "").strip().split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]) + " ..."


def _display_final_label(item: dict[str, Any]) -> str:
    return _label_to_display(item.get("final_label") or item.get("llm_label"), default="")


def _borderline_dataframe(items: list[dict[str, Any]]) -> pd.DataFrame:
    rows = [
        {
            "add": bool(item.get("approved", False)),
            "rule": item["disclaimer_id"],
            "score": round(float(item["verification_score"]), 4),
            "text": _text_preview(item.get("text", "")),
            "model_label": _verifier_label(float(item["verification_score"])),
            "qwen_label": _label_to_display(item.get("llm_label")),
            "final_label": _display_final_label(item),
        }
        for item in items
    ]
    return pd.DataFrame(rows)


def _agentic_review_dataframe(items: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in items:
        score = float(item.get("verification_score") or 0.0)
        is_generated = str(item.get("review_type", "")).strip() == "generated_synthetic"
        rows.append(
            {
                "add": bool(item.get("approved", False)),
                "rule": item["disclaimer_id"],
                "score": "" if is_generated else f"{score:.4f}",
                "anchor": str(item.get("anchor", "")).strip(),
                "text": str(item.get("text", "")).strip(),
                "model": "" if is_generated else _verifier_label(score),
                "qwen": _label_to_display(item.get("llm_label")),
                "human": _display_final_label(item),
            }
        )
    return pd.DataFrame(rows)




def _agentic_review_table_update(items: list[dict[str, Any]], *, max_height: int = 420):
    import gradio as gr

    dataframe = _agentic_review_dataframe(items)
    return gr.update(
        value=dataframe,
        max_height=max_height,
        row_count=max(len(dataframe), 1),
    )


def _agentic_comparison_dataframe(comparisons: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in comparisons:
        rows.append(
            {
                "transcript": str(item.get("transcript_id", "")).strip(),
                "rule": str(item.get("disclaimer_id", "")).strip(),
                "label": _label_to_display(item.get("final_label")),
                "before": round(float(item.get("before_score") or 0.0), 4),
                "after": round(float(item.get("after_score") or 0.0), 4),
                "target": str(item.get("target_direction", "")).strip(),
                "outcome": str(item.get("outcome", "")).strip(),
                "text": _text_preview(item.get("text", ""), max_words=14),
            }
        )
    return pd.DataFrame(rows)


def _format_agentic_progress(*, active: str | None = None, completed: set[str] | None = None, blocked: bool = False) -> str:
    completed = completed or set()
    step_keys = [key for key, _ in AGENTIC_PROGRESS_STEPS]
    label_by_key = dict(AGENTIC_PROGRESS_STEPS)
    completed_count = sum(1 for key in step_keys if key in completed)

    if active in step_keys:
        active_index = step_keys.index(active)
        current_index = active_index + 1
        current_label = label_by_key.get(active, "")
        start, end = AGENTIC_PROGRESS_BOUNDS.get(active, (0.0, 100.0))
        progress_percent = start + (end - start) * AGENTIC_PROGRESS_RUNNING_FRACTION
    elif completed_count:
        last_key = next((key for key in reversed(step_keys) if key in completed), None)
        if completed_count >= len(step_keys):
            current_index = len(step_keys)
            current_label = "Completed"
        else:
            current_index = min(completed_count + 1, len(step_keys))
            current_label = label_by_key.get(last_key, "")
        progress_percent = AGENTIC_PROGRESS_BOUNDS.get(last_key, (0.0, 0.0))[1] if last_key else 0.0
    else:
        current_index = 1
        current_label = "Awaiting start"
        progress_percent = 0.0

    progress_percent = min(100.0, max(0.0, progress_percent))
    accent = AGENTIC_PROGRESS_COLORS.get(
        active or next((key for key in reversed(step_keys) if key in completed), "review"),
        "#005eb8",
    )
    if blocked and active:
        current_label = f"{current_label} (blocked)"

    shell_classes = ["agentic-progress-shell"]
    if active and not blocked:
        shell_classes.append("is-running")
    if blocked:
        shell_classes.append("is-blocked")

    if active and active in AGENTIC_PROGRESS_BOUNDS and not blocked:
        stage_end = AGENTIC_PROGRESS_BOUNDS[active][1]
        progress_end = max(progress_percent, stage_end - 1.0)
    else:
        progress_end = progress_percent

    parts = [
        f"<div class='{' '.join(shell_classes)}' "
        f"style='--progress:{progress_percent:.1f}%; --progress-end:{progress_end:.1f}%; --accent:{html.escape(accent)};'>"
        "<div class='agentic-progress-header'>"
        "<div class='agentic-progress-title'>Agentic Loop Pipeline</div>"
        "<div class='agentic-progress-stage'>"
        f"<span class='label'>Step {current_index} of {len(step_keys)} &middot; {html.escape(current_label)}</span>"
        f"<span class='percent'>{progress_percent:.0f}%</span>"
        "</div>"
        "</div>"
        "<div class='agentic-progress-rail'><span class='agentic-progress-fill'></span></div>"
        "<div class='agentic-progress-track'>"
    ]
    for index, (key, label) in enumerate(AGENTIC_PROGRESS_STEPS, start=1):
        if key in completed:
            css_class = "done"
            glyph = "&#10003;"
        elif key == active:
            css_class = "blocked" if blocked else "running"
            glyph = "!" if blocked else str(index)
        else:
            css_class = "pending"
            glyph = str(index)
        parts.append(
            "<div class='agentic-progress-step "
            f"stage-{html.escape(key)} {css_class}'>"
            f"<span class='step-index'>{glyph}</span>"
            f"<span class='step-label'>{html.escape(label)}</span>"
            "</div>"
        )
    parts.append("</div></div>")
    return "".join(parts)


def _format_agentic_comparison_markdown(comparisons: list[dict[str, Any]] | None) -> str:
    comparisons = [item for item in comparisons or [] if isinstance(item, dict)]
    if not comparisons:
        return "### Before/After Score Comparison\nNo comparison available yet."

    lines = ["### Before/After Score Comparison"]
    for index, item in enumerate(comparisons, start=1):
        rule_id = str(item.get("disclaimer_id", "")).strip() or "unknown"
        human_label = _label_to_display(item.get("final_label")) or "Unknown"
        before_score = float(item.get("before_score") or 0.0)
        after_score = float(item.get("after_score") or 0.0)
        before_model = _verifier_label(before_score)
        after_model = _verifier_label(after_score)
        outcome = str(item.get("outcome", "")).strip().lower()
        anchor = str(item.get("anchor", "")).strip() or "No anchor available."
        phrase = str(item.get("text", "")).strip() or "No text available."

        if human_label in {"Pass", "Fail"} and after_model == human_label:
            status = "Success"
        elif outcome == "improved":
            status = "Improved, but still needs review"
        elif outcome == "unchanged":
            status = "Unchanged"
        else:
            status = "Needs review"

        lines.extend(
            [
                "",
                f"#### Case {index}: Rule {html.escape(rule_id)}",
                f"- **Status:** `{html.escape(status)}`",
                f"- **Anchor:** {html.escape(anchor)}",
                f"- **Text:** {html.escape(phrase)}",
                f"- **Label change:** model `{before_model}` -> human `{html.escape(human_label)}` -> model `{after_model}` after retrain",
                f"- **Score:** `{before_score:.4f}` -> `{after_score:.4f}`",
            ]
        )
    return "\n".join(lines)

def _row_checked(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"true", "t", "yes", "y", "1", "checked"}


def _score_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "" or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_human_label(value: Any) -> str:
    return _display_to_training_label(value)


def _records_from_table_value(table_value: Any) -> list[dict[str, Any]]:
    if table_value is None:
        return []
    if isinstance(table_value, pd.DataFrame):
        if table_value.empty:
            return []
        return table_value.to_dict(orient="records")
    if isinstance(table_value, list):
        if not table_value:
            return []
        if all(isinstance(row, dict) for row in table_value):
            return [dict(row) for row in table_value]
        records: list[dict[str, Any]] = []
        for row in table_value:
            if not isinstance(row, (list, tuple)):
                continue
            headers = (
                ["add", "transcript", "rule", "score", "text", "model_label", "qwen_label", "final_label"]
                if len(row) >= 8
                else ["add", "rule", "score", "text", "model_label", "qwen_label", "final_label"]
            )
            if len(row) == 8:
                headers = ["add", "rule", "score", "anchor", "text", "model", "qwen", "human"]
            records.append(dict(zip(headers, row)))
        return records
    if isinstance(table_value, dict):
        data = table_value.get("data")
        headers = table_value.get("headers") or table_value.get("columns")
        if isinstance(data, list) and isinstance(headers, list):
            header_names = [
                str(header.get("name", header.get("id", ""))).strip() if isinstance(header, dict) else str(header).strip()
                for header in headers
            ]
            return [dict(zip(header_names, row)) for row in data if isinstance(row, (list, tuple))]
        if isinstance(data, list):
            return _records_from_table_value(data)
    return []


def _merge_review_items(table_value: Any, items: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    base_items = [deepcopy(item) for item in (items or [])]
    records = _records_from_table_value(table_value)
    if not records:
        return []

    merged: list[dict[str, Any]] = []
    for index, row in enumerate(records):
        base = deepcopy(base_items[index]) if index < len(base_items) else {}
        score_value = row.get("score", row.get("verification_score"))
        qwen_value = row.get("qwen", row.get("qwen_label"))
        model_value = row.get("model", row.get("model_label", row.get("verifier_label")))
        final_value = row.get("human", row.get("final_label", row.get("final")))
        merged.append(
            {
                **base,
                "approved": _row_checked(row.get("add", row.get("add_to_training"))),
                "disclaimer_id": str(row.get("rule") or row.get("rule_id") or base.get("disclaimer_id", "")).strip(),
                "text": str(base.get("text", "")).strip(),
                "verification_score": _score_float(score_value, _score_float(base.get("verification_score"), 0.0)),
                "model_label": _normalize_human_label(model_value or base.get("model_label", "")),
                "llm_label": _normalize_human_label(qwen_value or base.get("llm_label", "")),
                "final_label": _normalize_human_label(
                    final_value
                    or model_value
                    or qwen_value
                    or base.get("final_label")
                    or base.get("llm_label")
                    or base.get("model_label")
                    or ""
                ),
            }
        )
    return merged


def _is_investigator_generated_review_item(item: dict[str, Any] | None) -> bool:
    if not isinstance(item, dict):
        return False
    review_type = str(item.get("review_type", "")).strip()
    source = str(item.get("source", "")).strip()
    transcript_id = str(item.get("transcript_id", "")).strip()
    return (
        review_type == "generated_synthetic"
        or source == "investigator_generated"
        or transcript_id.startswith("investigator_generated_")
    )


def _review_item_identity(item: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(item.get("disclaimer_id", "")).strip(),
        str(item.get("anchor", "")).strip(),
        str(item.get("text", "")).strip(),
        str(item.get("final_label", item.get("llm_label", ""))).strip(),
    )


def _refresh_investigator_generated_queue(
    review_items: list[dict[str, Any]] | None,
    generated_items: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    base_items = [
        dict(item)
        for item in (review_items or [])
        if isinstance(item, dict) and not _is_investigator_generated_review_item(item)
    ]
    generated_rows = [dict(item) for item in (generated_items or []) if isinstance(item, dict)]
    existing_keys = {_review_item_identity(item) for item in base_items}
    refreshed = [*base_items]
    for item in generated_rows:
        key = _review_item_identity(item)
        if key in existing_keys:
            continue
        refreshed.append(item)
        existing_keys.add(key)
    return refreshed


def _format_llm_review_summary(items: list[dict[str, Any]]) -> str:
    if not items:
        return DEFAULT_LLM_PANEL_MESSAGE

    lines = ["### LLM Review Suggestions"]
    for index, item in enumerate(items, start=1):
        label = _label_to_display(item.get("llm_label")) or "No suggestion"
        confidence = float(item.get("llm_confidence") or 0.0)
        rationale = str(item.get("llm_rationale", "")).strip() or "No rationale provided."
        phrase = str(item.get("text", "")).strip() or "No phrase available."
        anchor = str(item.get("anchor", "")).strip() or "No anchor available."
        lines.extend(
            [
                f"#### Review {index}",
                f"- Rule: `{item.get('disclaimer_id', '')}`",
                f"- Suggested label: `{label}`",
                f"- Confidence: `{confidence:.2f}`",
                f"- Anchor: {anchor}",
                f"- Phrase: {phrase}",
                f"- Rationale: {rationale}",
                "",
            ]
        )
    return "\n".join(lines).strip()


def _format_metric_snapshot(title: str, metrics: dict[str, Any] | None) -> list[str]:
    metrics = metrics if isinstance(metrics, dict) else {}
    if not metrics:
        return [f"### {title}", "- No metrics available."]
    return [
        f"### {title}",
        f"- Count: `{int(metrics.get('count', 0) or 0)}`",
        f"- Accuracy: `{float(metrics.get('accuracy', 0.0) or 0.0):.3f}`",
        f"- Precision: `{float(metrics.get('precision', 0.0) or 0.0):.3f}`",
        f"- Recall: `{float(metrics.get('recall', 0.0) or 0.0):.3f}`",
        f"- F1: `{float(metrics.get('f1', 0.0) or 0.0):.3f}`",
        f"- Macro F1: `{float(metrics.get('macro_f1', 0.0) or 0.0):.3f}`",
    ]


def _format_agentic_summary(summary: dict[str, Any]) -> str:
    if not isinstance(summary, dict) or not summary:
        return "## Agentic Loop\n- No run output available yet."

    lines = [
        "## Agentic Loop",
        f"- Status: `{str(summary.get('status', 'unknown')).strip()}`",
    ]
    message = str(summary.get("message", "")).strip()
    if message:
        lines.append(f"- Message: {message}")

    bootstrap = summary.get("bootstrap", {})
    if isinstance(bootstrap, dict) and bootstrap:
        lines.extend(
            [
                "",
                "### Bootstrap Gate",
                f"- Status: `{str(bootstrap.get('status', 'unknown')).strip()}`",
            ]
        )
        bootstrap_message = str(bootstrap.get("message", "")).strip()
        if bootstrap_message:
            lines.append(f"- Message: {bootstrap_message}")
        gate_metrics = bootstrap.get("gate_metrics")
        if isinstance(gate_metrics, dict):
            lines.extend(["", *_format_metric_snapshot("Synthetic Gate Metrics", gate_metrics)])

    if "incoming_count" in summary or "holdout_count" in summary:
        lines.extend(
            [
                "",
                "### Data Scope",
                f"- Incoming transcripts: `{int(summary.get('incoming_count', 0) or 0)}`",
                f"- Holdout transcripts: `{int(summary.get('holdout_count', 0) or 0)}`",
                f"- Reviewed anchors: `{int(summary.get('reviewed_count', 0) or 0)}`",
            ]
        )

    lines.extend(["", *_format_metric_snapshot("Holdout Metrics Before", summary.get("holdout_metrics_before"))])
    if isinstance(summary.get("holdout_metrics_after"), dict):
        lines.extend(["", *_format_metric_snapshot("Holdout Metrics After", summary.get("holdout_metrics_after"))])

    coverage = summary.get("coverage", {})
    if isinstance(coverage, dict) and coverage:
        lines.extend(
            [
                "",
                "### Coverage Augmentation",
                f"- Candidate corrected cases: `{int(coverage.get('candidate_case_count', 0) or 0)}`",
                f"- Coverage analyses: `{int(coverage.get('analysis_count', 0) or 0)}`",
                f"- New synthetic rows added this run: `{int(coverage.get('new_rows_count', 0) or 0)}`",
            ]
        )

    if "promoted" in summary:
        lines.extend(
            [
                "",
                "### Promotion",
                f"- Promoted candidate model: `{'yes' if bool(summary.get('promoted', False)) else 'no'}`",
            ]
        )

    return "\n".join(lines)


_STATUS_PILL_CLASSES = {
    "running": "running",
    "awaiting_human_approval": "awaiting",
    "awaiting-human-approval": "awaiting",
    "needs_review": "needs-review",
    "needs-review": "needs-review",
    "ready": "completed",
    "ok": "completed",
    "retrained": "completed",
    "completed": "completed",
    "error": "error",
    "blocked": "error",
}

_STATUS_PILL_LABELS = {
    "running": "Running",
    "awaiting_human_approval": "Awaiting human approval",
    "awaiting-human-approval": "Awaiting human approval",
    "needs_review": "Needs review",
    "needs-review": "Needs review",
    "completed": "Completed",
    "retrained": "Retrained",
    "ready": "Ready",
    "ok": "Ok",
    "error": "Error",
    "blocked": "Blocked",
}


def _status_pill_html(status: str) -> str:
    normalized = status.strip().lower()
    css = _STATUS_PILL_CLASSES.get(normalized, "unknown")
    label = _STATUS_PILL_LABELS.get(normalized, status.strip().replace("_", " ").title() or "Unknown")
    return f"<span class='status-pill status-{css}'>{html.escape(label)}</span>"


def _format_app_agentic_summary(summary: dict[str, Any]) -> str:
    if not isinstance(summary, dict) or not summary:
        return "## Agentic Loop\n_No run output yet._ Click **Start Agentic Review** to begin."

    status_raw = str(summary.get("status", "unknown")).strip()
    message = str(summary.get("message", "")).strip()

    lines = [
        "## Agentic Loop",
        f"**Status** &nbsp; {_status_pill_html(status_raw)}",
    ]
    if message:
        lines.extend(["", f"> **Agent message.** {message}"])

    stat_bullets: list[str] = []
    if "transcript_count" in summary:
        stat_bullets.append(f"- **{int(summary.get('transcript_count', 0) or 0)}** uploaded transcripts")
    if "borderline_count" in summary:
        stat_bullets.append(f"- **{int(summary.get('borderline_count', 0) or 0)}** borderline phrases")
    if "pass_review_count" in summary:
        stat_bullets.append(
            f"- **{int(summary.get('pass_review_count', 0) or 0)}** model-pass / Qwen-fail phrases"
        )
    if stat_bullets:
        lines.extend(["", "### Run snapshot", *stat_bullets])

    stage_status = summary.get("stage_status", [])
    if isinstance(stage_status, list) and stage_status:
        lines.extend(["", "### Agent steps"])
        for stage in stage_status:
            if not isinstance(stage, dict):
                continue
            agent = str(stage.get("agent", "Agent")).strip()
            status = str(stage.get("status", "unknown")).strip()
            stage_message = str(stage.get("message", "")).strip()
            tail = f" &middot; {stage_message}" if stage_message else ""
            lines.append(f"- **{agent}** &mdash; `{status}`{tail}")

    supervisor_summary = str(summary.get("supervisor_summary", "")).strip()
    if supervisor_summary or summary.get("before_payloads"):
        lines.extend(["", "### Supervisor summary"])
        table_html = _format_supervisor_summary_table(summary)
        if table_html:
            lines.append(table_html)
        elif supervisor_summary:
            for line in supervisor_summary.splitlines():
                stripped = line.strip()
                if not stripped:
                    lines.append("")
                elif stripped.startswith("Transcript "):
                    if lines and lines[-1] != "":
                        lines.append("")
                    lines.append(f"**{stripped}**")
                elif stripped.startswith("SupervisorAgent prepared"):
                    lines.append(f"**{stripped}**")
                elif stripped.endswith(":"):
                    lines.append(f"**{stripped}**")
                elif line.startswith("  "):
                    lines.append(f"  - {stripped}")
                else:
                    lines.append(f"- {stripped}")

    recommendation = str(summary.get("recommendation", "")).strip()
    if recommendation:
        lines.extend(["", "### Recommendation", f"> {recommendation}"])

    return "\n".join(lines)


def _payload_rule_score_and_status(payload: dict[str, Any], rule_id: str) -> tuple[float | None, str]:
    results = payload.get("results", {}) if isinstance(payload, dict) else {}
    result = results.get(str(rule_id), {}) if isinstance(results, dict) else {}
    if not isinstance(result, dict):
        return None, "UNKNOWN"
    status = str(result.get("status", "UNKNOWN")).strip().upper() or "UNKNOWN"
    evidence = result.get("evidence", {}) if isinstance(result.get("evidence", {}), dict) else {}
    score_value = evidence.get("verification_score")
    if score_value is None:
        claims = evidence.get("claims", {}) if isinstance(evidence, dict) else {}
        claim_scores: list[float] = []
        if isinstance(claims, dict):
            for claim_group in ("single", "mandatory", "standard"):
                for claim in claims.get(claim_group, []) or []:
                    if isinstance(claim, dict):
                        try:
                            claim_scores.append(float(claim.get("verification_score") or 0.0))
                        except (TypeError, ValueError):
                            pass
        score_value = min(claim_scores) if claim_scores else None
    try:
        return (float(score_value), status) if score_value is not None else (None, status)
    except (TypeError, ValueError):
        return None, status


def _format_score_cell(score: float | None, status: str) -> str:
    del status
    if score is None:
        return "-"
    return f"<code>{score:.3f}</code>"


def _format_payload_detail_html(payload: dict[str, Any]) -> str:
    results = payload.get("results", {}) if isinstance(payload, dict) else {}
    parts: list[str] = ["<div class='supervisor-detail'>"]
    if not isinstance(results, dict) or not results:
        parts.append("<p>No detailed inference result available.</p>")
    else:
        for rule_id, result in results.items():
            if not isinstance(result, dict):
                continue
            status = str(result.get("status", "UNKNOWN")).strip().upper() or "UNKNOWN"
            parts.append(
                "<div class='supervisor-detail-rule'>"
                f"<strong>Rule {html.escape(str(rule_id))}: {html.escape(status)}</strong>"
            )
            evidence = result.get("evidence", {}) if isinstance(result.get("evidence", {}), dict) else {}
            claims = evidence.get("claims", {}) if isinstance(evidence, dict) else {}
            parts.append("<ul>")
            if isinstance(claims, dict):
                wrote_claim = False
                for claim_group in ("single", "mandatory", "standard"):
                    claim_rows = claims.get(claim_group, [])
                    if not isinstance(claim_rows, list):
                        continue
                    for claim_order, claim in enumerate(claim_rows, start=1):
                        if not isinstance(claim, dict):
                            continue
                        wrote_claim = True
                        anchor_label = (
                            f"Anchor {claim_order}"
                            if str(rule_id).strip() == "102" and len(claim_rows) > 1
                            else "Anchor"
                        )
                        anchor = html.escape(str(claim.get("anchor", "") or "No anchor available."))
                        best_text = html.escape(str(claim.get("match_text", "") or "No matched text available."))
                        try:
                            score = float(claim.get("verification_score") or 0.0)
                        except (TypeError, ValueError):
                            score = 0.0
                        parts.append(
                            "<li>"
                            f"<strong>{html.escape(anchor_label)}:</strong> {anchor}<br>"
                            f"<strong>Best text:</strong> {best_text}<br>"
                            f"<strong>Score:</strong> <code>{score:.3f}</code>"
                            "</li>"
                        )
                if not wrote_claim:
                    best_text = html.escape(str(evidence.get("match_text", "") or "No matched text available."))
                    score, _ = _payload_rule_score_and_status(payload, str(rule_id))
                    score_text = f"{score:.3f}" if score is not None else "n/a"
                    parts.append(
                        "<li>"
                        f"<strong>Best text:</strong> {best_text}<br>"
                        f"<strong>Score:</strong> <code>{html.escape(score_text)}</code>"
                        "</li>"
                    )
            parts.append("</ul></div>")
    transcript = str(payload.get("transcript", "")).strip()
    if transcript:
        parts.append(
            "<details><summary>Show transcript text</summary>"
            f"<pre class='supervisor-transcript-text'>{html.escape(transcript)}</pre>"
            "</details>"
        )
    parts.append("</div>")
    return "".join(parts)


def _format_supervisor_summary_table(summary: dict[str, Any]) -> str:
    payloads = summary.get("before_payloads", []) if isinstance(summary, dict) else []
    payloads = [payload for payload in payloads if isinstance(payload, dict)]
    if not payloads:
        return ""

    rows = [
        "<div class='supervisor-table-wrap'>",
        "<table class='supervisor-table'>",
        "<thead><tr><th>Transcript</th><th>Rule 101</th><th>Rule 102</th><th>PASS</th></tr></thead>",
        "<tbody>",
    ]
    for index, payload in enumerate(payloads, start=1):
        transcript_id = str(payload.get("transcript_id", "")).strip() or f"transcript_{index}"
        score_101, status_101 = _payload_rule_score_and_status(payload, "101")
        score_102, status_102 = _payload_rule_score_and_status(payload, "102")
        compliant_rules = [rule for rule, status in (("101", status_101), ("102", status_102)) if status == "PASS"]
        compliant_text = ", ".join(compliant_rules) if compliant_rules else "None"
        details = _format_payload_detail_html(payload)
        rows.append(
            "<tr>"
            "<td>"
            f"<details><summary>Transcript {index}: {html.escape(transcript_id)}</summary>{details}</details>"
            "</td>"
            f"<td>{_format_score_cell(score_101, status_101)}</td>"
            f"<td>{_format_score_cell(score_102, status_102)}</td>"
            f"<td>{html.escape(compliant_text)}</td>"
            "</tr>"
        )
    rows.extend(["</tbody></table></div>"])
    return "".join(rows)


def _format_quality_report(report: dict[str, Any] | None) -> str:
    report = report if isinstance(report, dict) else {}
    audits = report.get("audits", [])
    if not report:
        return "### Synthetic Quality Report\n- No synthetic quality report available for this run."

    lines = [
        "### Synthetic Quality Report",
        f"- Status: `{str(report.get('status', 'unknown')).strip()}`",
        f"- Anchors below threshold: `{int(report.get('anchors_below_threshold', 0) or 0)}`",
    ]
    if "macro_f1" in report:
        lines.append(f"- Gate macro F1: `{float(report.get('macro_f1', 0.0) or 0.0):.3f}`")

    if isinstance(audits, list) and audits:
        for index, audit in enumerate(audits[:5], start=1):
            lines.extend(
                [
                    "",
                    f"#### Audit {index}",
                    f"- Rule: `{str(audit.get('disclaimer_id', '')).strip()}`",
                    f"- Anchor F1: `{float(audit.get('f1', 0.0) or 0.0):.3f}`",
                    f"- Coverage: `{str(audit.get('coverage_status', 'unknown')).strip()}`",
                    f"- Positive note: {str(audit.get('positive_quality_note', '')).strip() or 'No note.'}",
                    f"- Negative note: {str(audit.get('negative_quality_note', '')).strip() or 'No note.'}",
                    f"- Reason: {str(audit.get('reason', '')).strip() or 'No reason provided.'}",
                ]
            )
    return "\n".join(lines)


def _format_coverage_report(coverage: dict[str, Any] | None) -> str:
    coverage = coverage if isinstance(coverage, dict) else {}
    analyses = coverage.get("analyses", [])
    if not coverage:
        return "### Coverage Analysis\n- No coverage analysis available for this run."

    lines = [
        "### Coverage Analysis",
        f"- Candidate corrected cases: `{int(coverage.get('candidate_case_count', 0) or 0)}`",
        f"- New synthetic rows added: `{int(coverage.get('new_rows_count', 0) or 0)}`",
    ]
    if isinstance(analyses, list) and analyses:
        for index, item in enumerate(analyses[:6], start=1):
            lines.extend(
                [
                    "",
                    f"#### Case {index}",
                    f"- Rule: `{str(item.get('disclaimer_id', '')).strip()}`",
                    f"- Type: `{str(item.get('correction_type', '')).strip() or 'n/a'}`",
                    f"- Coverage: `{str(item.get('coverage_status', 'unknown')).strip()}`",
                    f"- Reviewed phrase: {str(item.get('reviewed_phrase', '')).strip() or 'No phrase available.'}",
                    f"- Reason: {str(item.get('reason', '')).strip() or 'No reason provided.'}",
                ]
            )
            variants = item.get("generated_variants", [])
            if isinstance(variants, list) and variants:
                lines.append(f"- Generated variants: {' | '.join(str(variant).strip() for variant in variants if str(variant).strip())}")
    return "\n".join(lines)


def _format_regression_investigation_report(summary: dict[str, Any] | None) -> str:
    summary = summary if isinstance(summary, dict) else {}
    comparisons = [
        item
        for item in summary.get("comparisons", [])
        if isinstance(item, dict) and str(item.get("outcome", "")).strip() == "regressed"
    ]
    diagnosis = summary.get("score_regression_diagnosis", {})
    analyses = diagnosis.get("analyses", []) if isinstance(diagnosis, dict) else []
    analyses = [item for item in analyses if isinstance(item, dict)]

    if not comparisons:
        return (
            "### InvestigatorAgent Diagnosis\n"
            "**All approved phrases moved in the right direction.** No regression investigation is needed."
        )

    if not analyses:
        return (
            "### InvestigatorAgent Diagnosis\n"
            f"**Found {len(comparisons)} regressed approved phrase(s)**, but no agent diagnosis is available yet. "
            "Click again after re-inference completes, or rerun the retrain-and-compare step."
        )

    lines = [
        "### InvestigatorAgent Diagnosis",
        f"**Found {len(comparisons)} approved phrase(s)** that moved in the wrong direction after retraining.",
    ]
    for index, item in enumerate(analyses, start=1):
        rule_id = str(item.get("disclaimer_id", "")).strip() or "unknown"
        anchor = _text_preview(item.get("anchor", ""), max_words=32) or "No anchor available."
        phrase = _text_preview(item.get("phrase", ""), max_words=32) or "No phrase available."
        before_score = float(item.get("before_score") or item.get("score") or 0.0)
        after_score = float(item.get("after_score") or 0.0)
        target = str(item.get("target_direction", "")).strip() or "n/a"
        outcome = str(item.get("investigator_outcome", "")).strip() or "needs_review"
        recommendation = (
            str(item.get("recommendation", "")).strip()
            or "Review this anchor's synthetic coverage before another retrain."
        )

        lines.extend(
            [
                "",
                "---",
                f"#### Case {index} &middot; Rule {rule_id}",
                f"- **Score movement:** `{before_score:.4f}` -> `{after_score:.4f}` &nbsp; (target direction: `{target}`)",
                f"- **Diagnosis:** `{outcome}`",
                f"- **Anchor:** {anchor}",
                f"- **Phrase:** {phrase}",
                f"- **Recommended action:** {recommendation}",
            ]
        )
    return "\n".join(lines)


def _format_review_case_investigation_report(report: dict[str, Any] | None) -> str:
    report = report if isinstance(report, dict) else {}
    analyses = report.get("analyses", []) if isinstance(report.get("analyses", []), list) else []
    analyses = [item for item in analyses if isinstance(item, dict)]
    if not analyses:
        return (
            "### InvestigatorAgent Diagnosis\n"
            "No model/human label disagreement is available for dataset investigation yet. "
            "Run Agentic Review first, or keep the human label as **Fail** for model-pass cases before investigating."
        )

    lines = [
        "### InvestigatorAgent Diagnosis",
        f"**Inspected {len(analyses)} model/human disagreement case(s)** against the synthetic dataset.",
    ]
    for index, item in enumerate(analyses, start=1):
        rule_id = str(item.get("disclaimer_id", "")).strip() or "unknown"
        anchor = _text_preview(item.get("anchor", ""), max_words=34) or "No anchor available."
        phrase = _text_preview(item.get("phrase", ""), max_words=34) or "No phrase available."
        tags = [str(tag).strip() for tag in item.get("cause_tags", []) if str(tag).strip()]
        cause = ", ".join(tags) if tags else "needs dataset review"
        solution_steps = [str(step).strip() for step in item.get("solution_steps", []) if str(step).strip()]
        action = " ".join(solution_steps) if solution_steps else str(item.get("recommendation", "")).strip()
        label_change = str(item.get("label_change", "")).strip() or "model/human disagreement"
        outcome = str(item.get("investigator_outcome") or cause).strip()

        lines.extend(
            [
                "",
                "---",
                f"#### Case {index} &middot; Rule {rule_id}",
                f"- **Label change:** `{label_change}`",
                f"- **LLM outcome:** `{outcome}`",
                f"- **Anchor:** {anchor}",
                f"- **Phrase:** {phrase}",
                f"- **Recommended action:** {action or 'Add the reviewed phrase, then consider targeted variants if coverage is thin.'}",
            ]
        )
        generated_samples = [str(sample).strip() for sample in item.get("generated_samples", []) if str(sample).strip()]
        if generated_samples:
            lines.append("")
            lines.append("> **Generated phrasings by DataGeneratorAgent**")
            for sample in generated_samples:
                lines.append(f"> - {sample}")
    return "\n".join(lines)


def _artifact_json(summary: dict[str, Any], key: str, default: Any) -> Any:
    artifacts = summary.get("artifacts", {})
    if not isinstance(artifacts, dict):
        return default
    path = artifacts.get(key)
    if not path:
        return default
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def _extract_uploaded_paths(upload_value: Any) -> list[Path]:
    if upload_value is None:
        return []
    values = upload_value if isinstance(upload_value, list) else [upload_value]
    paths: list[Path] = []
    for item in values:
        if item is None:
            continue
        if isinstance(item, str):
            paths.append(Path(item))
            continue
        if isinstance(item, dict):
            for key in ("path", "name"):
                value = item.get(key)
                if isinstance(value, str) and value:
                    paths.append(Path(value))
                    break
            continue
        for attr in ("path", "name"):
            value = getattr(item, attr, None)
            if isinstance(value, str) and value:
                paths.append(Path(value))
                break
    return paths


def _clean_uploaded_filename(path: Path) -> str:
    name = path.name
    cleaned = GRADIO_UPLOAD_PREFIX_RE.sub("", name).strip()
    if cleaned and cleaned != name:
        return cleaned
    parent_match = re.match(r"^gradio_[0-9a-f]{16,}$", path.parent.name, flags=re.IGNORECASE)
    if parent_match:
        return name
    return cleaned or name


def _copy_uploaded_paths_to_temp(uploaded_paths: list[Path], temp_dir: Path) -> None:
    used_names: set[str] = set()
    for index, path in enumerate(uploaded_paths, start=1):
        filename = _clean_uploaded_filename(path)
        stem = Path(filename).stem or f"transcript_{index}"
        suffix = Path(filename).suffix or path.suffix
        candidate = f"{stem}{suffix}"
        counter = 2
        while candidate in used_names or (temp_dir / candidate).exists():
            candidate = f"{stem}_{counter}{suffix}"
            counter += 1
        used_names.add(candidate)
        shutil.copy2(path, temp_dir / candidate)


def _materialize_incoming_source(
    transcript_text: str,
    upload_value: Any,
    folder_upload_value: Any = None,
    active_source: str | None = None,
) -> tuple[str | None, list[Path], str | None]:
    cleanup_paths: list[Path] = []
    uploaded_paths = _extract_uploaded_paths(upload_value)
    folder_paths = _extract_uploaded_paths(folder_upload_value)
    if uploaded_paths and folder_paths:
        active_source = str(active_source or "").strip().lower()
        if active_source == "file":
            folder_paths = []
        elif active_source == "folder":
            uploaded_paths = []
        else:
            return None, cleanup_paths, "Use either file upload or folder upload for one run, not both."

    if folder_paths:
        txt_paths = [path for path in folder_paths if path.suffix.lower() == ".txt"]
        if not txt_paths:
            return None, cleanup_paths, "The uploaded folder must contain at least one `.txt` transcript file."
        temp_dir = Path(tempfile.mkdtemp(prefix="policy_compliance_agent_incoming_"))
        cleanup_paths.append(temp_dir)
        _copy_uploaded_paths_to_temp(txt_paths, temp_dir)
        return str(temp_dir), cleanup_paths, None

    if uploaded_paths:
        suffixes = {path.suffix.lower() for path in uploaded_paths}
        if len(uploaded_paths) == 1 and suffixes == {".json"}:
            return str(uploaded_paths[0]), cleanup_paths, None
        if suffixes != {".txt"}:
            return None, cleanup_paths, "Upload either a single `.txt`/`.json` file, multiple `.txt` files, or one folder of `.txt` files."
        temp_dir = Path(tempfile.mkdtemp(prefix="policy_compliance_agent_incoming_"))
        cleanup_paths.append(temp_dir)
        _copy_uploaded_paths_to_temp(uploaded_paths, temp_dir)
        return str(temp_dir), cleanup_paths, None

    transcript = str(transcript_text or "").strip()
    if transcript:
        temp_dir = Path(tempfile.mkdtemp(prefix="policy_compliance_agent_incoming_"))
        cleanup_paths.append(temp_dir)
        transcript_path = temp_dir / "interactive_transcript.txt"
        transcript_path.write_text(transcript, encoding="utf-8")
        return str(transcript_path), cleanup_paths, None

    return None, cleanup_paths, "Upload transcript file(s) or a folder of `.txt` transcripts before running the agentic loop."


def _materialize_holdout_source(upload_value: Any) -> tuple[str | None, str | None]:
    uploaded_paths = _extract_uploaded_paths(upload_value)
    if not uploaded_paths:
        return None, None
    if len(uploaded_paths) != 1:
        return None, "Upload a single holdout JSON file when overriding the default holdout set."
    if uploaded_paths[0].suffix.lower() != ".json":
        return None, "The holdout override must be a `.json` file."
    return str(uploaded_paths[0]), None


def _load_uploaded_transcript(upload_value: Any) -> tuple[str, str]:
    uploaded_paths = _extract_uploaded_paths(upload_value)
    if not uploaded_paths:
        return "", "Choose a `.txt` transcript file first."
    if len(uploaded_paths) != 1:
        return "", "Upload one transcript `.txt` file at a time in Transcript Review."

    path = uploaded_paths[0]
    if path.suffix.lower() != ".txt":
        return "", "Transcript Review accepts `.txt` scripts only."
    try:
        transcript = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        return "", f"Could not read uploaded transcript: {exc}"
    if not transcript:
        return "", f"Uploaded script `{path.name}` is empty."
    return transcript, f"Loaded uploaded script: {path.name}"


def _chat_context(payload: dict[str, Any], review_items: list[dict[str, Any]]) -> dict[str, Any]:
    guidance = (
        "This is a policy-review demo for two travel-agency/helpdesk-style controls. "
        "The inference payload contains rule-level pass/fail decisions, scores, anchors, and matched evidence text. "
        "The review_items list contains the current human-review rows: all borderline phrases plus any model-pass phrase that Qwen labels Fail. "
        "After Qwen labeling, rows may include qwen/llm labels, confidence, and rationale. "
        "Use only this context when answering, clearly distinguish the model score from Qwen or human review labels, and avoid inventing evidence."
    )
    return {
        "assistant_context": guidance,
        "inference": payload if isinstance(payload, dict) else {},
        "review_items": review_items if isinstance(review_items, list) else [],
    }


def _append_chat_messages(
    history: list[dict[str, Any]] | None,
    *,
    user_text: str,
    assistant_text: str,
) -> list[dict[str, str]]:
    messages = _normalized_chat_messages(history)
    messages.append({"role": "user", "content": user_text})
    messages.append({"role": "assistant", "content": assistant_text})
    return messages


def _normalized_chat_messages(history: list[dict[str, Any]] | None) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for item in history or []:
        if isinstance(item, dict) and isinstance(item.get("role"), str) and isinstance(item.get("content"), str):
            messages.append({"role": item["role"], "content": item["content"]})
    return messages


def _append_user_message(history: list[dict[str, Any]] | None, user_text: str) -> list[dict[str, str]]:
    messages = _normalized_chat_messages(history)
    messages.append({"role": "user", "content": user_text})
    return messages


def _preset_prompt(
    prompt: str,
    payload: dict[str, Any],
    review_items: list[dict[str, Any]],
    history: list[dict[str, Any]] | None,
    config: dict[str, Any] | None = None,
):
    try:
        answer = answer_demo_question(
            prompt,
            inference_payload=_chat_context(payload, review_items),
            chat_history=history,
            config=config,
        )
    except Exception as exc:
        answer = f"LLM request failed: {exc}"
    messages = _append_chat_messages(history, user_text=prompt, assistant_text=answer)
    return messages, messages, ""


def _preset_agentic_prompt(
    prompt: str,
    summary_payload: dict[str, Any],
    history: list[dict[str, Any]] | None,
    config: dict[str, Any] | None = None,
):
    try:
        answer = answer_agentic_question(prompt, summary_payload=summary_payload, chat_history=history, config=config)
    except Exception as exc:
        answer = f"LLM request failed: {exc}"
    return _append_chat_messages(history, user_text=prompt, assistant_text=answer), ""


def build_demo_app(config_path: str | None = None):
    try:
        import gradio as gr
    except ModuleNotFoundError as exc:
        raise RuntimeError("Gradio is required for the demo app. Install with `pip install .[demo]`.") from exc

    config = load_demo_config(config_path)
    samples = load_demo_samples(config)
    sample_lookup = {item["transcript_id"]: item for item in samples if isinstance(item, dict) and item.get("transcript_id")}
    approved_summary = approve_demo_examples([], config=config)

    def load_sample(sample_id: str) -> tuple[str, str]:
        sample = sample_lookup.get(sample_id, {})
        transcript = str(sample.get("transcript", "")).strip()
        title = str(sample.get("title", sample_id)).strip()
        return transcript, f"Loaded sample: {title}" if transcript else "Sample not found."

    def load_uploaded_script(upload_value: Any) -> tuple[str, str]:
        return _load_uploaded_transcript(upload_value)

    def run_inference_ui(transcript: str) -> tuple[
        str,
        pd.DataFrame,
        dict[str, Any],
        str,
        list[dict[str, Any]],
        list[dict[str, Any]],
        str,
        str,
        str,
        list[dict[str, str]],
        list[dict[str, str]],
    ]:
        payload = run_demo_inference(transcript, config=config)
        candidate_items = payload.get("review_items", payload.get("borderline_items", []))
        display_items = filter_review_items_for_human_approval(candidate_items)
        dataframe = _borderline_dataframe(display_items)
        borderline_count = sum(1 for item in candidate_items if item.get("review_type") == "borderline")
        pass_count = sum(1 for item in candidate_items if item.get("review_type") == "pass")
        summary = (
            f"Found {borderline_count} borderline phrase(s) and {pass_count} model-pass phrase(s) for Qwen review. "
            "The table shows all borderline phrases; after Qwen labeling it will also show any model-pass phrase that Qwen marks Fail."
        )
        return (
            _format_results(payload),
            dataframe,
            payload,
            summary,
            display_items,
            candidate_items,
            DEFAULT_LLM_PANEL_MESSAGE,
            DEFAULT_LLM_PANEL_MESSAGE,
            "",
            [],
            [],
        )

    def label_borderline_ui(candidate_items: list[dict[str, Any]] | None):
        candidate_items = candidate_items if isinstance(candidate_items, list) else []
        display_items = filter_review_items_for_human_approval(candidate_items)
        if not candidate_items:
            yield (
                _borderline_dataframe([]),
                [],
                [],
                DEFAULT_LLM_PANEL_MESSAGE,
                DEFAULT_LLM_PANEL_MESSAGE,
                "No phrases to label.",
            )
            return

        yield (
            _borderline_dataframe(display_items),
            display_items,
            candidate_items,
            DEFAULT_LLM_PANEL_MESSAGE,
            DEFAULT_LLM_PANEL_MESSAGE,
            f"Labeling {len(candidate_items)} review candidate(s) with Qwen3-4B. This can take a few seconds the first time Ollama loads the model...",
        )
        try:
            labeled_candidates = label_borderline_items_with_ollama(candidate_items, config=config)
        except Exception as exc:
            error_message = f"### LLM Review Suggestions\nLLM request failed: {exc}"
            yield (
                _borderline_dataframe(display_items),
                display_items,
                candidate_items,
                error_message,
                error_message,
                f"LLM request failed: {exc}",
            )
            return
        labeled_display_items = filter_review_items_for_human_approval(labeled_candidates)
        summary = _format_llm_review_summary(labeled_display_items)
        borderline_count = sum(1 for item in labeled_display_items if item.get("review_type") == "borderline")
        pass_qwen_fail_count = sum(
            1
            for item in labeled_display_items
            if item.get("review_type") == "pass" and _label_to_display(item.get("llm_label")) == "Fail"
        )
        yield (
            _borderline_dataframe(labeled_display_items),
            labeled_display_items,
            labeled_candidates,
            summary,
            DEFAULT_LLM_PANEL_MESSAGE,
            f"Labeled {len(labeled_candidates)} phrase(s) with Ollama. Showing {borderline_count} borderline phrase(s) and {pass_qwen_fail_count} model-pass/Qwen-fail phrase(s).",
        )

    def approve_ui(dataframe: pd.DataFrame, review_items: list[dict[str, Any]] | None) -> tuple[str, Any]:
        if not _records_from_table_value(dataframe):
            summary = approve_demo_examples([], config=config)
            return "No labeled rows available to save.", gr.update(interactive=summary["ready_to_retrain"])

        approved_items = _merge_review_items(dataframe, review_items)
        selected_count = sum(1 for item in approved_items if bool(item.get("approved")))
        trainable_count = sum(
            1
            for item in approved_items
            if bool(item.get("approved")) and str(item.get("final_label", "")).strip() in {"Compliant", "Non-Compliant"}
        )
        summary = approve_demo_examples(approved_items, config=config, replace_existing=True)
        if selected_count == 0:
            message = (
                "No rows were selected. Tick `add` for each sample you want to add, "
                "then click Save Selected Examples again. "
                f"Latest-run stored examples: {summary['approved_count']}. "
                f"Retrain ready: {'yes' if summary['ready_to_retrain'] else 'no'}."
            )
        elif trainable_count == 0:
            message = (
                "No training examples were saved because the selected row(s) were marked `Skip` or did not have a valid `Pass`/`Fail` final label. "
                f"Latest-run stored examples: {summary['approved_count']}. "
                f"Retrain ready: {'yes' if summary['ready_to_retrain'] else 'no'}."
            )
        elif summary["added_count"] == 0:
            message = (
                "No new labeled examples were added. "
                "The current rows may already exist in the latest-run training store. "
                f"Latest-run stored examples: {summary['approved_count']}. "
                f"Retrain ready: {'yes' if summary['ready_to_retrain'] else 'no'}."
            )
        else:
            message = (
                f"Saved {summary['added_count']} labeled example(s) to the training store. "
                f"Latest-run stored examples: {summary['approved_count']}. "
                f"Retrain ready: {'yes' if summary['ready_to_retrain'] else 'no'}."
            )
        if not summary["ready_to_retrain"] and summary.get("readiness_note"):
            message = f"{message} {summary['readiness_note']}"
        return message, gr.update(interactive=summary["ready_to_retrain"])

    def retrain_ui() -> str:
        result = retrain_demo_verifier(config=config)
        return json.dumps(result, indent=2)

    def reset_base_models_ui():
        try:
            result = reset_demo_to_baseline(config=config)
            message = (
                "Synthetic-only base models are active again. "
                "Cleared approved examples, augmented data, and retrained model versions."
            )
            status = "reset"
        except Exception as exc:
            result = {"status": "error", "message": str(exc)}
            message = f"Base-model reset failed: {exc}"
            status = "error"
        summary = {
            "status": status,
            "message": message,
            "recommendation": "Upload transcript(s), then start agentic review from the base models.",
            "reset": result,
            "stage_status": [
                {
                    "agent": "SupervisorAgent",
                    "status": status,
                    "message": message,
                }
            ],
        }
        return (
            _format_agentic_progress(),
            _format_app_agentic_summary(summary),
            _agentic_review_dataframe([]),
            _format_agentic_comparison_markdown([]),
            json.dumps(result, indent=2, ensure_ascii=False),
            summary,
            [],
            "",
            gr.update(interactive=False),
        )

    def chat_ui(
        question: str,
        payload: dict[str, Any],
        review_items: list[dict[str, Any]] | None,
        history: list[dict[str, Any]] | None,
    ):
        question_text = str(question or "").strip()
        if not question_text:
            current_history = _normalized_chat_messages(history)
            yield current_history, current_history, ""
            return

        pending_history = _append_user_message(history, question_text)
        yield pending_history, pending_history, ""

        try:
            answer = answer_demo_question(
                question_text,
                inference_payload=_chat_context(payload, review_items or []),
                chat_history=history,
                config=config,
            )
        except Exception as exc:
            answer = f"LLM request failed: {exc}"
        final_history = [*pending_history, {"role": "assistant", "content": answer}]
        yield final_history, final_history, ""

    def start_agentic_review_ui(
        transcript: str,
        incoming_uploads: Any,
        incoming_folder_uploads: Any,
        active_upload_source: str,
    ):
        incoming_source, cleanup_paths, incoming_error = _materialize_incoming_source(
            transcript,
            incoming_uploads,
            incoming_folder_uploads,
            active_source=active_upload_source,
        )
        if incoming_error:
            failure = {"status": "blocked", "message": incoming_error}
            yield (
                _format_agentic_progress(active="review", blocked=True),
                _format_app_agentic_summary(failure),
                _agentic_review_dataframe([]),
                _format_agentic_comparison_markdown([]),
                "{}",
                failure,
                [],
                "",
                gr.update(interactive=False),
            )
            return

        yield (
            _format_agentic_progress(active="review"),
            "## Agentic Loop\n- Status: `running`\n- Message: InferenceAgent is processing transcript(s), then Qwen will classify borderline and model-pass phrases.",
            _agentic_review_dataframe([]),
            _format_agentic_comparison_markdown([]),
            "{}",
            {},
            [],
            "",
            gr.update(interactive=False),
        )

        try:
            transcripts = load_incoming_transcripts(
                config=config,
                incoming_source=incoming_source,
            )
            summary = run_agentic_review_cycle(transcripts, config=config)
        except Exception as exc:
            for path in cleanup_paths:
                shutil.rmtree(path, ignore_errors=True)
            failure = {
                "status": "error",
                "message": f"Agentic loop failed: {exc}",
            }
            yield (
                _format_agentic_progress(active="review", blocked=True),
                _format_app_agentic_summary(failure),
                _agentic_review_dataframe([]),
                _format_agentic_comparison_markdown([]),
                json.dumps(failure, indent=2),
                failure,
                [],
                "",
                gr.update(interactive=False),
            )
            return
        finally:
            for path in cleanup_paths:
                shutil.rmtree(path, ignore_errors=True)

        review_items = summary.get("review_items", []) if isinstance(summary.get("review_items"), list) else []
        completed_steps = {"review"} if review_items else {"review", "approval", "retrain", "reinference"}
        yield (
            _format_agentic_progress(active="approval" if review_items else None, completed=completed_steps),
            _format_app_agentic_summary(summary),
            _agentic_review_dataframe(review_items),
            _format_agentic_comparison_markdown([]),
            json.dumps(summary, indent=2, ensure_ascii=False),
            summary,
            [],
            "",
            gr.update(interactive=bool(review_items)),
        )

    def continue_agentic_ui(dataframe: pd.DataFrame, summary_payload: dict[str, Any]):
        summary_payload = summary_payload if isinstance(summary_payload, dict) else {}
        review_items = summary_payload.get("review_items", [])
        approved_items = _merge_review_items(dataframe, review_items if isinstance(review_items, list) else [])
        running = {
            **summary_payload,
            "status": "running",
            "message": "TrainerAgent is saving approved rows, retraining from base models, then rerunning inference with the candidate model.",
        }
        yield (
            _format_agentic_progress(active="retrain", completed={"review", "approval"}),
            _format_app_agentic_summary(running),
            _agentic_review_table_update([], max_height=260),
            _format_agentic_comparison_markdown([]),
            json.dumps(running, indent=2, ensure_ascii=False),
            running,
            [],
            "",
            "",
        )

        try:
            retrain_state = prepare_agentic_training_cycle(summary_payload, approved_items, config=config)
        except Exception as exc:
            retrain_state = {
                "status": "error",
                "message": f"Agentic retraining failed: {exc}",
                "comparisons": [],
            }

        if retrain_state.get("status") != "retrained":
            yield (
                _format_agentic_progress(active="retrain", completed={"review", "approval"}, blocked=True),
                _format_app_agentic_summary(retrain_state),
                _agentic_review_table_update([], max_height=260),
                _format_agentic_comparison_markdown([]),
                json.dumps(retrain_state, indent=2, ensure_ascii=False),
                retrain_state,
                [],
                "",
                "",
            )
            return

        reinference_running = {
            **retrain_state,
            "status": "running",
            "message": "InferenceAgent is rerunning the uploaded transcript(s) with the newly trained candidate model.",
        }
        yield (
            _format_agentic_progress(active="reinference", completed={"review", "approval", "retrain"}),
            _format_app_agentic_summary(reinference_running),
            _agentic_review_table_update([], max_height=260),
            _format_agentic_comparison_markdown([]),
            json.dumps(reinference_running, indent=2, ensure_ascii=False),
            reinference_running,
            [],
            "",
            "",
        )
        time.sleep(AGENTIC_REINFERENCE_PROGRESS_DWELL_SEC)

        try:
            result = complete_agentic_reinference_cycle(retrain_state, config=config)
        except Exception as exc:
            result = {
                **retrain_state,
                "status": "error",
                "message": f"Agentic re-inference failed: {exc}",
                "comparisons": [],
            }

        progress = (
            _format_agentic_progress(completed={"review", "approval", "retrain", "reinference"})
            if result.get("status") in {"completed", "needs_review"}
            else _format_agentic_progress(active="retrain", completed={"review", "approval"}, blocked=True)
        )
        yield (
            progress,
            _format_app_agentic_summary(result),
            _agentic_review_table_update([], max_height=260),
            _format_agentic_comparison_markdown(result.get("comparisons", []) if isinstance(result, dict) else []),
            json.dumps(result, indent=2, ensure_ascii=False),
            result,
            [],
            "",
            "",
        )

    def agentic_chat_ui(
        question: str,
        summary_payload: dict[str, Any],
        history: list[dict[str, Any]] | None,
    ):
        question_text = str(question or "").strip()
        if not question_text:
            yield _normalized_chat_messages(history), ""
            return

        pending_history = _append_user_message(history, question_text)
        yield pending_history, ""

        try:
            answer = answer_agentic_question(
                question_text,
                summary_payload=summary_payload,
                chat_history=history,
                config=config,
            )
        except Exception as exc:
            answer = f"LLM request failed: {exc}"
        yield [*pending_history, {"role": "assistant", "content": answer}], ""

    def append_agentic_user_message(
        question: str,
        history: list[dict[str, Any]] | None,
    ) -> tuple[list[dict[str, str]], str, str]:
        question_text = str(question or "").strip()
        if not question_text:
            return _normalized_chat_messages(history), "", ""
        return _append_user_message(history, question_text), "", question_text

    def append_agentic_prompt_message(
        history: list[dict[str, Any]] | None,
        prompt: str,
    ) -> tuple[list[dict[str, str]], str, str]:
        return _append_user_message(history, prompt), "", prompt

    def answer_agentic_chat_message(
        question: str,
        summary_payload: dict[str, Any],
        history: list[dict[str, Any]] | None,
    ) -> list[dict[str, str]]:
        question_text = str(question or "").strip()
        current_history = _normalized_chat_messages(history)
        if not question_text:
            return current_history
        try:
            answer = answer_agentic_question(
                question_text,
                summary_payload=summary_payload,
                chat_history=current_history[:-1],
                config=config,
            )
        except Exception as exc:
            answer = f"LLM request failed: {exc}"
        return [*current_history, {"role": "assistant", "content": answer}]

    def select_all_agentic_review_rows_ui(
        dataframe: pd.DataFrame,
        summary_payload: dict[str, Any],
    ) -> tuple[Any, dict[str, Any], str]:
        summary_payload = summary_payload if isinstance(summary_payload, dict) else {}
        review_items = summary_payload.get("review_items", [])
        current_items = [dict(item) for item in review_items if isinstance(item, dict)] if isinstance(review_items, list) else []
        merged_items = _merge_review_items(dataframe, current_items)
        if not merged_items:
            merged_items = current_items
        for item in merged_items:
            display_label = _display_final_label(item) or _label_to_display(
                item.get("llm_label") or item.get("model_label")
            )
            item["approved"] = display_label in {"Pass", "Fail"}
        updated_summary = {**summary_payload, "review_items": merged_items}
        return (
            _agentic_review_table_update(merged_items),
            updated_summary,
            json.dumps(updated_summary, indent=2, ensure_ascii=False),
        )

    def investigate_review_cases_ui(
        dataframe: pd.DataFrame,
        summary_payload: dict[str, Any],
    ) -> tuple[str, pd.DataFrame, dict[str, Any], str]:
        summary_payload = summary_payload if isinstance(summary_payload, dict) else {}
        review_items = summary_payload.get("review_items", [])
        current_items = [dict(item) for item in review_items if isinstance(item, dict)] if isinstance(review_items, list) else []
        merged_items = _merge_review_items(dataframe, current_items)
        if not merged_items:
            merged_items = current_items
        report = investigate_label_changed_cases_with_ollama(merged_items, config=config)
        generated_items = report.get("generated_training_items", []) if isinstance(report, dict) else []
        generated_items = [dict(item) for item in generated_items if isinstance(item, dict)]
        queue_source_items = merged_items or current_items
        updated_review_items = _refresh_investigator_generated_queue(queue_source_items, generated_items)
        updated_summary = {
            **summary_payload,
            "diagnosis": report,
            "review_items": updated_review_items,
        }
        return (
            _format_review_case_investigation_report(report),
            _agentic_review_table_update(updated_review_items),
            updated_summary,
            json.dumps(updated_summary, indent=2, ensure_ascii=False),
        )

    def investigate_regressions_ui(summary_payload: dict[str, Any]) -> tuple[str, dict[str, Any], str]:
        summary_payload = summary_payload if isinstance(summary_payload, dict) else {}
        comparisons = [
            item
            for item in summary_payload.get("comparisons", [])
            if isinstance(item, dict) and str(item.get("outcome", "")).strip() == "regressed"
        ]
        if comparisons:
            diagnosis = investigate_score_regressions_with_ollama(comparisons, config=config)
            summary_payload = {
                **summary_payload,
                "score_regression_diagnosis": diagnosis,
            }
        return (
            _format_regression_investigation_report(summary_payload),
            summary_payload,
            json.dumps(summary_payload, indent=2, ensure_ascii=False),
        )

    def choose_agentic_file_upload(upload_value: Any, current_source: str) -> tuple[Any, str]:
        if _extract_uploaded_paths(upload_value):
            return gr.update(value=None), "file"
        return gr.update(), current_source

    def choose_agentic_folder_upload(upload_value: Any, current_source: str) -> tuple[Any, str]:
        if _extract_uploaded_paths(upload_value):
            return gr.update(value=None), "folder"
        return gr.update(), current_source

    with gr.Blocks(title="UOB AI Labs") as app:
        gr.HTML(_hero_html())

        payload_state = gr.State({})
        review_items_state = gr.State([])
        review_candidate_items_state = gr.State([])
        chat_history_state = gr.State([])
        agentic_summary_state = gr.State({})
        agentic_empty_transcript_state = gr.State("")
        agentic_pending_question_state = gr.State("")
        agentic_upload_source_state = gr.State("")

        with gr.Tabs():
            with gr.Tab("Transcript Review"):
                with gr.Row(equal_height=True, elem_id="transcript-source-row"):
                    with gr.Column(scale=1, elem_classes=["transcript-source-card"]):
                        gr.HTML(
                            "<div class='source-kicker'>Option 1</div>"
                            "<div class='source-title'>Load a demo sample</div>"
                            "<div class='source-copy'>Start from a curated synthetic transcript for a quick walkthrough.</div>"
                        )
                        sample_dropdown = gr.Dropdown(
                            choices=sorted(sample_lookup.keys()),
                            label="Sample transcript",
                            value=next(iter(sample_lookup.keys()), None),
                        )
                        load_btn = gr.Button("Load selected sample")
                    with gr.Column(scale=1, elem_classes=["transcript-source-card"]):
                        gr.HTML(
                            "<div class='source-kicker'>Option 2</div>"
                            "<div class='source-title'>Upload your own script</div>"
                            "<div class='source-copy'>Use a local plain-text transcript and run it through the same review workflow.</div>"
                        )
                        transcript_upload = gr.UploadButton(
                            "Choose .txt transcript",
                            file_count="single",
                            file_types=[".txt"],
                            size="lg",
                            elem_id="upload-transcript-btn",
                        )
                        gr.HTML(
                            "<div class='source-note'>"
                            f"Tip: bundled scripts live in <code>{DEMO_TRANSCRIPT_SCRIPTS_DIR}</code>."
                            "</div>"
                        )

                transcript_box = gr.Textbox(label="Transcript", lines=10, placeholder="Paste a transcript here...")
                load_status = gr.Markdown()
                inference_btn = gr.Button("Run Inference", variant="primary")

                results_markdown = gr.Markdown()
                borderline_summary = gr.Markdown()
                borderline_table = gr.Dataframe(
                    headers=[
                        "add",
                        "rule",
                        "score",
                        "text",
                        "model_label",
                        "qwen_label",
                        "final_label",
                    ],
                    datatype=[
                        "bool",
                        "str",
                        "str",
                        "str",
                        "str",
                        "str",
                        "str",
                    ],
                    row_count=(0, "dynamic"),
                    column_count=(7, "fixed"),
                    interactive=True,
                    label="Human Review Queue",
                    elem_id="borderline-review-table",
                    wrap=True,
                    min_width=0,
                    column_widths=["84px", "96px", "136px", "420px", "172px", "172px", "172px"],
                )

                with gr.Row():
                    label_btn = gr.Button("Label Review Candidates with Qwen3-4B")
                    approve_btn = gr.Button("Save Selected Examples")
                    retrain_btn = gr.Button(
                        "Retrain Models",
                        interactive=approved_summary["ready_to_retrain"],
                        variant="secondary",
                    )

                approval_status = gr.Markdown()
                llm_review_results = gr.Markdown(DEFAULT_LLM_PANEL_MESSAGE, elem_id="llm-review-results")
                retrain_output = gr.Code(label="Retrain Output", language="json")

                with gr.Group(elem_id="llm-review-shell"):
                    with gr.Accordion("LLM Review Panel", open=False, elem_id="llm-review-panel"):
                        llm_review_summary = gr.Markdown(DEFAULT_LLM_PANEL_MESSAGE)
                        gr.Markdown("**Quick review prompts**")
                        with gr.Row(elem_id="llm-prompt-buttons"):
                            prompt_buttons = [gr.Button(prompt, size="sm") for prompt in QUICK_REVIEW_PROMPTS]
                        chatbot = gr.Chatbot(
                            label="Review Assistant",
                            elem_id="llm-review-chatbot",
                            height=320,
                            max_height=320,
                        )
                        chat_input = gr.Textbox(label="Ask a follow-up question")

                load_btn.click(load_sample, inputs=[sample_dropdown], outputs=[transcript_box, load_status])
                transcript_upload.upload(load_uploaded_script, inputs=[transcript_upload], outputs=[transcript_box, load_status])
                inference_btn.click(
                    run_inference_ui,
                    inputs=[transcript_box],
                    outputs=[
                        results_markdown,
                        borderline_table,
                        payload_state,
                        borderline_summary,
                        review_items_state,
                        review_candidate_items_state,
                        llm_review_results,
                        llm_review_summary,
                        approval_status,
                        chatbot,
                        chat_history_state,
                    ],
                )
                label_btn.click(
                    label_borderline_ui,
                    inputs=[review_candidate_items_state],
                    outputs=[
                        borderline_table,
                        review_items_state,
                        review_candidate_items_state,
                        llm_review_results,
                        llm_review_summary,
                        approval_status,
                    ],
                )
                approve_btn.click(
                    approve_ui,
                    inputs=[borderline_table, review_items_state],
                    outputs=[approval_status, retrain_btn],
                )
                retrain_btn.click(retrain_ui, outputs=[retrain_output])
                chat_input.submit(
                    chat_ui,
                    inputs=[chat_input, payload_state, review_items_state, chat_history_state],
                    outputs=[chatbot, chat_history_state, chat_input],
                )
                for button, prompt in zip(prompt_buttons, QUICK_REVIEW_PROMPTS):
                    button.click(
                        fn=lambda payload, review_items, history, prompt=prompt, config=config: _preset_prompt(prompt, payload, review_items, history, config=config),
                        inputs=[payload_state, review_items_state, chat_history_state],
                        outputs=[chatbot, chat_history_state, chat_input],
                    )

            with gr.Tab("Agentic Loop"):
                gr.Markdown(
                    "**Upload transcripts &middot; Agents review &middot; Human approve &middot; "
                    "Re-train and re-inference.**",
                    elem_id="agentic-tab-intro",
                )
                with gr.Row(equal_height=True, elem_classes=["agentic-upload-row"]):
                    with gr.Column(scale=1, elem_classes=["agentic-upload-card"]):
                        gr.HTML(
                            "<div class='source-kicker'>Option 1</div>"
                            "<div class='source-title'>Upload file(s)</div>"
                            "<div class='source-copy'>Choose one <code>.txt</code>/<code>.json</code> file or several <code>.txt</code> transcripts.</div>"
                        )
                        incoming_upload = gr.File(
                            label="Transcript file(s)",
                            file_count="multiple",
                            file_types=[".txt", ".json"],
                            height=110,
                        )
                    with gr.Column(scale=1, elem_classes=["agentic-upload-card"]):
                        gr.HTML(
                            "<div class='source-kicker'>Option 2</div>"
                            "<div class='source-title'>Upload folder</div>"
                            "<div class='source-copy'>Choose a local folder. The loop will use every <code>.txt</code> transcript inside it.</div>"
                        )
                        incoming_folder_upload = gr.File(
                            label="Folder of .txt transcripts",
                            file_count="directory",
                            file_types=[".txt"],
                            height=110,
                        )
                with gr.Row():
                    agentic_reset_base_btn = gr.Button(
                        "Reset to Base Models",
                        variant="secondary",
                        elem_id="agentic-reset-btn",
                    )
                    agentic_run_btn = gr.Button(
                        "Start Agentic Review",
                        variant="primary",
                        elem_id="agentic-run-btn",
                    )
                agentic_progress = gr.HTML(_format_agentic_progress(), elem_id="agentic-progress")
                agentic_summary_markdown = gr.Markdown("## Agentic Loop\n- No run output available yet.", elem_id="agentic-summary-markdown")
                agentic_review_table = gr.Dataframe(
                    headers=[
                        "add",
                        "rule",
                        "score",
                        "anchor",
                        "text",
                        "model",
                        "qwen",
                        "human",
                    ],
                    datatype=[
                        "bool",
                        "str",
                        "str",
                        "str",
                        "str",
                        "str",
                        "str",
                        "str",
                    ],
                    row_count=0,
                    column_count=8,
                    interactive=True,
                    label="Human Approval Queue",
                    elem_id="agentic-review-table",
                    wrap=True,
                    min_width=0,
                    column_widths=["54px", "64px", "82px", "27%", "34%", "72px", "72px", "72px"],
                    max_height=420,
                )
                agentic_comparison_markdown = gr.Markdown(
                    _format_agentic_comparison_markdown([]),
                    elem_id="agentic-comparison-markdown",
                )
                with gr.Row():
                    agentic_select_all_btn = gr.Button(
                        "Select All for Retraining",
                        variant="secondary",
                        elem_id="agentic-select-all-btn",
                    )
                    agentic_investigate_review_btn = gr.Button("Investigate Failed Cases", variant="secondary", elem_id="agentic-investigate-review-btn")
                    agentic_investigate_btn = gr.Button("Investigate Score Regressions", variant="secondary", elem_id="agentic-investigate-regression-btn")
                with gr.Row():
                    agentic_continue_btn = gr.Button(
                        "Approve, Retrain, Compare",
                        interactive=False,
                        variant="secondary",
                        elem_id="agentic-continue-btn",
                    )
                agentic_investigation_markdown = gr.Markdown(
                    "### InvestigatorAgent Diagnosis\n"
                    "Click **Investigate Failed Cases** to inspect synthetic coverage "
                    "for model/human disagreements before retraining.",
                    elem_id="agentic-investigation-markdown",
                )
                with gr.Accordion("Raw Agentic Output", open=False):
                    agentic_raw_output = gr.Code(label="Loop Output", language="json")

                with gr.Group(elem_id="agentic-analysis-shell"):
                    with gr.Accordion("Agentic Analysis Assistant", open=False, elem_id="agentic-analysis-panel"):
                        agentic_chat_summary = gr.Markdown(DEFAULT_AGENTIC_CHAT_MESSAGE)
                        gr.Markdown("**Quick prompts**")
                        with gr.Row(elem_id="agentic-prompt-buttons"):
                            agentic_prompt_buttons = [gr.Button(prompt, size="sm") for prompt in AGENTIC_QUICK_PROMPTS]
                        agentic_chatbot = gr.Chatbot(
                            label="Agentic Analysis Assistant",
                            elem_id="agentic-analysis-chatbot",
                            height=340,
                            max_height=340,
                        )
                        agentic_chat_input = gr.Textbox(label="Ask about the agentic run")

                incoming_upload.change(
                    choose_agentic_file_upload,
                    inputs=[incoming_upload, agentic_upload_source_state],
                    outputs=[incoming_folder_upload, agentic_upload_source_state],
                    queue=False,
                )
                incoming_folder_upload.change(
                    choose_agentic_folder_upload,
                    inputs=[incoming_folder_upload, agentic_upload_source_state],
                    outputs=[incoming_upload, agentic_upload_source_state],
                    queue=False,
                )

                agentic_run_btn.click(
                    start_agentic_review_ui,
                    inputs=[
                        agentic_empty_transcript_state,
                        incoming_upload,
                        incoming_folder_upload,
                        agentic_upload_source_state,
                    ],
                    outputs=[
                        agentic_progress,
                        agentic_summary_markdown,
                        agentic_review_table,
                        agentic_comparison_markdown,
                        agentic_raw_output,
                        agentic_summary_state,
                        agentic_chatbot,
                        agentic_chat_input,
                        agentic_continue_btn,
                    ],
                )

                agentic_reset_base_btn.click(
                    reset_base_models_ui,
                    outputs=[
                        agentic_progress,
                        agentic_summary_markdown,
                        agentic_review_table,
                        agentic_comparison_markdown,
                        agentic_raw_output,
                        agentic_summary_state,
                        agentic_chatbot,
                        agentic_chat_input,
                        agentic_continue_btn,
                    ],
                )

                agentic_continue_btn.click(
                    continue_agentic_ui,
                    inputs=[agentic_review_table, agentic_summary_state],
                    outputs=[
                        agentic_progress,
                        agentic_summary_markdown,
                        agentic_review_table,
                        agentic_comparison_markdown,
                        agentic_raw_output,
                        agentic_summary_state,
                        agentic_chatbot,
                        agentic_chat_input,
                        agentic_investigation_markdown,
                    ],
                )

                agentic_select_all_btn.click(
                    select_all_agentic_review_rows_ui,
                    inputs=[agentic_review_table, agentic_summary_state],
                    outputs=[
                        agentic_review_table,
                        agentic_summary_state,
                        agentic_raw_output,
                    ],
                )
                agentic_investigate_review_btn.click(
                    investigate_review_cases_ui,
                    inputs=[agentic_review_table, agentic_summary_state],
                    outputs=[
                        agentic_investigation_markdown,
                        agentic_review_table,
                        agentic_summary_state,
                        agentic_raw_output,
                    ],
                )
                agentic_investigate_btn.click(
                    investigate_regressions_ui,
                    inputs=[agentic_summary_state],
                    outputs=[
                        agentic_investigation_markdown,
                        agentic_summary_state,
                        agentic_raw_output,
                    ],
                )

                agentic_chat_input.submit(
                    append_agentic_user_message,
                    inputs=[agentic_chat_input, agentic_chatbot],
                    outputs=[agentic_chatbot, agentic_chat_input, agentic_pending_question_state],
                    queue=False,
                ).then(
                    answer_agentic_chat_message,
                    inputs=[agentic_pending_question_state, agentic_summary_state, agentic_chatbot],
                    outputs=[agentic_chatbot],
                )
                for button, prompt in zip(agentic_prompt_buttons, AGENTIC_QUICK_PROMPTS):
                    button.click(
                        fn=lambda history, prompt=prompt: append_agentic_prompt_message(history, prompt),
                        inputs=[agentic_chatbot],
                        outputs=[agentic_chatbot, agentic_chat_input, agentic_pending_question_state],
                        queue=False,
                    ).then(
                        answer_agentic_chat_message,
                        inputs=[agentic_pending_question_state, agentic_summary_state, agentic_chatbot],
                        outputs=[agentic_chatbot],
                    )

    return app.queue(default_concurrency_limit=1)


def launch_demo_app(
    *,
    config_path: str | None = None,
    server_name: str = "127.0.0.1",
    server_port: int = 7860,
    share: bool = False,
) -> None:
    _extend_gradio_server_start_timeout()
    app = build_demo_app(config_path)
    app.launch(server_name=server_name, server_port=server_port, share=share, css=APP_CSS)
