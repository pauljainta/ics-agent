#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ics_agent.knowledge import load_manual_bundle
from ics_agent.orchestrator import ExperimentOrchestrator
from ics_agent.scenarios import ScenarioCatalog
from ics_agent.testbeds.registry import observation_focus_fields, trend_fields


DOCUMENTS_ROOT = REPO_ROOT.parent / "Documents"
CACHE_ROOT = REPO_ROOT / ".cache" / "phase35"
EXTRACTED_ROOT = CACHE_ROOT / "extracted"
DRAFT_ROOT = CACHE_ROOT / "drafts"
ASSEMBLED_ROOT = CACHE_ROOT / "frozen_contexts"

SOURCE_PDFS: dict[str, list[Path]] = {
    "water_treatment": [
        DOCUMENTS_ROOT / "Water Treatment Manual" / "SWaT Operation Manual_Final.pdf",
        DOCUMENTS_ROOT / "Water Treatment Manual" / "SWAT.pdf",
        DOCUMENTS_ROOT / "Water Treatment Manual" / "SWAT_Dataset.pdf",
        DOCUMENTS_ROOT / "Water Treatment Manual" / "SWAT_List_of_attacks_Final.pdf",
    ],
    "water_distribution": [
        DOCUMENTS_ROOT / "Water Distribution Manuals" / "3055366.3055375.pdf",
        DOCUMENTS_ROOT
        / "Water Distribution Manuals"
        / "Detection_of_Cyber_Attacks_on_a_Water_Distribution_Testbed.pdf",
        DOCUMENTS_ROOT / "Water Distribution Manuals" / "1906.02279v1.pdf",
    ],
}

COMMITTED_PRIMERS: dict[str, Path] = {
    "water_treatment": REPO_ROOT
    / "testbeds"
    / "water_treatment"
    / "knowledge"
    / "phase35"
    / "water_treatment_domain_primer.md",
    "water_distribution": REPO_ROOT
    / "testbeds"
    / "water_distribution"
    / "knowledge"
    / "phase35"
    / "water_distribution_domain_primer.md",
}

PROFILE_IDS: dict[str, str] = {
    "water_treatment": "black_box_high_cyber_high_domain_treatment_attacker",
    "water_distribution": "black_box_high_cyber_high_domain_distribution_attacker",
}

PROCESS_KEYWORDS: dict[str, tuple[str, ...]] = {
    "water_treatment": (
        "water",
        "tank",
        "pump",
        "valve",
        "flow",
        "pressure",
        "chemical",
        "dosing",
        "membrane",
        "filtration",
        "alarm",
        "interlock",
        "manual",
        "control",
        "level",
    ),
    "water_distribution": (
        "water",
        "tank",
        "pump",
        "valve",
        "flow",
        "pressure",
        "consumer",
        "storage",
        "supply",
        "return",
        "distribution",
        "alarm",
        "level",
        "quality",
    ),
}

EXCLUDED_MARKERS = (
    "attack ",
    "attacks ",
    "dataset",
    "scenario",
    "figure",
    "table",
    "doi",
    "copyright",
    "authorized licensed use",
    "ieee",
    "conference",
)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _iter_testbeds(target: str) -> list[str]:
    if target == "all":
        return list(SOURCE_PDFS)
    if target not in SOURCE_PDFS:
        raise SystemExit(f"unsupported testbed `{target}`")
    return [target]


def _extract_target(testbed: str, pdf_path: Path) -> Path:
    return EXTRACTED_ROOT / testbed / f"{pdf_path.stem}.txt"


def _draft_target(testbed: str) -> Path:
    return DRAFT_ROOT / testbed / f"{testbed}_domain_primer_draft.md"


def _assembled_target(profile_id: str) -> Path:
    return ASSEMBLED_ROOT / f"{profile_id}.json"


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _extract_with_pdftotext(pdf_path: Path, output_path: Path) -> None:
    subprocess.run(
        ["pdftotext", "-layout", "-nopgbrk", str(pdf_path), str(output_path)],
        check=True,
    )


def _extract_with_python(pdf_path: Path, output_path: Path) -> None:
    last_error: Exception | None = None
    for module_name in ("pypdf", "PyPDF2"):
        try:
            module = __import__(module_name)
            reader = module.PdfReader(str(pdf_path))
            text = "\n".join((page.extract_text() or "") for page in reader.pages)
            output_path.write_text(text)
            return
        except Exception as exc:  # pragma: no cover - depends on local fallback packages
            last_error = exc
    raise RuntimeError(
        f"unable to extract `{pdf_path}` with pdftotext fallback"
    ) from last_error


def extract_sources(testbed: str) -> list[Path]:
    created: list[Path] = []
    has_pdftotext = shutil.which("pdftotext") is not None
    for pdf_path in SOURCE_PDFS[testbed]:
        if not pdf_path.exists():
            raise FileNotFoundError(f"missing source pdf: {pdf_path}")
        output_path = _extract_target(testbed, pdf_path)
        _ensure_parent(output_path)
        if has_pdftotext:
            _extract_with_pdftotext(pdf_path, output_path)
        else:  # pragma: no cover - pdftotext is present in this environment
            _extract_with_python(pdf_path, output_path)
        created.append(output_path)
    return created


def _sanitize_sentence(text: str, *, testbed: str) -> str:
    candidate = " ".join(text.split())
    if len(candidate) < 40:
        return ""
    lowered = candidate.lower()
    if not any(keyword in lowered for keyword in PROCESS_KEYWORDS[testbed]):
        return ""
    if any(marker in lowered for marker in EXCLUDED_MARKERS):
        return ""
    candidate = re.sub(r"https?://\S+", "", candidate)
    candidate = re.sub(r"\[[0-9,\- ]+\]", "", candidate)
    candidate = re.sub(r"\b(?:swat|wadi|sutd|itrust|rovisys|ieee|cps)\b", "", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"\b(?:WT|WD)-[A-Za-z0-9-]+\b", "process stage", candidate)
    candidate = re.sub(r"\bP\d[A-Z]?\b", "process stage", candidate)
    candidate = re.sub(
        r"\b\d+[A-Z]?(?:-[A-Z]{1,6}\d{3,4}(?:-[A-Z]{1,6}\d{3,4})?)\b",
        "component alias",
        candidate,
    )
    candidate = re.sub(r"\b[A-Za-z]+(?:_[A-Za-z0-9]+){1,}\b", "signal alias", candidate)
    candidate = re.sub(r"\b\d{3,}\b", "threshold value", candidate)
    candidate = re.sub(r"\s+", " ", candidate).strip(" ,.;:-")
    if not candidate:
        return ""
    candidate = candidate[0].upper() + candidate[1:]
    return candidate


def build_draft_primer(testbed: str) -> Path:
    sentences: list[str] = []
    seen: set[str] = set()
    extracted_paths = sorted((EXTRACTED_ROOT / testbed).glob("*.txt"))
    if not extracted_paths:
        extracted_paths = extract_sources(testbed)
    for path in extracted_paths:
        raw_text = path.read_text(errors="ignore")
        chunks = re.split(r"(?<=[.!?])\s+|\n+", raw_text)
        for chunk in chunks:
            sanitized = _sanitize_sentence(chunk, testbed=testbed)
            if not sanitized or sanitized in seen:
                continue
            seen.add(sanitized)
            sentences.append(sanitized)
    output_path = _draft_target(testbed)
    _ensure_parent(output_path)
    lines = [
        f"# {testbed.replace('_', ' ').title()} Phase35 Draft Primer",
        "",
        "Local-only sanitized draft generated from extracted source documents.",
        "This draft is for audit support and is not a committed runtime manual.",
        "",
        "Source PDFs:",
    ]
    lines.extend(f"- {path.name}" for path in SOURCE_PDFS[testbed])
    lines.extend(["", "Sanitized candidate statements:"])
    lines.extend(f"- {sentence}" for sentence in sentences[:80])
    output_path.write_text("\n".join(lines) + "\n")
    return output_path


def _distribution_exact_tokens() -> set[str]:
    contract = yaml.safe_load(
        (REPO_ROOT / "testbeds" / "water_distribution" / "tag_contract.yaml").read_text()
    )
    tags = set((contract.get("tags") or {}).keys())
    tags.update({"WD-UC0", "WD-Supply", "WD-Grid", "WD-Return"})
    return tags


def _treatment_exact_tokens() -> set[str]:
    tokens = set(observation_focus_fields("water_treatment"))
    tokens.update(trend_fields("water_treatment"))
    tokens.update({"WT-P1", "WT-P2", "WT-P3"})
    return tokens


def _scenario_markers(testbed: str) -> set[str]:
    catalog = ScenarioCatalog(
        testbed_id="water_distribution" if testbed == "water_distribution" else None
    )
    selectors = ["attack/*", "normal/*", "readiness/*", "smoke/*", "recheck/*"]
    markers: set[str] = set()
    for selector in selectors:
        try:
            scenarios = catalog.resolve_selectors([selector])
        except Exception:
            continue
        for scenario in scenarios:
            markers.add(str(scenario.catalog_id))
            markers.add(str(scenario.catalog_id).split("/")[-1])
    return markers


def audit_committed_primer(testbed: str) -> list[str]:
    primer_path = COMMITTED_PRIMERS[testbed]
    text = primer_path.read_text()
    violations: list[str] = []
    tokens = (
        _distribution_exact_tokens()
        if testbed == "water_distribution"
        else _treatment_exact_tokens()
    )
    for token in sorted(tokens):
        if token and token in text:
            violations.append(f"exact token leaked: {token}")
    for marker in sorted(_scenario_markers(testbed)):
        if marker and marker in text:
            violations.append(f"scenario marker leaked: {marker}")
    if re.search(r"\b\d{3,}\b", text):
        violations.append("numeric threshold-like token leaked")
    for phrase in (
        "best leverage surface",
        "highest leverage",
        "preferred attack surface",
        "surface ranking",
        "exact tag",
        "exact threshold",
    ):
        if phrase in text.lower():
            violations.append(f"direct prior phrase leaked: {phrase}")
    return violations


def assemble_profile_context(profile_id: str) -> Path:
    orchestrator = ExperimentOrchestrator(
        testbed_id="water_distribution" if "distribution" in profile_id else None
    )
    profile = dict(orchestrator.agent_profiles[profile_id])
    llm = dict(profile.get("llm") or {})
    prompt_path = REPO_ROOT / str(llm.get("prompt_file", ""))
    prompt_text = prompt_path.read_text().strip()
    knowledge_files = list(profile.get("knowledge_files", []))
    manual_text = load_manual_bundle(knowledge_files)
    payload = {
        "profile_id": profile_id,
        "knowledge_mode": profile.get("knowledge_mode"),
        "prompt_file": str(prompt_path.relative_to(REPO_ROOT)),
        "prompt_sha256": _sha256_text(prompt_text),
        "knowledge_files": [
            {
                "path": str(
                    ((REPO_ROOT / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()).relative_to(REPO_ROOT)
                ),
                "sha256": _sha256_text(
                    (
                        (REPO_ROOT / path).resolve()
                        if not Path(path).is_absolute()
                        else Path(path).resolve()
                    ).read_text()
                ),
            }
            for path in knowledge_files
        ],
        "assembled_manual_sha256": _sha256_text(manual_text),
        "assembled_prompt": prompt_text,
        "assembled_manual": manual_text,
    }
    output_path = _assembled_target(profile_id)
    _ensure_parent(output_path)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    return output_path


def run_all(testbed: str) -> list[Path]:
    outputs: list[Path] = []
    outputs.extend(extract_sources(testbed))
    outputs.append(build_draft_primer(testbed))
    violations = audit_committed_primer(testbed)
    if violations:
        raise RuntimeError(
            f"committed primer audit failed for `{testbed}`: {', '.join(violations)}"
        )
    outputs.append(assemble_profile_context(PROFILE_IDS[testbed]))
    return outputs


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline extraction, sanitization, and audit helpers for phase35 knowledge packs."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name in ("extract", "draft", "audit", "all"):
        subparser = subparsers.add_parser(name)
        subparser.add_argument(
            "--testbed",
            default="all",
            choices=["all", "water_treatment", "water_distribution"],
        )

    assemble = subparsers.add_parser("assemble-profile")
    assemble.add_argument("--profile-id", required=True)
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "assemble-profile":
        output_path = assemble_profile_context(args.profile_id)
        print(output_path)
        return 0

    for testbed in _iter_testbeds(args.testbed):
        if args.command == "extract":
            for path in extract_sources(testbed):
                print(path)
        elif args.command == "draft":
            print(build_draft_primer(testbed))
        elif args.command == "audit":
            violations = audit_committed_primer(testbed)
            if violations:
                for violation in violations:
                    print(f"{testbed}: {violation}")
                return 1
            print(f"{testbed}: audit ok")
        elif args.command == "all":
            for path in run_all(testbed):
                print(path)
        else:  # pragma: no cover - argparse keeps this unreachable
            raise RuntimeError(f"unsupported command: {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
