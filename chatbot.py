"""Conversational backend for the medical diagnosis expert system.

`ChatSession` is the single source of truth for the conversation flow and is
shared by both the CLI in this file and the Streamlit UI in
`streamlit_app.py`. Each turn appends to the running transcript so the user
never has to repeat earlier symptoms.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from expert_system import ALL_SYMPTOMS, EXTRACTOR, MedicalExpertSystem, diagnose, followup_symptoms, is_confident
from nlp_processor import SymptomExtractor

MAX_FOLLOWUP_ROUNDS = 3


def humanize(symptom):
    return symptom.replace("__", " ").replace("_", " ").strip()


@dataclass
class TurnResult:
    """Structured outcome of a single conversation turn."""
    status: str               # "no_symptoms" | "no_match" | "need_followup" | "diagnosed"
    new_symptoms: set[str] = field(default_factory=set)
    all_symptoms: set[str] = field(default_factory=set)
    results: list[Any] = field(default_factory=list)   # list[(Diagnosis, percentage)]
    followups: list[str] = field(default_factory=list)


class ChatSession:
    """Stateful conversation handler.

    The session keeps the running transcript (`combined_text`) plus the
    current cumulative symptom set, and on every `step()` re-runs NLP over
    the full transcript and the diagnosis engine over the current symptoms.
    """

    def __init__(self, extractor: SymptomExtractor | None = None,
                 max_followup_rounds: int = MAX_FOLLOWUP_ROUNDS):
        self.extractor = extractor or EXTRACTOR
        self.max_followup_rounds = max_followup_rounds
        self.history: list[tuple[str, str]] = []   # [(role, text)] for the report
        self.combined_text: str = ""
        self.last_symptoms: set[str] = set()
        self.last_results: list[Any] = []
        self.follow_up_rounds: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def reset(self):
        self.history.clear()
        self.combined_text = ""
        self.last_symptoms = set()
        self.last_results = []
        self.follow_up_rounds = 0

    # ------------------------------------------------------------------
    # Core turn handling
    # ------------------------------------------------------------------
    def step(self, user_text):
        """Process one user utterance and return the structured outcome."""
        self.history.append(("user", user_text))
        self.combined_text = f"{self.combined_text} {user_text}".strip()

        symptoms = self.extractor.extract(self.combined_text)
        new_only = symptoms - self.last_symptoms
        self.last_symptoms = symptoms

        if not symptoms:
            return TurnResult(status="no_symptoms")

        results = diagnose(symptoms)
        self.last_results = results
        if not results:
            return TurnResult(status="no_match", new_symptoms=new_only,
                              all_symptoms=symptoms)

        if is_confident(results) or self.follow_up_rounds >= self.max_followup_rounds:
            return TurnResult(
                status="diagnosed",
                new_symptoms=new_only,
                all_symptoms=symptoms,
                results=results,
            )

        self.follow_up_rounds += 1
        return TurnResult(
            status="need_followup",
            new_symptoms=new_only,
            all_symptoms=symptoms,
            results=results,
            followups=followup_symptoms(results, n=5),
        )

    def record_assistant(self, text):
        """Optional: record the assistant's reply in the transcript history."""
        self.history.append(("assistant", text))

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------
    def generate_report(self):
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        lines = [
            "# Medical Diagnosis Report",
            "",
            f"_Generated: {now}_",
            "",
        ]

        if not self.last_symptoms and not self.last_results:
            lines += ["_No symptoms have been recorded yet._", ""]
            return "\n".join(lines)

        lines += ["## Reported symptoms", ""]
        if self.combined_text:
            lines.append("**User description:**")
            lines.append("")
            lines.append(f"> {self.combined_text}")
            lines.append("")
        if self.last_symptoms:
            lines.append("**Identified symptoms:**")
            for s in sorted(self.last_symptoms):
                lines.append(f"- {humanize(s)}")
            lines.append("")

        if self.last_results:
            top_diag, _ = self.last_results[0]
            lines += ["## Diagnosis", ""]
            lines.append(f"**Most likely diagnosis:** {top_diag['disease']}")
            lines.append("")
            lines.append("**Possible matches:**")
            for diag, pct in self.last_results:
                lines.append(f"- {diag['disease']} - {pct:.0f}% certainty")
            lines.append("")

            precautions = top_diag.get("precautions", ())
            if precautions:
                lines += ["## Recommended precautions", ""]
                for p in precautions:
                    lines.append(f"- {p}")
                lines.append("")
            else:
                lines += ["_No precautions recorded for this disease._", ""]

        if self.history:
            lines += ["## Conversation transcript", ""]
            for role, text in self.history:
                speaker = "**You**" if role == "user" else "**Bot**"
                lines.append(f"- {speaker}: {text}")
            lines.append("")

        lines += [
            "---",
            "_This report is informational only and is not a substitute for professional medical advice._",
        ]
        return "\n".join(lines)