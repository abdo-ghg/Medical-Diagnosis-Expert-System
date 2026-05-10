"""Medical diagnosis expert system using the Experta rule engine.

Knowledge representation:
  - DiseaseProfile facts encode each disease's symptom set, per-symptom weights
    (frequency in that disease's training rows), and precautions.
  - UserSymptoms is a single working-memory fact carrying what the user reported.
  - Diagnosis facts are *inferred* by a rule that pairs each DiseaseProfile with
    UserSymptoms, computes a confidence score, and asserts the result.

Inference:
  - The `match_disease` rule fires once per (DiseaseProfile, UserSymptoms) pair.
  - Score = weighted_recall * precision (both in [0,1]).
  - The caller normalizes top-K diagnosis scores into percentages (the format
    requested by the project PDF).
"""
from __future__ import annotations

# --- Compatibility shim: experta's bundled frozendict uses collections.Mapping
# which was removed in Python 3.10+. We restore the alias before importing.
import collections
import collections.abc

if not hasattr(collections, "Mapping"):
    collections.Mapping = collections.abc.Mapping
    collections.MutableMapping = collections.abc.MutableMapping
# ----------------------------------------------------------------------------

from collections import defaultdict
from pathlib import Path

import pandas as pd
from experta import DefFacts, Fact, Field, KnowledgeEngine, MATCH, Rule

from nlp_processor import SymptomExtractor

DATASET_PATH = Path(__file__).parent / "Medical Diagnosis Expert System.csv"


# ---------------------------------------------------------------------------
# Fact classes - the "knowledge representation" layer.
# ---------------------------------------------------------------------------
class DiseaseProfile(Fact):
    """Static knowledge: a disease and its symptom profile."""
    name = Field(str, mandatory=True)
    symptoms = Field(object, mandatory=True)      # frozenset of symptom names
    weights = Field(object, mandatory=True)       # tuple of (symptom, weight) pairs
    precautions = Field(object, default=())       # tuple of strings


class UserSymptoms(Fact):
    """Working memory: symptoms the user has reported so far."""
    names = Field(object, mandatory=True)         # frozenset of symptom names


class Diagnosis(Fact):
    """Inferred: a disease whose profile overlaps with the user's symptoms."""
    disease = Field(str, mandatory=True)
    score = Field(float, mandatory=True)
    matched = Field(object, mandatory=True)
    missing = Field(object, mandatory=True)
    precautions = Field(object, default=())


# ---------------------------------------------------------------------------
# Dataset -> DiseaseProfile facts.
# ---------------------------------------------------------------------------
def _load_profiles(csv_path: Path | str = DATASET_PATH):
    df = pd.read_csv(csv_path)
    df["disease"] = df["disease"].astype(str).str.strip()

    syms_of: dict[str, set[str]] = defaultdict(set)
    freq_of: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    row_count: dict[str, int] = defaultdict(int)
    prec_of: dict[str, tuple[str, ...]] = {}

    for _, row in df.iterrows():
        disease = row["disease"]
        row_count[disease] += 1

        if pd.notna(row["symptoms"]):
            for sym in str(row["symptoms"]).split(","):
                sym = sym.strip()
                if sym:
                    syms_of[disease].add(sym)
                    freq_of[disease][sym] += 1

        if disease not in prec_of and pd.notna(row["precautions"]):
            items = [p.strip() for p in str(row["precautions"]).split(",") if p.strip()]
            if items:
                prec_of[disease] = tuple(items)

    profiles = []
    for d, sset in syms_of.items():
        weights = tuple((s, freq_of[d][s] / row_count[d]) for s in sset)
        profiles.append({
            "name": d,
            "symptoms": frozenset(sset),
            "weights": weights,
            "precautions": prec_of.get(d, ()),
        })
    return profiles


def _canonicalize_profiles(raw_profiles: list[dict], canonical_of: dict[str, str]) -> list[dict]:
    """Rewrite profiles so synonym-collision symptoms collapse to one canonical name.

    When two raw symptoms map to the same canonical (e.g. `belly_pain` and
    `stomach_pain` because of the SYNONYMS map), we keep the higher of their
    weights for that disease. Without this step the user's "stomach pain"
    matches both raw symptoms and inflates the symptom count, which hurts
    the precision term in the score.
    """
    out: list[dict] = []
    for p in raw_profiles:
        new_syms: set[str] = set()
        new_weights: dict[str, float] = {}
        for sym, w in p["weights"]:
            c = canonical_of.get(sym, sym)
            new_syms.add(c)
            new_weights[c] = max(new_weights.get(c, 0.0), float(w))
        out.append({
            "name": p["name"],
            "symptoms": frozenset(new_syms),
            "weights": tuple(new_weights.items()),
            "precautions": p["precautions"],
        })
    return out


def _bootstrap():
    """One-shot: load CSV, compute canonical map, deduplicate profiles, build extractor."""
    raw_profiles = _load_profiles()
    raw_all_symptoms: set[str] = set().union(*(p["symptoms"] for p in raw_profiles))

    # Temporary extractor purely to compute the canonical map; it sees every
    # raw symptom name and reports which ones collapse to the same stems.
    temp_extractor = SymptomExtractor(raw_all_symptoms)
    canonical_of = dict(temp_extractor.canonical_of)

    profiles = _canonicalize_profiles(raw_profiles, canonical_of)
    all_canonical: set[str] = set().union(*(p["symptoms"] for p in profiles))

    # The shared extractor used by the chatbot / Streamlit — built only over
    # canonical symptoms so extract() never returns duplicates.
    extractor = SymptomExtractor(all_canonical)
    return profiles, all_canonical, extractor


_PROFILES, ALL_SYMPTOMS, EXTRACTOR = _bootstrap()


# ---------------------------------------------------------------------------
# The Experta engine.
# ---------------------------------------------------------------------------
class MedicalExpertSystem(KnowledgeEngine):
    profiles: list[dict] = _PROFILES  # class-level so reset() doesn't reload

    @DefFacts()
    def _initial(self):
        for p in self.profiles:
            yield DiseaseProfile(**p)

    @Rule(
        DiseaseProfile(
            name=MATCH.disease,
            symptoms=MATCH.dsyms,
            weights=MATCH.w,
            precautions=MATCH.prec,
        ),
        UserSymptoms(names=MATCH.user_syms),
    )
    def match_disease(self, disease, dsyms, w, prec, user_syms):
        matched = user_syms & dsyms
        if not matched:
            return
        weights = dict(w)
        w_matched = sum(weights[s] for s in matched)
        w_total = sum(weights[s] for s in dsyms)
        recall = (w_matched / w_total) if w_total else 0.0
        precision = len(matched) / len(user_syms) if user_syms else 0.0
        score = recall * precision
        if score > 0:
            self.declare(Diagnosis(
                disease=disease,
                score=float(score),
                matched=frozenset(matched),
                missing=frozenset(dsyms - matched),
                precautions=prec,
            ))


# ---------------------------------------------------------------------------
# High-level convenience API used by the chatbot / demo.
# (ALL_SYMPTOMS and EXTRACTOR are populated by _bootstrap() above.)
# ---------------------------------------------------------------------------
# Threshold tuning: when to ask follow-ups vs. commit to a diagnosis.
MIN_TOP_SCORE = 0.60
AMBIGUITY_GAP = 0.10
TOP_K = 3


def diagnose(user_symptoms: set[str], top_k: int = TOP_K):
    """Run the engine on the given symptoms and return [(Diagnosis, percentage), ...]."""
    es = MedicalExpertSystem()
    es.reset()
    if not user_symptoms:
        return []
    es.declare(UserSymptoms(names=frozenset(user_symptoms)))
    es.run()

    results = sorted(
        (f for f in es.facts.values() if isinstance(f, Diagnosis)),
        key=lambda d: d["score"],
        reverse=True,
    )[:top_k]

    total = sum(d["score"] for d in results)
    if total <= 0:
        return [(d, 0.0) for d in results]
    return [(d, d["score"] / total * 100.0) for d in results]


def is_confident(results: list[tuple[Diagnosis, float]]) -> bool:
    if not results:
        return False
    if results[0][0]["score"] < MIN_TOP_SCORE:
        return False
    if len(results) >= 2 and (results[0][0]["score"] - results[1][0]["score"]) < AMBIGUITY_GAP:
        return False
    return True


def followup_symptoms(results: list[tuple[Diagnosis, float]], n: int = 5) -> list[str]:
    """Discriminating symptoms among the leading candidates that are still missing."""
    if not results:
        return []
    candidates: dict[str, float] = {}
    weight_lookup = {p["name"]: dict(p["weights"]) for p in MedicalExpertSystem.profiles}
    for diag, _pct in results:
        d_weights = weight_lookup.get(diag["disease"], {})
        for s in diag["missing"]:
            candidates[s] = candidates.get(s, 0.0) + d_weights.get(s, 0.0) * diag["score"]
    ordered = sorted(candidates.items(), key=lambda kv: kv[1], reverse=True)
    return [s for s, _ in ordered[:n]]


