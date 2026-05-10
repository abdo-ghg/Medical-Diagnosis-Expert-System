"""NLP layer that turns free-text user input into a set of known symptoms.

Pipeline: lowercase -> strip punctuation -> tokenize -> stopword removal ->
spell-correct against medical vocab -> lemmatize -> stem. We then match by
checking whether all the stemmed parts of a symptom (e.g. "skin_rash" ->
{"skin", "rash"}) appear in the user tokens.
"""
from __future__ import annotations

import re
from difflib import get_close_matches

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

# A few user phrases that don't directly map to any symptom token.
SYNONYMS = {
    "throwup": "vomiting",
    "puke": "vomiting",
    "puking": "vomiting",
    "throwing": "vomiting",
    "tired": "fatigue",
    "exhausted": "fatigue",
    "sneeze": "sneezing",
    "feverish": "fever",
    "tummy": "stomach",
    "belly": "stomach",
    "shaky": "shivering",
    "shake": "shivering",
    "dizzy": "dizziness",
    "tearing": "watering_from_eyes",
    "teary": "watering_from_eyes",
    "achy": "pain",
    "ache": "pain",
}


class SymptomExtractor:
    # Words shorter than this are not spell-corrected (too risky).
    SPELL_MIN_LEN = 4
    # Similarity ratio cutoff for difflib's get_close_matches.
    SPELL_CUTOFF = 0.75

    def __init__(self, all_symptoms):
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words("english"))

        # Map each symptom to its tuple of stemmed parts (stopwords dropped).
        self.symptom_stems: dict[str, tuple[str, ...]] = {}
        for sym in all_symptoms:
            parts = [p for p in sym.replace("__", "_").split("_") if p and p not in self.stop_words]
            self.symptom_stems[sym] = tuple(self._normalize_word(p) for p in parts)

        # Group symptoms that collapse to the same stemmed signature (e.g.
        # `belly_pain` and `stomach_pain` collide because the SYNONYMS map sends
        # "belly" -> "stomach"). For each group we pick one canonical name -
        # preferring symptoms whose parts are NOT keys in SYNONYMS, breaking
        # ties alphabetically. Both the user-side extract and the disease KB
        # then deduplicate to the canonical, removing accidental duplicates.
        sig_to_candidates: dict[tuple[str, ...], list[str]] = {}
        for sym, sig in self.symptom_stems.items():
            if sig:
                sig_to_candidates.setdefault(sig, []).append(sym)

        def _canonical_rank(sym):
            parts = sym.replace("__", "_").split("_")
            penalty = sum(1 for p in parts if p in SYNONYMS)
            return (penalty, sym)

        self.canonical_of: dict[str, str] = {}
        for sig, candidates in sig_to_candidates.items():
            canonical = min(candidates, key=_canonical_rank)
            for sym in candidates:
                self.canonical_of[sym] = canonical

        # Medical vocabulary used for spell correction. We pull every word that
        # appears in a symptom name plus the synonym keys, so typos like
        # "itchin" -> "itching", "skiin rash" -> "skin rash" can be repaired.
        vocab: set[str] = set()
        for sym in all_symptoms:
            for part in sym.replace("__", "_").split("_"):
                if part and part not in self.stop_words:
                    vocab.add(part)
        vocab |= set(SYNONYMS.keys())
        self.medical_vocab: list[str] = sorted(vocab)
        # Stems of those vocab words - used to detect "this token is already a
        # known medical term, no correction needed".
        self.medical_stems: set[str] = {self._normalize_word(w) for w in vocab}

    def _normalize_word(self, word):
        word = word.lower()
        word = SYNONYMS.get(word, word)
        word = self.lemmatizer.lemmatize(word, pos="v")
        word = self.lemmatizer.lemmatize(word, pos="n")
        return self.stemmer.stem(word)

    def _correct_token(self, token):
        """If `token` doesn't match any medical term, try a fuzzy correction."""
        if len(token) < self.SPELL_MIN_LEN:
            return token
        # Already a known medical term (after lemma/stem)? Leave it alone.
        if self._normalize_word(token) in self.medical_stems:
            return token
        matches = get_close_matches(token, self.medical_vocab, n=1, cutoff=self.SPELL_CUTOFF)
        return matches[0] if matches else token

    def _tokenize(self, text):
        text = text.lower()
        text = re.sub(r"[^a-z\s]", " ", text)
        tokens = word_tokenize(text)
        tokens = [t for t in tokens if t and t not in self.stop_words]
        return [self._correct_token(t) for t in tokens]

    def extract(self, user_text):
        """Return the set of known symptom keys (canonicalized) mentioned in `user_text`."""
        tokens = self._tokenize(user_text)
        token_stems = {self._normalize_word(t) for t in tokens}

        found: set[str] = set()
        for symptom, parts in self.symptom_stems.items():
            if parts and all(p in token_stems for p in parts):
                found.add(self.canonical_of.get(symptom, symptom))
        return found

    def to_canonical(self, symptom):
        return self.canonical_of.get(symptom, symptom)