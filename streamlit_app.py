"""Streamlit chat UI - thin layer over the ChatSession backend in chatbot.py.

Run with:
    streamlit run streamlit_app.py
"""
from __future__ import annotations

from datetime import datetime

import streamlit as st

from chatbot import ChatSession, TurnResult, humanize
from expert_system import ALL_SYMPTOMS, MedicalExpertSystem


WELCOME = (
    "Hello! I'm a medical diagnosis assistant. "
    "Describe your symptoms in plain English and I'll try to help. "
    "If I'm not sure I'll ask follow-up questions - your earlier symptoms are always remembered."
)


# ---------------------------------------------------------------------------
# Session-state helpers
# ---------------------------------------------------------------------------
def _ensure_state():
    if "session" not in st.session_state:
        st.session_state.session = ChatSession()
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": WELCOME}]
    if "report_md" not in st.session_state:
        st.session_state.report_md = None


def _start_new_chat():
    st.session_state.session = ChatSession()
    st.session_state.messages = [{"role": "assistant", "content": WELCOME}]
    st.session_state.report_md = None


def _generate_report():
    st.session_state.report_md = st.session_state.session.generate_report()


# ---------------------------------------------------------------------------
# Reply formatting (turns ChatSession.TurnResult into chat markdown)
# ---------------------------------------------------------------------------
def _format_diagnosis_block(result: TurnResult):
    top_diag, _ = result.results[0]
    parts = ["**Possible matches:**"]
    for diag, pct in result.results:
        parts.append(f"- {diag['disease']} - **{pct:.0f}%** certainty")
    parts.append("")
    parts.append(f"**Most likely diagnosis:** {top_diag['disease']}")
    precautions = top_diag.get("precautions", ())
    if precautions:
        parts.append("")
        parts.append("**Recommended precautions:**")
        for p in precautions:
            parts.append(f"- {p}")
    else:
        parts.append("")
        parts.append("_(No precautions recorded for this disease.)_")
    return "\n".join(parts)


def _format_followup_block(result: TurnResult):
    guesses = ", ".join(f"{d['disease']} ({pct:.0f}%)" for d, pct in result.results)
    parts = [f"I'm not certain yet. Current top guesses: {guesses}.", ""]
    if result.followups:
        parts.append("Do you also experience any of these? (you can answer with just yes/no or list the ones you have)")
        for s in result.followups:
            parts.append(f"- {humanize(s)}")
    else:
        parts.append("Could you describe any other symptoms?")
    return "\n".join(parts)


def _reply_for(result: TurnResult):
    note = ""
    if result.new_symptoms:
        pretty = ", ".join(humanize(s) for s in sorted(result.new_symptoms))
        note = f"_Noted: {pretty}._\n\n"
    elif result.status not in {"no_symptoms"}:
        note = "_(No new symptoms picked up from that turn.)_\n\n"

    if result.status == "no_symptoms":
        return "I couldn't identify any symptoms in that. Could you rephrase?"
    if result.status == "no_match":
        return note + "None of my known diseases match those symptoms. Try describing them differently."
    if result.status == "need_followup":
        return note + _format_followup_block(result)
    if result.status == "diagnosed":
        return note + _format_diagnosis_block(result)
    return "Sorry, something unexpected happened."


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Medical Diagnosis Expert System", page_icon=":hospital:")
st.title(":hospital: Medical Diagnosis Expert System")
st.caption("Powered by Experta + NLTK. Describe your symptoms - the bot will ask follow-ups if it's not sure.")

_ensure_state()
session: ChatSession = st.session_state.session

# ---------------- Sidebar ----------------
with st.sidebar:
    st.subheader("Knowledge base")
    st.write(f"- {len(MedicalExpertSystem.profiles)} diseases")
    st.write(f"- {len(ALL_SYMPTOMS)} symptoms")

    st.subheader("Current case")
    if session.last_symptoms:
        for s in sorted(session.last_symptoms):
            st.write(f"- {humanize(s)}")
    else:
        st.write("_no symptoms yet_")

    if session.combined_text:
        st.subheader("Transcript")
        st.text_area(
            "What you've said so far",
            value=session.combined_text,
            height=120,
            disabled=True,
            label_visibility="collapsed",
        )

    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        st.button("New chat", use_container_width=True, on_click=_start_new_chat)
    with col_b:
        st.button("Generate report", use_container_width=True, on_click=_generate_report)

    if st.session_state.report_md:
        st.download_button(
            "Download report (.md)",
            data=st.session_state.report_md,
            file_name=f"diagnosis_report_{datetime.now():%Y%m%d_%H%M}.md",
            mime="text/markdown",
            use_container_width=True,
        )
        with st.expander("Preview report"):
            st.markdown(st.session_state.report_md)

# ---------------- Chat ----------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_text := st.chat_input("Describe your symptoms..."):
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    result = session.step(user_text)
    reply = _reply_for(result)
    session.record_assistant(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.markdown(reply)
