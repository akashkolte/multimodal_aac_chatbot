"""
Streamlit frontend — webcam + chat + live metrics dashboard.

Panels:
  Left sidebar  — persona selector, session controls, live affect display
  Centre        — chat interface with streaming response
  Right sidebar — latency breakdown, bucket priors bar chart

Run: streamlit run ui/app.py
"""

from __future__ import annotations

import requests
import streamlit as st

# ── Config ─────────────────────────────────────────────────────────────────────
API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="AAC Chatbot",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Session state init ─────────────────────────────────────────────────────────
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_latency" not in st.session_state:
    st.session_state.last_latency = {}
if "last_affect" not in st.session_state:
    st.session_state.last_affect = "NEUTRAL"
if "affect_override" not in st.session_state:
    st.session_state.affect_override = None


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("AAC Chatbot")

    # Persona selection
    try:
        users_resp = requests.get(f"{API_BASE}/users", timeout=3)
        users = users_resp.json().get("users", [])
    except Exception:
        users = []
        st.error("API not reachable — start the FastAPI server first.")

    user_options = {u["id"]: f"{u['name']} ({u['condition']})" for u in users}
    selected = st.selectbox(
        "Select persona",
        options=list(user_options.keys()),
        format_func=lambda k: user_options.get(k, k),
    )

    if selected != st.session_state.user_id:
        st.session_state.user_id = selected
        st.session_state.messages = []
        try:
            requests.post(f"{API_BASE}/session/reset", params={"user_id": selected})
        except Exception:
            pass

    st.divider()

    # Affect override (for demo / testing without webcam)
    st.subheader("Affect Override")
    st.caption("Simulates webcam affect detection")
    affect_choice = st.radio(
        "Current affect",
        ["Auto (webcam)", "HAPPY", "FRUSTRATED", "NEUTRAL", "SURPRISED"],
        index=0,
    )
    st.session_state.affect_override = (
        None if affect_choice == "Auto (webcam)" else affect_choice
    )

    st.divider()

    # Live affect indicator
    st.subheader("Detected Affect")
    affect_emoji = {
        "HAPPY": "😊",
        "FRUSTRATED": "😤",
        "NEUTRAL": "😐",
        "SURPRISED": "😲",
    }
    af = st.session_state.last_affect
    st.markdown(f"### {affect_emoji.get(af, '❓')} {af}")

    # Webcam placeholder
    st.divider()
    st.subheader("Webcam Feed")
    st.info(
        "Live webcam sensing runs in the sensing client.\nAffect is sent to the API automatically."
    )


# ── Main chat area ─────────────────────────────────────────────────────────────
st.header(f"Talking as: {user_options.get(st.session_state.user_id, '—')}")

chat_col, metrics_col = st.columns([3, 1])

with chat_col:
    for msg in st.session_state.messages:
        role_label = "Partner" if msg["role"] == "partner" else "AAC User"
        with st.chat_message("user" if msg["role"] == "partner" else "assistant"):
            st.markdown(f"**{role_label}:** {msg['content']}")

    query = st.chat_input("Type as the communication partner…")

    if query and st.session_state.user_id:
        st.session_state.messages.append({"role": "partner", "content": query})
        with st.chat_message("user"):
            st.markdown(f"**Partner:** {query}")

        with st.chat_message("assistant"):
            with st.spinner("Generating response…"):
                try:
                    payload = {
                        "user_id": st.session_state.user_id,
                        "query": query,
                        "affect_override": st.session_state.affect_override,
                    }
                    resp = requests.post(f"{API_BASE}/chat", json=payload, timeout=15)
                    resp.raise_for_status()
                    data = resp.json()

                    response_text = data.get("response", "I don't know.")
                    st.markdown(f"**AAC User:** {response_text}")

                    st.session_state.messages.append(
                        {"role": "aac_user", "content": response_text}
                    )
                    st.session_state.last_affect = data.get("affect", "NEUTRAL")
                    st.session_state.last_latency = data.get("latency", {})

                    if not data.get("guardrail_passed", True):
                        st.warning("⚠ Guardrail triggered — response was sanitised.")

                except requests.exceptions.Timeout:
                    st.error("Request timed out. Is the server running?")
                except Exception as e:
                    st.error(f"Error: {e}")

with metrics_col:
    st.subheader("Turn Latency (s)")
    lat = st.session_state.last_latency
    if lat:
        for key, label in [
            ("t_sensing", "Sensing"),
            ("t_intent", "Intent"),
            ("t_retrieval", "Retrieval"),
            ("t_generation", "Generation"),
            ("t_total", "**Total**"),
        ]:
            val = lat.get(key, 0.0)
            st.metric(label=label, value=f"{val:.3f}s")
    else:
        st.caption("No turn yet.")
