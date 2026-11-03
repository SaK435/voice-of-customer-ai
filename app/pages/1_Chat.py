"""Chat with your reviews: a Streamlit page for the review agent."""

import sys
from pathlib import Path

import streamlit as st

# Make the src folder importable from this page
sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))
from agent import SYSTEM_PROMPT, run_agent  # noqa: E402

EXAMPLES = [
    "What do unhappy customers complain about most?",
    "How do people feel about the scent?",
    "Is price a problem for customers?",
    "What do customers talk about the most?",
]

st.set_page_config(page_title="Chat with your reviews", layout="wide")
st.title("Chat with your reviews")
st.caption("A local AI agent that answers questions by running analyses on the review data.")

# Conversation memory survives page reruns thanks to session_state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    st.session_state.history = []

# Sidebar: example questions and a reset button
st.sidebar.markdown("**Try asking**")
clicked = None
for example in EXAMPLES:
    if st.sidebar.button(example):
        clicked = example
if st.sidebar.button("Clear chat"):
    del st.session_state["messages"]
    del st.session_state["history"]
    st.rerun()


def show_tools(tools: list) -> None:
    if tools:
        with st.expander("How I got this answer"):
            for t in tools:
                st.code(t, language="python")


# Show the conversation so far
for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        show_tools(turn.get("tools", []))

# New question from the chat box or an example button
question = st.chat_input("Ask about the reviews...") or clicked
if question:
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.history.append({"role": "user", "content": question})
    st.session_state.messages.append({"role": "user", "content": question})

    tool_log = []
    with st.chat_message("assistant"):
        with st.spinner("Analysing reviews..."):
            answer = run_agent(
                st.session_state.messages,
                verbose=False,
                on_tool=lambda name, args: tool_log.append(f"{name}({args})"),
            )
        st.markdown(answer)
        show_tools(tool_log)
    st.session_state.history.append({"role": "assistant", "content": answer, "tools": tool_log})