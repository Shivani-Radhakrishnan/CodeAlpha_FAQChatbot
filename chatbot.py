"""BotBuddy Streamlit UI. All matching logic lives in :mod:`botbuddy.matcher`.

Run with:  streamlit run chatbot.py
"""

from __future__ import annotations

import html
import logging
import random

import streamlit as st
import streamlit.components.v1 as components

from botbuddy import config
from botbuddy.matcher import FAQDataError, FAQMatcher, MatchResult

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)  # quiet model-download chatter
logger = logging.getLogger("botbuddy.app")

QUICK_ACTIONS: dict[str, str] = {
    "🛍 Track My Order": "Can I track my order?",
    "🔄 Start a Return": "What is your return policy?",
    "💬 Contact Support": "How do I contact support?",
}
RANDOM_SUGGESTIONS: tuple[str, ...] = (
    "Where's my order?",
    "How do I return something?",
    "Do you offer student discounts?",
    "Can I pay in installments?",
    "Do you ship abroad?",
)
THANKS_WORDS: tuple[str, ...] = ("thank", "thanks", "thx", "appreciate")

CSS = """
<style>
    @keyframes fadeIn {
        0% { opacity: 0; transform: translateY(10px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    .chat-message {
        animation: fadeIn 0.3s ease-out;
        margin: 10px 0;
        padding: 15px;
        border-radius: 15px;
    }
    .user-message {
        background-color: #f0f2f6;
        color: black;
        margin-left: 20%;
        border-bottom-right-radius: 5px !important;
    }
    .bot-message {
        background-color: #4f8bf9;
        color: white;
        margin-right: 20%;
        border-bottom-left-radius: 5px !important;
    }
    .meta {
        display: block;
        margin-top: 6px;
        font-size: 0.75rem;
        opacity: 0.85;
    }
    .stTextInput>div>div>input {
        padding: 12px !important;
        border-radius: 15px !important;
    }
    .header {
        background: linear-gradient(90deg, #4f8bf9, #6a5acd);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
    }
</style>
"""

CONFETTI_HTML = """
<canvas id="confetti" style="position:fixed;top:0;left:0;width:100%;height:100%;z-index:9999;pointer-events:none"></canvas>
<script>
const canvas = document.getElementById('confetti');
const ctx = canvas.getContext('2d');
canvas.width = window.innerWidth; canvas.height = window.innerHeight;
const colors = ['#ff0000', '#00ff00', '#0000ff', '#ffff00', '#ff00ff', '#00ffff'];
const pieces = Array.from({length: 150}, () => ({
  x: Math.random() * canvas.width, y: Math.random() * canvas.height - canvas.height,
  r: Math.random() * 4 + 1, d: Math.random() * 3 + 2,
  color: colors[Math.floor(Math.random() * colors.length)]
}));
function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  pieces.forEach(p => {
    ctx.beginPath(); ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
    ctx.fillStyle = p.color; ctx.fill();
    p.y += p.d;
    if (p.y > canvas.height) { p.y = -10; p.x = Math.random() * canvas.width; }
  });
  requestAnimationFrame(draw);
}
setTimeout(() => canvas.remove(), 3000);
draw();
</script>
"""


@st.cache_resource(show_spinner="Loading the FAQ knowledge base...")
def get_matcher() -> FAQMatcher:
    """Build the matcher once per server process."""
    return FAQMatcher.from_csv(config.FAQ_PATH, method=config.DEFAULT_METHOD)


def init_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("show_details", True)
    st.session_state.setdefault("confetti", False)


def ask(question: str) -> None:
    """Append a user question and BotBuddy's reply to the conversation."""
    question = question.strip()
    if not question:
        return
    result: MatchResult = get_matcher().respond(question)
    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.messages.append(
        {
            "role": "bot",
            "content": result.answer,
            "status": result.status,
            "score": result.score,
            "matched": result.faq.question if result.faq else None,
            "suggestions": [c.faq.question for c in result.suggestions],
        }
    )
    if any(word in question.lower() for word in THANKS_WORDS):
        st.session_state.confetti = True


def render_message(index: int, message: dict) -> None:
    """Render one chat bubble; user text is HTML-escaped before display."""
    text = html.escape(message["content"])
    if message["role"] == "user":
        st.markdown(f'<div class="chat-message user-message">{text}</div>', unsafe_allow_html=True)
        return

    meta = ""
    if st.session_state.show_details and "status" in message:
        if message["status"] == "answer":
            meta = (
                f'<span class="meta">Matched "{html.escape(message["matched"])}" '
                f"· confidence {message['score']:.0%}</span>"
            )
        else:
            meta = (
                f'<span class="meta">Best score {message["score"]:.0%} '
                "- below the answer threshold</span>"
            )
    st.markdown(
        f'<div class="chat-message bot-message">{text}{meta}</div>', unsafe_allow_html=True
    )
    if message.get("suggestions"):
        st.caption("Did you mean:")
        cols = st.columns(len(message["suggestions"]))
        for col, suggestion in zip(cols, message["suggestions"]):
            if col.button(suggestion, key=f"suggest-{index}-{suggestion}", use_container_width=True):
                ask(suggestion)
                st.rerun()


def render_sidebar(matcher: FAQMatcher) -> None:
    with st.sidebar:
        st.subheader("✨ Quick Actions")
        for label, question in QUICK_ACTIONS.items():
            if st.button(label, use_container_width=True):
                ask(question)
                st.rerun()
        st.divider()
        st.markdown("### 💡 Tips")
        st.info(
            "- Ask naturally, e.g. *Where's my package?*\n"
            "- Not sure? I'll suggest the closest questions\n"
            "- Type *human* to reach a person"
        )
        st.divider()
        st.toggle("Show match details", key="show_details")
        if st.button("🧹 Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        st.caption(
            f"{len(matcher.faqs)} FAQs · matching method: `{matcher.method}` "
            f"· answer threshold {matcher.threshold:.2f}"
        )


def main() -> None:
    st.set_page_config(page_title="BotBuddy", page_icon="🤖")
    st.markdown(CSS, unsafe_allow_html=True)
    init_state()

    try:
        matcher = get_matcher()
    except (FAQDataError, RuntimeError) as exc:
        logger.exception("Could not start BotBuddy")
        st.error(f"BotBuddy could not load its knowledge base: {exc}")
        st.stop()

    st.markdown(
        """
        <div class="header">
            <h1 style="margin:0;padding:0;font-size:1.9rem">🤖 BotBuddy - Your Shopping Assistant</h1>
            <p style="margin:0;padding:0;font-size:0.9rem">Ask me anything about orders, returns, or products!</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_sidebar(matcher)

    if not st.session_state.messages:
        st.markdown(
            """
            <div class="chat-message bot-message">
                <strong>BotBuddy:</strong> Hi there! I'm here to help with:
                <ul>
                    <li>Order tracking</li>
                    <li>Returns &amp; exchanges</li>
                    <li>Shipping and payments</li>
                    <li>Your account</li>
                </ul>
                Try asking: "Where's my order?" or "How do I return something?"
            </div>
            """,
            unsafe_allow_html=True,
        )

    for i, message in enumerate(st.session_state.messages):
        render_message(i, message)

    if st.session_state.confetti:
        components.html(CONFETTI_HTML, height=0)
        st.session_state.confetti = False

    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input(
            "Type your message...", placeholder="Ask me anything...", label_visibility="collapsed"
        )
        cols = st.columns([5, 1])
        send = cols[0].form_submit_button("Send", use_container_width=True)
        surprise = cols[1].form_submit_button("🎯", help="Ask a random example question", use_container_width=True)
    if send and user_input.strip():
        ask(user_input)
        st.rerun()
    if surprise:
        ask(random.choice(RANDOM_SUGGESTIONS))
        st.rerun()


if __name__ == "__main__":
    main()
