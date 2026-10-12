"""Smoke tests for the Streamlit UI using Streamlit's AppTest harness."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "chatbot.py")


@pytest.fixture
def app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception
    return at


def _button(at: AppTest, label: str):
    return next(b for b in at.button if label in b.label)


def _bubbles(at: AppTest) -> list[str]:
    return [m.value for m in at.markdown if "chat-message" in m.value]


def test_app_starts_with_welcome(app):
    assert any("Hi there!" in b for b in _bubbles(app))


def test_question_gets_answer_and_confidence(app):
    app.text_input[0].input("Can I track my order?")
    _button(app, "Send").click()
    app.run()
    bubbles = _bubbles(app)
    assert any("tracking link" in b for b in bubbles)
    assert any("confidence" in b for b in bubbles)


def test_user_html_is_escaped(app):
    app.text_input[0].input("<b>bold</b> track order")
    _button(app, "Send").click()
    app.run()
    assert any("&lt;b&gt;bold&lt;/b&gt;" in b for b in _bubbles(app))


def test_quick_action_and_clear(app):
    _button(app, "Track My Order").click().run()
    assert len(app.session_state.messages) == 2
    _button(app, "Clear chat").click().run()
    assert app.session_state.messages == []


def test_did_you_mean_button_asks_the_suggestion(app):
    app.session_state["messages"] = [
        {"role": "user", "content": "pasword"},
        {
            "role": "bot",
            "content": "Sorry...",
            "status": "suggest",
            "score": 0.2,
            "matched": None,
            "suggestions": ["How do I reset my password?"],
        },
    ]
    app.run()
    _button(app, "How do I reset my password?").click().run()
    assert app.session_state.messages[-2]["content"] == "How do I reset my password?"
    assert "Forgot Password" in app.session_state.messages[-1]["content"]
