"""Shared fixtures."""

from __future__ import annotations

import pytest

from botbuddy import config
from botbuddy.matcher import FAQ, FAQMatcher, load_faqs


@pytest.fixture(scope="session")
def faqs() -> list[FAQ]:
    return load_faqs(config.FAQ_PATH)


@pytest.fixture(scope="session")
def hybrid(faqs: list[FAQ]) -> FAQMatcher:
    return FAQMatcher(faqs, method="hybrid")
