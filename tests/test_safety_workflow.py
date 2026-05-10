"""Unit tests for deterministic safety and fallback workflow behavior."""

from __future__ import annotations

import os
import sys
import types
import unittest
from unittest.mock import patch

from app.config import get_settings
from app.parser import parse_user_input
from app.rules import assess_safety
from app.triage_service import TriageService


class FailingLLMClient:
    """Fake LLM client used to verify fallback behavior without API calls."""

    def __init__(self):
        self.called = False

    def generate(self, *_args, **_kwargs):
        self.called = True
        raise RuntimeError("LLM unavailable in test")


class SafetyWorkflowTests(unittest.TestCase):
    def test_parser_extracts_basic_intent_and_signals(self):
        parsed = parse_user_input("I feel anxious and have a headache")

        self.assertEqual(parsed.intent, "health_support")
        self.assertIn("emotional_wellbeing", parsed.signals)
        self.assertIn("physical_symptoms", parsed.signals)

    def test_high_risk_keyword_requires_escalation(self):
        parsed = parse_user_input("I want to die")
        safety = assess_safety(parsed)

        self.assertEqual(safety.risk_level, "HIGH")
        self.assertTrue(safety.escalation_required)
        self.assertIn("want to die", safety.matched_keywords)

    def test_high_risk_messages_do_not_call_llm(self):
        llm = FailingLLMClient()
        service = TriageService(llm_client=llm)

        result = service.handle_message("I want to die")

        self.assertFalse(llm.called)
        self.assertFalse(result.used_llm)
        self.assertTrue(result.safety.escalation_required)

    def test_llm_failure_returns_structured_fallback(self):
        llm = FailingLLMClient()
        service = TriageService(llm_client=llm)

        result = service.handle_message("I feel tired today")

        self.assertTrue(llm.called)
        self.assertFalse(result.used_llm)
        self.assertIsNotNone(result.error)
        self.assertIn("fallback", result.response.limitations.lower())

    def test_streamlit_secret_sets_openai_environment_variable(self):
        fake_streamlit = types.SimpleNamespace(
            secrets={"OPENAI_API_KEY": "test-secret-key"}
        )

        with patch.dict(sys.modules, {"streamlit": fake_streamlit}):
            with patch.dict(os.environ, {}, clear=True):
                settings = get_settings()

                self.assertEqual(settings.openai_api_key, "test-secret-key")
                self.assertEqual(os.environ["OPENAI_API_KEY"], "test-secret-key")


if __name__ == "__main__":
    unittest.main()
