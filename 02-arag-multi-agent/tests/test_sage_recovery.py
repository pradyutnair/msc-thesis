"""Unit tests for SAGE recovery reliability features."""

from __future__ import annotations

import asyncio
import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.multi_agent.planner import RetrievalTask, SagePlan
from src.multi_agent.sage_pipeline import SagePipeline
from src.multi_agent.search_agent import is_unsupported_answer
from src.multi_agent.synthesizer import Synthesizer
from src.multi_agent.verifier import Verifier


class _FakeLLM:
    def __init__(self, content: str = "not-json"):
        self.content = content

    async def async_chat(self, messages, tools=None, temperature=0.0):
        return {
            "message": {"content": self.content},
            "cost": 0.0,
        }


class VerifierFailClosedTests(unittest.TestCase):
    def test_verifier_parse_failure_returns_insufficient_and_gaps(self) -> None:
        verifier = Verifier(llm_client=_FakeLLM(content="not-json"), fail_open_on_error=False)
        plan = SagePlan(
            question_type="bridge",
            tasks=[
                RetrievalTask(
                    id=0,
                    query="alpha, birthplace",
                    search_method="keyword",
                    goal="Find birthplace",
                    depends_on=[],
                )
            ],
            expected_answer_type="location",
        )

        res = asyncio.run(
            verifier.verify(
                question="Where was alpha born?",
                plan=plan,
                chunks=[],
                extracted_evidence_by_task=None,
            )
        )
        self.assertFalse(res.sufficient)
        self.assertFalse(res.parse_ok)
        self.assertGreaterEqual(len(res.gaps), 1)


class VerifierExtractionConsistencyTests(unittest.TestCase):
    def test_unsupported_extracted_claim_forces_insufficient(self) -> None:
        verifier = Verifier(
            llm_client=_FakeLLM(
                content=(
                    '{"sufficient": true, "verified_chunks": ["c0"], '
                    '"irrelevant_chunks": [], "gaps": []}'
                )
            ),
            fail_open_on_error=False,
        )
        plan = SagePlan(
            question_type="bridge",
            tasks=[
                RetrievalTask(
                    id=0,
                    query="alpha, birthplace",
                    search_method="keyword",
                    goal="Find birthplace",
                    depends_on=[],
                )
            ],
            expected_answer_type="location",
        )

        chunks = [
            {
                "id": "c0",
                "task_id": 0,
                "text": "Alpha mountain has elevation 3000 meters.",
            }
        ]

        res = asyncio.run(
            verifier.verify(
                question="Where was alpha born?",
                plan=plan,
                chunks=chunks,
                extracted_evidence_by_task={0: ["Alpha was born in Paris."]},
            )
        )

        self.assertTrue(res.parse_ok)
        self.assertFalse(res.sufficient)
        self.assertGreaterEqual(len(res.gaps), 1)
        self.assertIsNotNone(res.failure_reason)
        self.assertIn("unsupported_extracted_evidence", res.failure_reason)


class DependencyResolutionTests(unittest.TestCase):
    def test_resolve_placeholders_requires_supported_dependency(self) -> None:
        task = RetrievalTask(
            id=1,
            query="[answer_0], population",
            search_method="keyword",
            goal="Find population of [answer_0]",
            depends_on=[0],
        )

        q1, g1, unresolved1 = SagePipeline._resolve_task_placeholders(
            task,
            {0: {"answer": "Paris", "supported": False}},
        )
        self.assertIn("[answer_0]", q1)
        self.assertIn(0, unresolved1)

        q2, g2, unresolved2 = SagePipeline._resolve_task_placeholders(
            task,
            {0: {"answer": "Paris", "supported": True}},
        )
        self.assertIn("Paris", q2)
        self.assertNotIn("[answer_0]", q2)
        self.assertEqual(unresolved2, [])
        self.assertIn("Paris", g2)


class UnsupportedAnswerTests(unittest.TestCase):
    def test_unsupported_answer_detected(self) -> None:
        self.assertTrue(
            is_unsupported_answer(
                answer="Some answer",
                loops=2,
                evidence_count=0,
                retrieved_chunk_count=0,
            )
        )
        self.assertFalse(
            is_unsupported_answer(
                answer="Some answer",
                loops=2,
                evidence_count=1,
                retrieved_chunk_count=0,
            )
        )


class SynthesizerSanitizerTests(unittest.TestCase):
    def test_sanitize_answer_removes_xml_and_markdown_wrappers(self) -> None:
        raw = "<final_answer>**Answer:** `Paris`</final_answer>"
        cleaned = Synthesizer.sanitize_answer(raw)
        self.assertEqual(cleaned, "Paris")


if __name__ == "__main__":
    unittest.main()
