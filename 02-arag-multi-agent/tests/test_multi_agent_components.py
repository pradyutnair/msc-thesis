"""Targeted unit tests for MA²RAG core components."""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest

import numpy as np

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arag.core.context import AgentContext
from arag.tools.read_chunk import ReadChunkTool
from src.multi_agent.dispatcher import Dispatcher
from src.multi_agent.evidence_cache import EvidenceCache
from src.multi_agent.types import CachedDocument, DecompositionPlan, SubQuestion


class DispatcherWaveTests(unittest.TestCase):
    def test_depends_on_only_creates_separate_waves(self) -> None:
        plan = DecompositionPlan(
            question_type="bridge",
            sub_questions=[
                SubQuestion(index=0, text="hop1", depends_on=[]),
                SubQuestion(index=1, text="hop2", depends_on=[0]),
            ],
            dependency_edges=[],
        )

        waves = Dispatcher._compute_waves(plan)
        wave_indices = [[sq.index for sq in wave] for wave in waves]
        self.assertEqual(wave_indices, [[0], [1]])


class EvidenceCacheTests(unittest.TestCase):
    def test_get_relevant_returns_closest_doc(self) -> None:
        cache = EvidenceCache(enabled=True)
        asyncio.run(
            cache.put(
                CachedDocument(
                    doc_id="a",
                    text="A",
                    embedding=np.array([1.0, 0.0], dtype=np.float32),
                    source_agent=0,
                    retrieval_score=0.8,
                )
            )
        )
        asyncio.run(
            cache.put(
                CachedDocument(
                    doc_id="b",
                    text="B",
                    embedding=np.array([0.0, 1.0], dtype=np.float32),
                    source_agent=1,
                    retrieval_score=0.7,
                )
            )
        )

        result = asyncio.run(
            cache.get_relevant(np.array([1.0, 0.0], dtype=np.float32), top_k=1)
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].doc_id, "a")


class ReadChunkCacheWriteThroughTests(unittest.TestCase):
    class _DummyCache:
        def __init__(self):
            self.docs = []

        def put_sync(self, doc: CachedDocument) -> bool:
            self.docs.append(doc)
            return True

    def test_read_chunk_writes_to_cache_when_available(self) -> None:
        chunks = [{"id": "1", "text": "Chunk one text."}]
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
            json.dump(chunks, tf)
            chunks_path = tf.name

        cache = self._DummyCache()
        tool = ReadChunkTool(chunks_file=chunks_path, evidence_cache=cache)

        context = AgentContext()
        context.source_agent = 3

        _, log = tool.execute(context=context, chunk_ids=["1"])
        self.assertEqual(log["cache_writes"], 1)
        self.assertEqual(len(cache.docs), 1)
        self.assertEqual(cache.docs[0].doc_id, "1")
        self.assertEqual(cache.docs[0].source_agent, 3)


if __name__ == "__main__":
    unittest.main()
