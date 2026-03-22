"""M7 Pipeline: CORAL — Collaborative Evidence-Pooling Multi-Agent RAG.

Three pillars: Collaboration + Efficiency + Performance.

Flow:
  1. Decompose question into typed sub-questions with dependency DAG
  2. Compute execution levels (adaptive: concurrent vs sequential)
  3. For each level, process sub-questions:
     a. Evidence Pool Search: search predecessors' retrieved docs (0 LLM cost)
     b. New Corpus Retrieval: multi-query E5 sentence retrieval + neighborhood
     c. Worker Extraction: ONE unified LLM call (pool + new + predecessor context)
     d. Collaborative Correction: worker can correct predecessor's answer
     e. Post answer + docs to blackboard for downstream workers
  4. Synthesize final answer from blackboard state
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

from multi_agent.m7.blackboard import Blackboard, DecompositionPlan, SQStatus, SubQuestion
from multi_agent.m7.retriever import M7Retriever
from multi_agent.m7.llm_client import VllmChatClient, token_tracker
from multi_agent.m7.agents import DecomposerAgent, WorkerAgent, SynthesizerAgent

logger = logging.getLogger(__name__)

# -- Stopwords for evidence leads extraction ---------------------------

_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "just",
    "and", "but", "or", "if", "while", "although", "because", "until",
    "that", "which", "who", "whom", "this", "these", "those", "it", "its",
    "he", "she", "they", "them", "his", "her", "their", "we", "you", "i",
    "me", "my", "our", "your", "what", "also", "about", "up", "one", "two",
})


@dataclass
class M7PipelineResult:
    """Result from a single M7 pipeline run."""
    qid: str = ""
    question: str = ""
    gold_answer: str = ""
    pred_answer: str = ""
    wall_clock_seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    api_calls: int = 0
    retrieval_rounds: int = 0
    num_sub_questions: int = 0
    question_type: str = ""
    snapshot: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class M7Pipeline:
    """CORAL: Collaborative Evidence-Pooling Multi-Agent RAG Pipeline."""

    def __init__(
        self,
        llm: VllmChatClient,
        retriever: M7Retriever,
        decomposer_prompt: str,
        worker_prompt: str,
        synthesizer_prompt: str,
        top_k: int = 10,
        max_retries: int = 1,
        evidence_pool_top_k: int = 5,
    ):
        self.retriever = retriever
        self.decomposer = DecomposerAgent(llm, decomposer_prompt)
        self.worker = WorkerAgent(llm, worker_prompt)
        self.synthesizer = SynthesizerAgent(llm, synthesizer_prompt)
        self.top_k = top_k
        self.max_retries = max_retries
        self.evidence_pool_top_k = evidence_pool_top_k

    def answer_question(self, question: str) -> tuple[str, dict[str, Any]]:
        """Run the full CORAL pipeline on a single question.

        Returns (answer, trace_dict).
        """
        blackboard = Blackboard(question)

        # Phase 1: Decompose
        plan = self.decomposer.decompose(question)
        blackboard.set_plan(plan)
        logger.info(
            "Decomposed '%s' into %d SQs (type=%s)",
            question[:50], len(plan.sub_questions), plan.question_type,
        )

        total_retrieval_rounds = 0

        # Phase 2: Get execution levels (adaptive scheduling)
        levels = blackboard.get_execution_levels()
        logger.info(
            "Execution levels: %s",
            [[sq.index for sq in level] for level in levels],
        )

        # Phase 3: Process level by level
        for level_idx, level in enumerate(levels):
            if len(level) > 1:
                # Concurrent execution for independent sub-questions
                logger.info(
                    "Level %d: concurrent execution of SQs %s",
                    level_idx, [sq.index for sq in level],
                )
                with ThreadPoolExecutor(max_workers=len(level)) as pool:
                    futures = {
                        pool.submit(self._process_sq, sq, blackboard): sq
                        for sq in level
                    }
                    for future in as_completed(futures):
                        sq = futures[future]
                        try:
                            rounds = future.result()
                            total_retrieval_rounds += rounds
                        except Exception as exc:
                            logger.error(
                                "SQ-%d processing failed: %s", sq.index, exc,
                            )
                            blackboard.mark_failed(sq.index)
            else:
                # Sequential execution
                sq = level[0]
                logger.info("Level %d: sequential SQ-%d", level_idx, sq.index)
                try:
                    rounds = self._process_sq(sq, blackboard)
                    total_retrieval_rounds += rounds
                except Exception as exc:
                    logger.error(
                        "SQ-%d processing failed: %s", sq.index, exc,
                    )
                    blackboard.mark_failed(sq.index)

        # Phase 4: Synthesize
        answers = blackboard.get_all_answers()
        if answers:
            if len(plan.sub_questions) == 1 and 0 in answers:
                final_answer = answers[0]
            else:
                final_answer = self.synthesizer.synthesize(blackboard)
        else:
            # All failed: direct retrieval fallback
            final_answer = self._direct_fallback(question)

        snapshot = blackboard.get_snapshot()
        snapshot["retrieval_rounds"] = total_retrieval_rounds

        return final_answer, snapshot

    def _process_sq(
        self, sq: SubQuestion, blackboard: Blackboard,
    ) -> int:
        """Process a single sub-question through the CORAL worker pipeline.

        Returns number of new corpus retrieval rounds used.
        """
        retrieval_rounds = 0

        # Resolve placeholders with current blackboard state
        resolved_text = blackboard.resolve_placeholders(sq.text)
        resolved_queries = [
            blackboard.resolve_placeholders(q) for q in sq.search_queries
        ]

        # ----- STEP 1: Evidence Pool Search (0 LLM cost, 0 new retrieval) -----
        evidence_pool = blackboard.get_evidence_pool(sq.index)
        pool_results: list[dict[str, Any]] = []
        if evidence_pool:
            pool_results = self.retriever.search_evidence_pool(
                resolved_text, evidence_pool, top_k=self.evidence_pool_top_k,
            )
            logger.info(
                "SQ-%d evidence pool: %d docs available, %d selected",
                sq.index, len(evidence_pool), len(pool_results),
            )

        # ----- STEP 2: New corpus retrieval -----
        if resolved_queries:
            new_docs = self.retriever.multi_query_retrieve(
                resolved_queries, top_k=self.top_k,
            )
        else:
            new_docs = self.retriever.retrieve(
                resolved_text, top_k=self.top_k,
            )
        retrieval_rounds += 1

        # Deduplicate: remove from new_docs anything already in pool_results
        pool_doc_ids = {d.get("doc_id") for d in pool_results}
        new_docs = [d for d in new_docs if d.get("doc_id") not in pool_doc_ids]

        # ----- STEP 3: Build predecessor context for collaboration -----
        predecessor_context = ""
        verification_instruction = ""
        if sq.depends_on:
            context_parts = []
            for dep_idx in sq.depends_on:
                dep_answer = blackboard.get_answer(dep_idx)
                dep_docs = blackboard.get_documents_for(dep_idx)[:3]
                if dep_answer:
                    context_parts.append(
                        f"Previous hop {dep_idx} answered: {dep_answer}"
                    )
                if dep_docs:
                    context_parts.append("Their evidence:")
                    for doc in dep_docs:
                        content_preview = doc.get("content", "")[:200]
                        context_parts.append(
                            f"  [{doc.get('doc_id', '')}] {content_preview}"
                        )
            predecessor_context = "\n".join(context_parts)
            verification_instruction = (
                "IMPORTANT: Review the previous worker's answer above. "
                "If it seems wrong based on the evidence (e.g., a state "
                "instead of a country), put the corrected answer in "
                "predecessor_correction."
            )

        # ----- STEP 4: Worker extraction (1 LLM call) -----
        result = self.worker.extract(
            sub_question=resolved_text,
            answer_type=sq.answer_type,
            evidence_pool_docs=pool_results,
            new_docs=new_docs,
            predecessor_context=predecessor_context,
            verification_instruction=verification_instruction,
        )

        # ----- STEP 5: Handle predecessor correction -----
        correction = result.get("predecessor_correction", "").strip()
        if (
            correction
            and correction.lower() not in ("", "n/a", "none")
            and sq.depends_on
        ):
            for dep_idx in sq.depends_on:
                old_answer = blackboard.get_answer(dep_idx)
                if old_answer and old_answer != correction:
                    logger.info(
                        "Worker collaboration: SQ-%d corrected SQ-%d: "
                        "'%s' -> '%s'",
                        sq.index, dep_idx, old_answer, correction,
                    )
                    blackboard.correct_answer(dep_idx, correction)

            # Re-resolve with corrected predecessor answer and re-extract
            resolved_text = blackboard.resolve_placeholders(sq.text)
            resolved_queries = [
                blackboard.resolve_placeholders(q)
                for q in sq.search_queries
            ]

            # Re-retrieve with corrected entity
            if resolved_queries:
                new_docs = self.retriever.multi_query_retrieve(
                    resolved_queries, top_k=self.top_k,
                )
            else:
                new_docs = self.retriever.retrieve(
                    resolved_text, top_k=self.top_k,
                )
            retrieval_rounds += 1

            # Deduplicate again
            new_docs = [
                d for d in new_docs if d.get("doc_id") not in pool_doc_ids
            ]

            # Re-extract with corrected context
            result = self.worker.extract(
                sub_question=resolved_text,
                answer_type=sq.answer_type,
                evidence_pool_docs=pool_results,
                new_docs=new_docs,
                predecessor_context=predecessor_context,
                verification_instruction="",
            )

        # ----- STEP 6: Post to blackboard -----
        answer = result.get("answer", "").strip()
        all_docs = pool_results + new_docs
        leads = self._extract_evidence_leads(all_docs, answer)

        if answer and answer.lower() not in ("unknown", ""):
            blackboard.post_answer(sq.index, answer, all_docs, leads)
        else:
            # Retry: re-retrieve with the resolved question text directly
            retries = 0
            while (
                answer.lower() in ("unknown", "", "not found")
                and retries < self.max_retries
            ):
                retry_docs = self.retriever.retrieve(
                    resolved_text, top_k=self.top_k,
                )
                retrieval_rounds += 1
                retry_result = self.worker.extract(
                    sub_question=resolved_text,
                    answer_type=sq.answer_type,
                    evidence_pool_docs=pool_results,
                    new_docs=retry_docs,
                    predecessor_context=predecessor_context,
                    verification_instruction="",
                )
                answer = retry_result.get("answer", "").strip()
                if answer and answer.lower() not in ("unknown", ""):
                    all_docs = pool_results + retry_docs
                    leads = self._extract_evidence_leads(all_docs, answer)
                    break
                retries += 1

            if answer and answer.lower() not in ("unknown", ""):
                blackboard.post_answer(sq.index, answer, all_docs, leads)
            else:
                blackboard.mark_failed(sq.index)

        return retrieval_rounds

    def _extract_evidence_leads(
        self,
        documents: list[dict[str, Any]],
        answer: str,
    ) -> list[str]:
        """Extract distinctive bigrams from retrieved docs as evidence leads.

        These are passed to downstream hops to improve their retrieval.
        """
        all_text = " ".join(
            doc.get("content", "") for doc in documents[:10]
        ).lower()

        words = re.findall(r"[a-z]+(?:'[a-z]+)?", all_text)
        words = [w for w in words if w not in _STOPWORDS and len(w) > 2]

        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
        bigram_counts = Counter(bigrams)

        distinctive = [
            bg for bg, count in bigram_counts.most_common(50)
            if 1 <= count <= 3
        ]

        answer_lower = answer.lower().strip()
        if answer_lower and answer_lower not in ("unknown", "none"):
            distinctive = [answer_lower] + distinctive

        return distinctive[:10]

    def _direct_fallback(self, question: str) -> str:
        """When all sub-questions fail, do a direct retrieve + extract."""
        logger.warning("All sub-questions failed, attempting direct fallback")
        documents = self.retriever.retrieve(question, top_k=self.top_k)
        result = self.worker.extract(
            sub_question=question,
            answer_type="a short factual phrase",
            evidence_pool_docs=[],
            new_docs=documents,
            predecessor_context="",
            verification_instruction="",
        )
        answer = result.get("answer", "").strip()
        return answer if answer else "Unable to find answer"
