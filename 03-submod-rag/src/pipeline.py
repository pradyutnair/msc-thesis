"""Submodular RAG v2: Sentence-level extraction + cross-hop evidence selection.

Architecture:
  1. Planner: decompose question into sub-question DAG
  2. Per sub-question (topological order):
     a. Programmatic retrieval (keyword + semantic, phrase-aware, RRF)
     b. Cross-hop scoring: boost chunks with resolved entities, penalize parent-SQ overlap
     c. Sentence extraction: pull ONLY sentences matching query entities
     d. Worker LLM: extract answer from focused sentences (not full chunks)
  3. Synthesizer: combine sub-answers into final answer

Novel contribution: Cross-hop submodular selection where evidence for SQ-j
penalizes overlap with parent SQ-i evidence, encouraging retrieval of NEW
information about the resolved entity.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import re
import time
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ── Prompts ──────────────────────────────────────────────────────────

DECOMPOSE_PROMPT = open(
    os.path.join(os.path.dirname(__file__), "..", "prompts", "decomposer.txt"),
    encoding="utf-8",
).read()

EXTRACT_PROMPT = """You are an answer extraction agent. Read the evidence sentences and extract the answer.

RULES:
1. The answer IS in the evidence — read every sentence carefully.
2. Output ONLY the answer: a name, date, number, or place. Maximum 5 words.
3. Do NOT output sentences or explanations. Just the entity.
4. If multiple entities appear, pick the one that DIRECTLY answers the question.

Example:
  Question: "Where was Albert Einstein born?"
  Evidence: "Albert Einstein was born on 14 March 1879 in Ulm, in the Kingdom of Württemberg."
  Answer: Ulm

Example:
  Question: "What year did Acornsoft end?"
  Evidence: "Acornsoft was dissolved in 1986 after being acquired by Olivetti."
  Answer: 1986
"""

SYNTHESIZE_PROMPT = """Combine sub-question answers to produce the final answer.

## Question
{question}

## Expected Answer Type
{expected_answer}

## Sub-Question Answers
{evidence_blocks}

## Instructions
- COMPARISON: compare numerically. "born first" = smaller year. Return the ENTITY NAME, not yes/no.
- YES/NO: normalize first. "American" = "US" = "United States". "Japanese American" IS "American". Bias yes.
- BRIDGE: the LAST sub-question's answer is the final answer.
- Maximum 5 words. No sentences.

FINAL ANSWER:"""


# ── Pipeline ─────────────────────────────────────────────────────────

class SubmodPipeline:

    def __init__(self, llm_client, chunks_file: str, index_dir: str,
                 model_name: str = "intfloat/e5-base-v2", device: str = None,
                 retrieval_budget: int = 20, selection_k: int = 5):
        self.llm = llm_client
        self.retrieval_budget = retrieval_budget
        self.selection_k = selection_k

        self.chunks = self._load_chunks(chunks_file)
        self.chunks_dict = {c["id"]: c["text"] for c in self.chunks}

        # Build sentence index per chunk for sentence-level extraction
        self.chunk_sentences: dict[str, list[str]] = {}
        for c in self.chunks:
            sents = [s.strip() for s in re.split(r'[.!?\n]+', c["text"]) if s.strip() and len(s.strip()) > 10]
            self.chunk_sentences[c["id"]] = sents

        self._load_index(index_dir)

        from sentence_transformers import SentenceTransformer
        self.embedder = SentenceTransformer(model_name, device=device)
        logger.info("SubmodPipeline ready: %d chunks, %d sentences",
                     len(self.chunks), len(self.sentences))

    @staticmethod
    def _load_chunks(path: str) -> list[dict]:
        raw = json.load(open(path, encoding="utf-8"))
        chunks = []
        for item in raw:
            if isinstance(item, str):
                parts = item.split(":", 1)
                chunks.append({"id": parts[0], "text": parts[1]})
            else:
                chunks.append(item)
        return chunks

    def _load_index(self, index_dir: str):
        with open(os.path.join(index_dir, "sentence_index.pkl"), "rb") as f:
            data = pickle.load(f)
        self.sentences = data["sentences"]
        self.sent_embeddings = data["embeddings"]
        self.sentence_to_chunk = data["sentence_to_chunk"]

    # ── Retrieval ────────────────────────────────────────────────────

    def keyword_search(self, keywords: list[str], top_k: int = 20) -> list[tuple[str, float]]:
        """Phrase-aware keyword search with 10x boost for exact multi-word matches."""
        phrases = []
        if len(keywords) > 1:
            phrases.append(" ".join(keywords).lower())
        for kw in keywords:
            if " " in kw.strip():
                phrases.append(kw.lower().strip())

        results = []
        for chunk in self.chunks:
            text_lower = chunk["text"].lower()
            score = 0.0
            for phrase in phrases:
                if phrase in text_lower:
                    score += 10.0 * len(phrase)
            for kw in keywords:
                if kw:
                    count = text_lower.count(kw.lower())
                    if count > 0:
                        score += count * len(kw)
            if score > 0:
                results.append((chunk["id"], score))
        results.sort(key=lambda x: -x[1])
        return results[:top_k]

    def semantic_search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        """Sentence-embedding based semantic search."""
        q_emb = self.embedder.encode([query], normalize_embeddings=True)[0]
        sims = np.dot(self.sent_embeddings, q_emb)
        top_idx = np.argsort(sims)[::-1][:top_k * 3]

        chunk_scores: dict[str, float] = {}
        for idx in top_idx:
            cid = self.sentence_to_chunk[idx]
            sim = float(sims[idx])
            if cid not in chunk_scores or sim > chunk_scores[cid]:
                chunk_scores[cid] = sim

        ranked = sorted(chunk_scores.items(), key=lambda x: -x[1])
        return ranked[:top_k]

    def retrieve(self, query: str, search_queries: list[str] | None = None,
                 budget: int | None = None) -> list[tuple[str, float]]:
        """Keyword + semantic retrieval with reciprocal rank fusion."""
        budget = budget or self.retrieval_budget

        kw_scores: dict[str, float] = {}
        keywords = [w for w in query.replace("?", "").split() if len(w) > 2]
        for cid, score in self.keyword_search(keywords, top_k=budget):
            kw_scores[cid] = kw_scores.get(cid, 0) + score

        for sq in (search_queries or []):
            full_query = [sq.strip()] if sq.strip() else []
            for cid, score in self.keyword_search(full_query, top_k=budget):
                kw_scores[cid] = kw_scores.get(cid, 0) + score
            kws = [w.strip() for w in sq.replace("?", "").split() if len(w.strip()) > 2]
            for cid, score in self.keyword_search(kws, top_k=budget):
                kw_scores[cid] = kw_scores.get(cid, 0) + score

        sem_scores = dict(self.semantic_search(query, top_k=budget))

        # RRF
        kw_ranked = sorted(kw_scores.items(), key=lambda x: -x[1])
        sem_ranked = sorted(sem_scores.items(), key=lambda x: -x[1])
        kw_rank = {cid: i for i, (cid, _) in enumerate(kw_ranked)}
        sem_rank = {cid: i for i, (cid, _) in enumerate(sem_ranked)}

        K_RRF = 60
        fused: dict[str, float] = {}
        for cid in set(kw_scores) | set(sem_scores):
            score = 0.0
            if cid in kw_rank:
                score += 1.0 / (K_RRF + kw_rank[cid])
            if cid in sem_rank:
                score += 1.0 / (K_RRF + sem_rank[cid])
            fused[cid] = score

        return sorted(fused.items(), key=lambda x: -x[1])[:budget]

    # ── Cross-Hop Evidence Selection (NOVEL) ─────────────────────────

    def cross_hop_select(self, candidates: list[tuple[str, float]], query: str,
                         parent_evidence_cids: list[str] | None = None,
                         resolved_entity: str | None = None,
                         k: int | None = None) -> list[tuple[str, float]]:
        """Cross-hop submodular selection.

        For dependent sub-questions, this:
        1. Boosts chunks mentioning the resolved entity from parent SQ
        2. Penalizes chunks that overlap with parent evidence (already seen)
        3. Selects k chunks maximizing relevance - redundancy - parent_overlap

        This encourages retrieval of NEW information about the resolved entity,
        not re-retrieval of the same evidence from the parent hop.
        """
        k = k or self.selection_k
        if len(candidates) <= k:
            return candidates

        cids = [cid for cid, _ in candidates]
        texts = [self.chunks_dict.get(cid, "") for cid in cids]

        # Encode
        q_emb = self.embedder.encode([query], normalize_embeddings=True)[0]
        chunk_embs = self.embedder.encode(texts, normalize_embeddings=True)
        relevance = np.dot(chunk_embs, q_emb)  # (n,)

        # Entity boost: if a chunk mentions the resolved entity, boost its score
        entity_boost = np.zeros(len(candidates))
        if resolved_entity and len(resolved_entity) > 2:
            for i, cid in enumerate(cids):
                if resolved_entity.lower() in texts[i].lower():
                    entity_boost[i] = 0.3  # Significant boost

        # Parent overlap penalty: penalize chunks similar to parent evidence
        parent_penalty = np.zeros(len(candidates))
        if parent_evidence_cids:
            parent_texts = [self.chunks_dict.get(cid, "") for cid in parent_evidence_cids
                           if cid in self.chunks_dict]
            if parent_texts:
                parent_embs = self.embedder.encode(parent_texts, normalize_embeddings=True)
                # Max similarity to any parent evidence chunk
                overlap = np.max(np.dot(chunk_embs, parent_embs.T), axis=1)
                parent_penalty = 0.5 * overlap  # Penalize overlap

        # Adjusted scores
        adjusted = relevance + entity_boost - parent_penalty

        # Greedy selection with within-set redundancy penalty
        sim_matrix = np.dot(chunk_embs, chunk_embs.T)
        selected: list[int] = []
        remaining = set(range(len(candidates)))

        for _ in range(k):
            best_gain = -np.inf
            best_idx = -1

            for i in remaining:
                # Redundancy with already-selected
                redundancy = max((sim_matrix[i, j] for j in selected), default=0.0)
                gain = adjusted[i] - 0.3 * redundancy

                if gain > best_gain:
                    best_gain = gain
                    best_idx = i

            if best_idx == -1:
                break
            selected.append(best_idx)
            remaining.discard(best_idx)

        return [(cids[i], float(adjusted[i])) for i in selected]

    # ── Sentence Extraction ──────────────────────────────────────────

    def extract_sentences(self, chunk_ids: list[str], query: str,
                          entity_hints: list[str] | None = None) -> list[str]:
        """Extract sentences from chunks that mention query entities.

        Returns focused sentences, not full chunks — this prevents the LLM
        from being confused by unrelated content in multi-topic chunks.
        """
        # Build search terms from query + entity hints
        terms = set()
        for word in query.replace("?", "").split():
            if len(word) > 3 and word.lower() not in {"what", "when", "where", "which", "does", "that", "this", "from", "about", "with", "have", "been", "were", "their"}:
                terms.add(word.lower())
        for hint in (entity_hints or []):
            if hint and len(hint) > 2:
                terms.add(hint.lower())

        matched_sentences: list[str] = []
        for cid in chunk_ids:
            sents = self.chunk_sentences.get(cid, [])
            for sent in sents:
                sent_lower = sent.lower()
                # A sentence is relevant if it contains at least 2 query terms
                # OR contains an entity hint as a phrase
                term_matches = sum(1 for t in terms if t in sent_lower)
                hint_match = any(h.lower() in sent_lower for h in (entity_hints or []) if h and len(h) > 2)

                if hint_match or term_matches >= 2:
                    matched_sentences.append(f"[{cid}] {sent}")

        # If no sentence-level matches, fall back to full chunk text (truncated)
        if not matched_sentences:
            for cid in chunk_ids[:3]:
                text = self.chunks_dict.get(cid, "")
                if text:
                    matched_sentences.append(f"[{cid}] {text[:300]}")

        return matched_sentences

    # ── Decomposition ────────────────────────────────────────────────

    def decompose(self, question: str) -> tuple[list[dict], str, str]:
        prompt = DECOMPOSE_PROMPT.replace("{question}", question)
        response = self.llm.chat(
            messages=[{"role": "user", "content": prompt}],
            tools=None, temperature=0.0,
        )
        raw = response["message"].get("content", "")
        return self._parse_decomposition(raw, question)

    @staticmethod
    def _parse_decomposition(raw: str, question: str):
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
        raw = re.sub(r"<think>.*", "", raw, flags=re.DOTALL)
        raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        raw = re.sub(r"\s*```$", "", raw)
        try:
            data = json.loads(raw.strip())
        except json.JSONDecodeError:
            return [{"id": 0, "text": question, "dependencies": [],
                     "search_queries": [question], "known_entities": [],
                     "search_hints": []}], "single_hop", "an entity"

        q_type = data.get("question_type", "single_hop")
        expected = data.get("expected_answer", "an entity")
        sqs = []
        for sq in data.get("sub_questions", []):
            sqs.append({
                "id": int(sq["index"]),
                "text": sq["text"],
                "dependencies": [int(d) for d in sq.get("depends_on", [])],
                "search_queries": sq.get("search_queries", []),
                "known_entities": sq.get("known_entities", []),
                "search_hints": sq.get("search_hints", []),
            })
        if not sqs:
            sqs = [{"id": 0, "text": question, "dependencies": [],
                    "search_queries": [question], "known_entities": [],
                    "search_hints": []}]
        return sqs, q_type, expected

    # ── LLM Extraction ───────────────────────────────────────────────

    def extract_answer(self, sub_question: str, evidence_sentences: list[str]) -> str:
        """Extract answer from focused sentences."""
        evidence = "\n".join(evidence_sentences[:15])  # Cap at 15 sentences
        response = self.llm.chat(
            messages=[
                {"role": "system", "content": EXTRACT_PROMPT},
                {"role": "user", "content": f"Question: {sub_question}\n\nEvidence:\n{evidence}"},
            ],
            tools=None, temperature=0.0,
        )
        raw = response["message"].get("content", "")
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
        raw = re.sub(r"<think>.*", "", raw, flags=re.DOTALL)
        answer = raw.strip().split("\n")[0].strip().strip("\"'`*")
        answer = re.sub(r"^(?:The\s+)?(?:final\s+)?answer\s+is\s+", "", answer, flags=re.IGNORECASE)
        answer = re.sub(r"^(?:FINAL\s+)?ANSWER\s*:\s*", "", answer, flags=re.IGNORECASE)
        answer = re.sub(r"^(?:Based\s+on|According\s+to)\s+(?:the\s+)?(?:evidence|information)[^,]*,\s*", "", answer, flags=re.IGNORECASE)
        # Extract entity from verbose patterns
        for pattern in [
            r"^(?:The\s+)?(?:nationality|country|birthplace|director|city|region|publisher|performer|composer)\s+(?:of\s+.+?\s+)?(?:is|was)\s+(.+?)$",
            r"^.+?\s+(?:was|is)\s+born\s+(?:in|on)\s+(.+?)$",
            r"^.+?\s+(?:was|is)\s+(?:located|based|situated)\s+(?:in|at)\s+(.+?)$",
            r"^.+?\s+(?:is|was)\s+mentioned\s+in\s+the\s+context\s+of\s+(.+?)$",
        ]:
            m = re.match(pattern, answer, re.IGNORECASE)
            if m:
                extracted = m.group(m.lastindex).strip().strip("\"'`.,;:!?")
                if extracted and 2 < len(extracted) < len(answer):
                    answer = extracted
                    break
        answer = re.sub(r"\s*[\.,;:!?]+$", "", answer)
        return answer

    # ── Synthesis ─────────────────────────────────────────────────────

    def synthesize(self, question: str, sub_questions: list[dict],
                   entity_registry: dict[str, str], expected_answer: str) -> str:
        parts = []
        for sq in sub_questions:
            answer = entity_registry.get(f"answer_{sq['id']}", "unknown")
            parts.append(f"Sub-question {sq['id']}: {sq['text']}\nAnswer: {answer}")
        evidence_block = "\n\n".join(parts)

        prompt = SYNTHESIZE_PROMPT.replace("{question}", question)
        prompt = prompt.replace("{evidence_blocks}", evidence_block)
        prompt = prompt.replace("{expected_answer}", expected_answer or "an entity")

        response = self.llm.chat(
            messages=[{"role": "user", "content": prompt}],
            tools=None, temperature=0.0,
        )
        raw = response["message"].get("content", "")
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
        raw = re.sub(r"<think>.*", "", raw, flags=re.DOTALL)
        match = re.search(r"FINAL\s*ANSWER\s*:\s*(.+)", raw, re.IGNORECASE)
        if match:
            return match.group(1).strip().strip("\"'`*")
        lines = [l.strip() for l in raw.strip().split("\n") if l.strip()]
        return lines[-1] if lines else ""

    # ── Full Pipeline ─────────────────────────────────────────────────

    def run(self, question: str) -> dict[str, Any]:
        t0 = time.monotonic()

        sub_questions, q_type, expected_answer = self.decompose(question)
        logger.info("Decomposed into %d SQs (%s): %s", len(sub_questions), q_type, question[:60])

        entity_registry: dict[str, str] = {}
        sq_evidence_cids: dict[int, list[str]] = {}  # Track evidence per SQ for cross-hop
        order = self._topological_sort(sub_questions)

        for sq in order:
            # Resolve placeholders
            text = sq["text"]
            queries = list(sq["search_queries"])
            hints = list(sq.get("search_hints", []))
            known = list(sq.get("known_entities", []))

            for key, val in entity_registry.items():
                placeholder = f"[{key}]"
                text = text.replace(placeholder, val)
                queries = [q.replace(placeholder, val) for q in queries]
                hints = [h.replace(placeholder, val) for h in hints]

            # Phase 2a: Programmatic retrieval
            candidates = self.retrieve(text, queries, budget=self.retrieval_budget)

            # Phase 2b: Cross-hop evidence selection
            parent_cids: list[str] = []
            resolved_entity: str | None = None
            for dep_id in sq["dependencies"]:
                parent_cids.extend(sq_evidence_cids.get(dep_id, []))
                resolved_entity = entity_registry.get(f"answer_{dep_id}")

            selected = self.cross_hop_select(
                candidates, text,
                parent_evidence_cids=parent_cids if parent_cids else None,
                resolved_entity=resolved_entity,
                k=self.selection_k,
            )

            selected_cids = [cid for cid, _ in selected]
            sq_evidence_cids[sq["id"]] = selected_cids

            # Phase 2c: Sentence-level extraction
            entity_hints = known + hints + ([resolved_entity] if resolved_entity else [])
            evidence_sentences = self.extract_sentences(selected_cids, text, entity_hints)

            # Phase 2d: LLM extraction
            answer = self.extract_answer(text, evidence_sentences)
            entity_registry[f"answer_{sq['id']}"] = answer
            logger.info("  SQ-%d: '%s' -> '%s' (%d sents from %d chunks)",
                        sq["id"], text[:50], answer[:40],
                        len(evidence_sentences), len(selected_cids))

        # Phase 3: Synthesize
        if len(sub_questions) == 1:
            final_answer = entity_registry.get("answer_0", "")
        else:
            final_answer = self.synthesize(question, sub_questions, entity_registry, expected_answer)

        elapsed = time.monotonic() - t0
        logger.info("Answer: '%s' (%.1fs)", final_answer[:40], elapsed)

        return {
            "pred_answer": final_answer,
            "entity_registry": entity_registry,
            "sub_questions": [{"id": sq["id"], "text": sq["text"],
                               "answer": entity_registry.get(f"answer_{sq['id']}", "")}
                              for sq in order],
            "question_type": q_type,
            "expected_answer": expected_answer,
            "wall_clock_seconds": elapsed,
        }

    @staticmethod
    def _topological_sort(sub_questions: list[dict]) -> list[dict]:
        solved: set[int] = set()
        remaining = list(sub_questions)
        order: list[dict] = []
        while remaining:
            progress = False
            for sq in list(remaining):
                if all(d in solved for d in sq["dependencies"]):
                    order.append(sq)
                    solved.add(sq["id"])
                    remaining.remove(sq)
                    progress = True
                    break
            if not progress:
                order.extend(remaining)
                break
        return order
