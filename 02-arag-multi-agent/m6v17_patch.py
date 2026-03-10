#!/usr/bin/env python3
"""M6 v17: Remove synthesizer template + all programmatic overrides.
Planner reasons directly about sub-answers with thinking ON."""

from pathlib import Path

BASE = Path("/projects/prjs1800/msc-thesis/02-arag-multi-agent")
M6 = BASE / "src" / "multi_agent" / "m6"

def patch(path, old, new):
    content = path.read_text()
    if old not in content:
        print(f"  WARNING: not found in {path.name}")
        print(f"  Looking for: {repr(old[:100])}")
        return False
    content = content.replace(old, new, 1)
    path.write_text(content)
    print(f"  OK: {path.name}")
    return True

print("=== Replace _synthesize: remove overrides, simple reasoning ===")

patch(
    M6 / "planner_agent.py",
    '''    async def _synthesize(
        self, observation: dict[str, Any], blackboard: Blackboard,
    ) -> int:
        question = observation["question"]
        sub_questions = observation["sub_questions"]
        verified_evidence = observation["verified_evidence"]
        entity_registry = observation["entity_registry"]

        evidence_blocks = self._build_evidence_blocks(sub_questions, verified_evidence, entity_registry)
        entity_str = "\\n".join(f"- {k} = {v}" for k, v in entity_registry.items()) if entity_registry else "None"

        expected_answer = getattr(blackboard, "expected_answer", "") or "an entity"
        prompt = self._synthesize_template.format(
            question=question,
            evidence_blocks=evidence_blocks,
            entity_registry=entity_str,
            expected_answer=expected_answer,
        )
        messages = [{"role": "user", "content": prompt}]

        total_tokens = 0
        try:
            response = await self._async_chat(messages=messages, tools=None, temperature=0.0)
            raw = response["message"].get("content", "")
            total_tokens += int(response.get("cost", 0.0) * 1_000_000)
        except Exception as exc:
            logger.error("Planner synthesis LLM error: %s", exc)
            answer = await blackboard.salvage_answer()
            answer = _normalize_answer(answer, question)
            await blackboard.set_final_answer(answer)
            await blackboard.terminate("SYNTHESIZED_FALLBACK")
            return 0

        answer = self._extract_answer(raw)
        if not answer or _is_refusal(answer):
            answer = await blackboard.salvage_answer()
            logger.info("Planner: synthesis empty/refusal, salvaged: '%s'", answer[:80])

        answer = _normalize_answer(answer, question)

        # ── Bridge answer correction ──
        # If the synthesizer returned an intermediate entity instead of the
        # final-hop answer, correct it. This is the #1 failure mode on MuSiQue.
        is_bridge = self._question_type == "bridge" or any(
            sq.get("dependencies") for sq in sub_questions
        )
        if is_bridge and answer:
            answer = self._correct_bridge_answer(
                answer, question, sub_questions, entity_registry,
            )

        # ── Comparison answer correction ──
        is_comparison = self._question_type == "comparison" or not any(
            sq.get("dependencies") for sq in sub_questions
        )
        if is_comparison and answer:
            corrected = self._correct_comparison_answer(
                answer, question, sub_questions, entity_registry,
            )
            if corrected and corrected.lower() != answer.lower():
                logger.info(
                    "Planner: comparison correction '%s' → '%s'",
                    answer[:40], corrected[:40],
                )
                answer = _normalize_answer(corrected, question)

        if self.enable_consistency_check and self._consistency_template and answer:
            answer, cons_tokens = await self._consistency_check(question, answer, evidence_blocks)
            total_tokens += cons_tokens

        await blackboard.set_final_answer(answer)
        await blackboard.terminate("SYNTHESIZED")
        logger.info("Planner: final answer '%s'", answer[:80])
        return total_tokens''',
    r'''    async def _synthesize(
        self, observation: dict[str, Any], blackboard: Blackboard,
    ) -> int:
        question = observation["question"]
        sub_questions = observation["sub_questions"]
        verified_evidence = observation["verified_evidence"]
        entity_registry = observation["entity_registry"]

        evidence_blocks = self._build_evidence_blocks(sub_questions, verified_evidence, entity_registry)
        entity_str = "\n".join(f"- {k} = {v}" for k, v in entity_registry.items()) if entity_registry else "None"
        expected_answer = getattr(blackboard, "expected_answer", "") or ""

        # Simple reasoning prompt — no template, no overrides. Planner reasons with thinking ON.
        prompt_parts = [
            f"Question: {question}",
            "",
            f"Sub-question answers:\n{evidence_blocks}",
            "",
            f"Entity registry:\n{entity_str}",
        ]
        if expected_answer:
            prompt_parts.append(f"\nThe answer should be: {expected_answer}")
        prompt_parts.append("\nUsing the sub-question answers above, answer the original question. Reply with ONLY the final answer — a short entity, name, number, date, or yes/no. Nothing else.")

        messages = [{"role": "user", "content": "\n".join(prompt_parts)}]

        total_tokens = 0
        try:
            response = await self._async_chat(messages=messages, tools=None, temperature=0.0)
            raw = response["message"].get("content", "")
            total_tokens += int(response.get("cost", 0.0) * 1_000_000)
        except Exception as exc:
            logger.error("Planner synthesis LLM error: %s", exc)
            answer = await blackboard.salvage_answer()
            answer = _normalize_answer(answer, question)
            await blackboard.set_final_answer(answer)
            await blackboard.terminate("SYNTHESIZED_FALLBACK")
            return 0

        answer = self._extract_answer(raw)
        if not answer or _is_refusal(answer):
            answer = await blackboard.salvage_answer()
            logger.info("Planner: synthesis empty/refusal, salvaged: '%s'", answer[:80])

        answer = _normalize_answer(answer, question)

        await blackboard.set_final_answer(answer)
        await blackboard.terminate("SYNTHESIZED")
        logger.info("Planner: final answer '%s'", answer[:80])
        return total_tokens''',
)

print("\n=== Done ===")
