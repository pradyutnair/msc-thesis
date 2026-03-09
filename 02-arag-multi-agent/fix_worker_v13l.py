"""Apply targeted verbose answer cleanup to worker_agent.py for v13l.

Fix: After _clean_answer, if answer is >60 chars and contains refusal patterns,
replace with "" so the sub-question is marked FAILED instead of posting garbage.

This is SAFE because:
- Only targets answers >60 chars (legitimate short answers untouched)
- Only triggers on clear refusal language patterns
- Different from v13j's broad _is_refusal which checked ALL answers
"""

path = "/projects/prjs1800/msc-thesis/02-arag-multi-agent/src/multi_agent/m6/worker_agent.py"
with open(path, "r") as f:
    content = f.read()

# Add verbose answer cleanup after _clean_answer call in act()
old = '''        answer = self._clean_answer(answer)

        await blackboard.post_evidence(evidence, sq_id, answer, self.agent_id)

        is_usable = bool(answer) and answer.lower() not in ("unknown", "error", "")'''

new = '''        answer = self._clean_answer(answer)

        # Targeted verbose/refusal cleanup: only for long answers with clear refusal text
        if len(answer) > 60:
            answer_lower = answer.lower()
            _verbose_patterns = [
                "the evidence does not", "does not mention", "not explicitly mentioned",
                "no evidence confirms", "not specified in", "the provided documents",
                "there is no ", "cannot be determined", "not found in the",
            ]
            if any(p in answer_lower for p in _verbose_patterns):
                logger.info("%s: SQ-%d verbose/refusal answer cleared: '%s'", self.agent_id, sq_id, answer[:60])
                answer = ""

        await blackboard.post_evidence(evidence, sq_id, answer, self.agent_id)

        is_usable = bool(answer) and answer.lower() not in ("unknown", "error", "")'''

assert old in content, f"Old string not found in worker_agent.py"
content = content.replace(old, new)

with open(path, "w") as f:
    f.write(content)

# Verify
with open(path, "r") as f:
    t = f.read()
assert "_verbose_patterns" in t, "Missing verbose patterns"
print("Applied worker verbose answer cleanup fix")
