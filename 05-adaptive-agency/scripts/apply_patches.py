#!/usr/bin/env python3
"""Apply all Phase 0 patches to the codebase.

Run from the project root:
    python scripts/apply_patches.py

Patches:
  1. Replace build_tools() in run_escalation.py and collect_escalation_trajectories.py
  2. Fix SFT data filtering with gold-answer check (0D)
  3. Fix escalation reward with gold-answer tie-breaking (0E)
"""

import re
import sys
from pathlib import Path


def patch_build_tools(filepath: Path) -> bool:
    """Replace inline build_tools() with import from shared module."""
    text = filepath.read_text()

    # Check if already patched
    if "from arag.tools.build_tools import build_tools" in text:
        print(f"  {filepath.name}: already patched (build_tools)")
        return False

    # Remove the old build_tools function definition
    # Pattern: def build_tools(config: Config) -> ToolRegistry: ... (until next def or top-level code)
    old_func_pattern = re.compile(
        r"^def build_tools\(config.*?\n(?=\ndef |\n[a-zA-Z]|\nclass |\nasync def |\nif __name__)",
        re.MULTILINE | re.DOTALL,
    )

    match = old_func_pattern.search(text)
    if not match:
        print(f"  {filepath.name}: WARNING - could not find build_tools() definition")
        return False

    # Remove old function
    text = text[:match.start()] + text[match.end():]

    # Remove old imports that are no longer needed (they're in build_tools.py now)
    # Keep SemanticSearchTool import removal cautious
    old_imports = [
        "from arag.tools.keyword_search import KeywordSearchTool\n",
        "from arag.tools.read_chunk import ReadChunkTool\n",
        "from arag.tools.semantic_search import SemanticSearchTool\n",
    ]
    for imp in old_imports:
        text = text.replace(imp, "")

    # Add new import after the registry import
    text = text.replace(
        "from arag.tools.registry import ToolRegistry\n",
        "from arag.tools.registry import ToolRegistry\nfrom arag.tools.build_tools import build_tools\n",
    )

    filepath.write_text(text)
    print(f"  {filepath.name}: patched build_tools()")
    return True


def patch_sft_gold_filtering(filepath: Path) -> bool:
    """Fix 0D: Add gold-answer filtering to SFT data selection."""
    text = filepath.read_text()

    if "gold_answer" in text and "containment check" in text:
        print(f"  {filepath.name}: already patched (gold filtering)")
        return False

    # Replace _pick_best_answer to accept gold_answer parameter
    old_pick = '''def _pick_best_answer(structured: dict, agentic: dict) -> str | None:
    """Pick the best available answer from counterfactual pair.

    Priority: if both usable and match, use either. If both usable but
    different, prefer agentic (uses more thorough search). If only one
    usable, use that one.
    """
    s_ans = structured.get("answer", "")
    a_ans = agentic.get("answer", "")
    s_ok = _is_usable(s_ans)
    a_ok = _is_usable(a_ans)

    if a_ok and s_ok:
        # Both usable — prefer agentic (more thorough search)
        # But if structured found more evidence, prefer it
        s_ev = structured.get("evidence_count", 0)
        a_ev = agentic.get("evidence_count", 0)
        if s_ev > a_ev and _normalize(s_ans) != _normalize(a_ans):
            return s_ans
        return a_ans
    elif a_ok:
        return a_ans
    elif s_ok:
        return s_ans
    return None'''

    new_pick = '''def _pick_best_answer(structured: dict, agentic: dict, gold_answer: str = "") -> str | None:
    """Pick the best available answer from counterfactual pair.

    Uses gold-answer containment check to resolve disagreements between
    structured and agentic workers. This eliminates noisy training signal
    from incorrect answers.
    """
    s_ans = structured.get("answer", "")
    a_ans = agentic.get("answer", "")
    s_ok = _is_usable(s_ans)
    a_ok = _is_usable(a_ans)

    if a_ok and s_ok:
        # Both usable — check if they agree
        if _normalize(s_ans) == _normalize(a_ans):
            return a_ans  # agreement

        # Disagreement: use gold answer to pick the correct one
        if gold_answer:
            gold_norm = _normalize(gold_answer)
            s_match = gold_norm in _normalize(s_ans) or _normalize(s_ans) in gold_norm
            a_match = gold_norm in _normalize(a_ans) or _normalize(a_ans) in gold_norm
            if a_match and not s_match:
                return a_ans
            elif s_match and not a_match:
                return s_ans
            elif not s_match and not a_match:
                return None  # neither matches gold — skip this SQ
        # No gold or both match: prefer agentic
        return a_ans
    elif a_ok:
        return a_ans
    elif s_ok:
        return s_ans
    return None'''

    if old_pick in text:
        text = text.replace(old_pick, new_pick)
    else:
        print(f"  {filepath.name}: WARNING - could not find exact _pick_best_answer()")
        return False

    # Update the call site to pass gold_answer
    old_call = "best_answer = _pick_best_answer(structured, agentic)"
    new_call = "best_answer = _pick_best_answer(structured, agentic, q_data.get(\"gold_answer\", \"\"))"

    if old_call in text:
        text = text.replace(old_call, new_call)
    else:
        print(f"  {filepath.name}: WARNING - could not find _pick_best_answer call site")

    # Add filtering stats logging
    old_log = '''        logger.info(
            "SFT dataset: %d samples from %d SQs "
            "(skipped=%d aggregate, %d both_failed, %d files)",
            len(self.samples), total_sqs, n_skipped, n_both_failed,
            len(trajectories_files),
        )'''

    new_log = '''        logger.info(
            "SFT dataset: %d samples from %d SQs "
            "(skipped=%d aggregate, %d both_failed, %d usable, %d files)",
            len(self.samples), total_sqs, n_skipped, n_both_failed,
            n_usable, len(trajectories_files),
        )'''

    text = text.replace(old_log, new_log)

    filepath.write_text(text)
    print(f"  {filepath.name}: patched with gold-answer SFT filtering")
    return True


def patch_escalation_gold_reward(filepath: Path) -> bool:
    """Fix 0E: Add gold-answer tie-breaking to escalation rewards."""
    text = filepath.read_text()

    if "gold_answer: str" in text and "def compute_escalation_rewards" in text:
        print(f"  {filepath.name}: already patched (gold reward)")
        return False

    # Replace compute_escalation_rewards signature and body
    old_sig = '''def compute_escalation_rewards(
    structured: dict,
    agentic: dict,
    cost_penalty: float = 0.2,
) -> tuple[float, float]:'''

    new_sig = '''def compute_escalation_rewards(
    structured: dict,
    agentic: dict,
    cost_penalty: float = 0.2,
    gold_answer: str = "",
) -> tuple[float, float]:'''

    if old_sig in text:
        text = text.replace(old_sig, new_sig)
    else:
        print(f"  {filepath.name}: WARNING - could not find compute_escalation_rewards sig")
        return False

    # Replace the ambiguous case (both usable, different answers)
    old_ambiguous = '''    elif s_usable and a_usable and not answers_match:
        # Both found different answers — uncertain which is better
        # Slight preference for ACCEPT (avoid extra compute when uncertain)
        r_accept = 0.3
        r_escalate = 0.5 - cost_penalty'''

    new_ambiguous = '''    elif s_usable and a_usable and not answers_match:
        # Both found different answers — use gold to determine correct one
        if gold_answer:
            gold_norm = _normalize(gold_answer)
            s_match = gold_norm in _normalize(s_answer) or _normalize(s_answer) in gold_norm
            a_match = gold_norm in _normalize(a_answer) or _normalize(a_answer) in gold_norm
            if s_match and not a_match:
                # Structured was correct — ACCEPT is right
                r_accept = 1.0
                r_escalate = -cost_penalty
            elif a_match and not s_match:
                # Agentic was correct — ESCALATE is justified
                r_accept = -0.5
                r_escalate = 1.0 - cost_penalty
            else:
                # Both match or neither — ambiguous
                r_accept = 0.3
                r_escalate = 0.5 - cost_penalty
        else:
            # No gold answer — slight preference for ACCEPT
            r_accept = 0.3
            r_escalate = 0.5 - cost_penalty'''

    if old_ambiguous in text:
        text = text.replace(old_ambiguous, new_ambiguous)
    else:
        print(f"  {filepath.name}: WARNING - could not find ambiguous case block")

    # Update call site in EscalationGRPODataset to pass gold_answer
    old_call = '''                    # Compute counterfactual rewards
                    r_accept, r_escalate = compute_escalation_rewards(
                        structured, agentic, cost_penalty,
                    )'''

    new_call = '''                    # Compute counterfactual rewards (with gold-answer tie-breaking)
                    r_accept, r_escalate = compute_escalation_rewards(
                        structured, agentic, cost_penalty,
                        gold_answer=q_data.get("gold_answer", ""),
                    )'''

    if old_call in text:
        text = text.replace(old_call, new_call)
    else:
        print(f"  {filepath.name}: WARNING - could not find reward call site")

    filepath.write_text(text)
    print(f"  {filepath.name}: patched with gold-answer escalation rewards")
    return True


def main():
    project_root = Path(__file__).parent.parent

    print("Applying Phase 0 patches...")
    print()

    # 1. Patch build_tools in runner scripts
    print("1. Patching build_tools():")
    for script_name in ["run_escalation.py", "collect_escalation_trajectories.py"]:
        filepath = project_root / "scripts" / script_name
        if filepath.exists():
            patch_build_tools(filepath)
        else:
            print(f"  {script_name}: NOT FOUND")

    print()

    # 2. Fix SFT gold-answer filtering (0D)
    print("2. Patching SFT gold-answer filtering (0D):")
    sft_path = project_root / "scripts" / "train_worker_sft.py"
    if sft_path.exists():
        patch_sft_gold_filtering(sft_path)
    else:
        print(f"  train_worker_sft.py: NOT FOUND")

    print()

    # 3. Fix escalation gold-answer rewards (0E)
    print("3. Patching escalation gold-answer rewards (0E):")
    esc_path = project_root / "scripts" / "train_escalation_grpo.py"
    if esc_path.exists():
        patch_escalation_gold_reward(esc_path)
    else:
        print(f"  train_escalation_grpo.py: NOT FOUND")

    print()
    print("All patches applied. Run smoke test to verify.")


if __name__ == "__main__":
    main()
