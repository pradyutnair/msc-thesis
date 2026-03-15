import json
with open("trajectories/pilot_hotpotqa/escalation_trajectories.jsonl") as f:
    trajs = [json.loads(l) for l in f if l.strip()]

for t in trajs[:3]:
    print("Q:", t["question"][:80])
    for sq in t["sub_question_results"]:
        if sq.get("skipped"):
            continue
        s = sq.get("structured", {})
        a = sq.get("agentic", {})
        s_ans = s.get("answer", "")
        a_ans = a.get("answer", "")
        s_ev = s.get("evidence_count", 0)
        a_ev = a.get("evidence_count", 0)
        s_tok = s.get("tokens", 0)
        a_tok = a.get("tokens", 0)
        s_err = s.get("error", "")
        a_err = a.get("error", "")
        s_status = s.get("status", "")
        a_status = a.get("status", "")
        print(f"  SQ-{sq['sq_id']}: {sq['resolved_text'][:70]}")
        print(f"    Structured: ans='{s_ans}', ev={s_ev}, tok={s_tok}, status={s_status}, err={s_err}")
        print(f"    Agentic:    ans='{a_ans}', ev={a_ev}, tok={a_tok}, status={a_status}, err={a_err}")
    print()
