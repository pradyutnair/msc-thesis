import json, sys
path = sys.argv[1] if len(sys.argv) > 1 else "trajectories/pilot_hotpotqa/escalation_trajectories.jsonl"
with open(path) as f:
    trajs = [json.loads(l) for l in f if l.strip()]
print(f"Completed: {len(trajs)} questions")
for t in trajs:
    esc = t["escalation_summary"]
    q = t["question"][:70]
    nsq = t["num_sub_questions"]
    st = t["total_structured_tokens"]
    at = t["total_agentic_tokens"]
    print(f"  Q: {q}...")
    print(f"    SQs={nsq} ACCEPT={esc['ACCEPT']} ESCALATE={esc['ESCALATE']} FAILED={esc['BOTH_FAILED']}  s_tok={st} a_tok={at}")
