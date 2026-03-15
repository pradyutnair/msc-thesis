import json, sys

path = sys.argv[1] if len(sys.argv) > 1 else "results/pilot_50q/hotpotqa/predictions.jsonl"
with open(path) as f:
    preds = [json.loads(l) for l in f if l.strip()]

for p in preds[:5]:
    q = p["question"][:70]
    pred = p["pred_answer"][:50]
    gold = p["gold_answer"][:50]
    print(f"Q: {q}")
    print(f"  pred='{pred}' | gold='{gold}'")
    print(f"  tokens={p['total_tokens']}, ticks={p['total_ticks']}, term={p['termination_reason']}")
    print(f"  verified={p['verified_count']}, failed={p['failed_count']}")
    er = p.get("entity_registry", {})
    if er:
        for k, v in er.items():
            print(f"    entity: {k} = {str(v)[:60]}")
    for sq in p.get("sub_questions", []):
        ans = str(sq.get("answer", ""))[:60]
        print(f"    SQ-{sq['id']}: [{sq['status']}] '{ans}'")
    print()
