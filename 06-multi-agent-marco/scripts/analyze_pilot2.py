import json

with open("results/pilot_50q/hotpotqa/predictions.jsonl") as f:
    preds = [json.loads(l) for l in f if l.strip()]

total_sqs = sum(len(p.get("sub_questions", [])) for p in preds)
verified = sum(p["verified_count"] for p in preds)
failed = sum(p["failed_count"] for p in preds)
empty_pred = sum(1 for p in preds if not p["pred_answer"])

print(f"Total questions: {len(preds)}")
print(f"Total SQs: {total_sqs}")
print(f"Verified: {verified} ({verified/max(total_sqs,1)*100:.0f}%)")
print(f"Failed: {failed} ({failed/max(total_sqs,1)*100:.0f}%)")
print(f"Empty final answers: {empty_pred}")

# Compare v1 vs v2 answers
print("\n=== Answer comparison (first 10) ===")
for p in preds[:10]:
    q = p["question"][:55]
    pred = p["pred_answer"][:40]
    gold = p["gold_answer"][:40]
    v = p["verified_count"]
    f2 = p["failed_count"]
    match = "Y" if pred.lower().strip() == gold.lower().strip() else ""
    print(f"  {match:2s} Q: {q}...")
    print(f"     pred='{pred}' | gold='{gold}' | v={v} f={f2}")

# Check if the _clean_answer >200 char filter is killing things
long_answers = 0
for p in preds:
    for sq in p.get("sub_questions", []):
        ans = sq.get("answer", "")
        if ans and len(ans) > 200:
            long_answers += 1
print(f"\nSQ answers >200 chars: {long_answers}")
print(f"Questions with all SQs failed: {sum(1 for p in preds if p['verified_count'] == 0)}")
