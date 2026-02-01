# FlashRAG Benchmark Results Summary

Both jobs completed. Below are the results:

## Naive Generation (No Retrieval)

| Dataset           | Metric | Our Result | Paper Result | Status  |
|-------------------|--------|------------|--------------|---------|
| NQ                | EM     | 22.0%      | 22.6%        | Close   |
| TriviaQA          | EM     | 55.0%      | 55.7%        | Close   |
| HotpotQA          | F1     | 27.3%      | 28.4%        | Close   |
| 2WikiMultihopQA   | F1     | 29.1%      | 33.9%        | Lower   |
| PopQA             | F1     | 26.9%      | 21.7%        | Better  |
| WebQA             | EM     | 21.0%      | 18.8%        | Better  |

## Standard RAG (Retrieve + Generate)

| Dataset           | Metric | Our Result | Paper Result | Status      |
|-------------------|--------|------------|--------------|-------------|
| NQ                | EM     | 4.1%       | 35.1%        | Much lower  |
| TriviaQA          | EM     | 24.6%      | 58.8%        | Much lower  |
| HotpotQA          | F1     | 11.4%      | 35.3%        | Much lower  |
| 2WikiMultihopQA   | F1     | 11.0%      | 21.0%        | Lower       |
| PopQA             | F1     | 5.4%       | 36.7%        | Much lower  |
| WebQA             | EM     | 5.5%       | 15.7%        | Lower       |

---

## Observations

- **Naive Generation**: Most results align closely with the paper (NQ, TriviaQA, HotpotQA are within ~1–2% of reported values).
- **Standard RAG**: Results are significantly lower than expected, suggesting a problem with retrieval or the index configuration. The index is set to `/projects/prjs1800/datasets/flashrag/indexes/e5_Flat.index`, but retrieval may not be functioning correctly.

## Next Steps

The poor performance of Standard RAG points toward a retrieval issue. Potential causes include:
- The index file might not be properly built or could be incompatible.
- There may be a mismatch in retrieval configuration.
- Evaluation settings could be different from those used in the paper.

All results are saved in `/projects/prjs1800/results/flashrag/`.

- **Summary**: Naive generation matches paper results; standard RAG requires further investigation.