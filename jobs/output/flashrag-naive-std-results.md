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

## How Naive Generation Works

Naive generation is a baseline method that generates answers **without any retrieval** - it relies solely on the LLM's internal knowledge. This serves as a baseline to measure the impact of retrieval-augmented methods.

### Key Components

**1. NoOpRetriever (Dummy Retriever)**

The naive method uses a `NoOpRetriever` class that returns empty retrieval results for all queries:

```python
class NoOpRetriever:
    """Dummy retriever for naive baseline (no retrieval used)."""

    def batch_search(self, query, num=None, return_score=False):
        qlist = [query] if isinstance(query, str) else query
        if return_score:
            return [[] for _ in qlist], [[] for _ in qlist]
        return [[] for _ in qlist]

    def _save_cache(self):
        pass
```

This ensures no documents are retrieved, forcing the model to answer from its internal knowledge.

**2. Naive Run Pipeline**

The `naive_run()` method in `SequentialPipeline` directly generates answers without retrieval:

```python
def naive_run(self, dataset, do_eval=True, pred_process_fun=None):
    # direct generation without RAG
    input_prompts = [self.prompt_template.get_string(question=q) for q in dataset.question]
    dataset.update_output("prompt", input_prompts)

    pred_answer_list = self.generator.generate(input_prompts)
    dataset.update_output("pred", pred_answer_list)

    dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)
    return dataset
```

**3. Prompt Construction**

The prompt template formats the question without any reference documents. When `retrieval_result=None`, the `formatted_reference` is empty, resulting in a prompt like:

```
Answer the question based on the given document.
Only give me the answer and do not output any other words.

The following are given documents.


Question: [question]
```

Since there are no documents provided, the model must rely entirely on its pre-trained knowledge to answer the question.

**4. Generation Process**

The LLM (Meta-Llama-3-8B-Instruct) generates answers directly from the question prompt using its internal knowledge, without any external context from a retrieval corpus.

### Summary

Naive generation:
1. Uses `NoOpRetriever` to return empty retrieval results
2. Constructs prompts with only the question (no context documents)
3. Generates answers using the LLM's internal knowledge via `generator.generate()`
4. Evaluates the outputs against ground truth

This baseline helps quantify how much retrieval improves performance compared to relying solely on the model's parametric knowledge.

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