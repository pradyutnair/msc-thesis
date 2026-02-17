# A-RAG Reproduction: All Experiments Comparison

## Overview

This directory contains results from four experimental configurations reproducing the A-RAG agentic retrieval pipeline, benchmarked against the original paper (GPT-4o-mini).

All experiments use 1000 questions per dataset across HotpotQA, MuSiQue, and 2WikiMultihop.

## Experiment Configurations

<table>
  <thead>
    <tr>
      <th>ID</th>
      <th>Generator</th>
      <th>Embedding</th>
      <th>Judge</th>
      <th>Index Source</th>
      <th>Hardware</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>E1</b></td>
      <td>Qwen2.5-7B-Instruct</td>
      <td>E5-base-v2</td>
      <td>N/A (F1/EM only)</td>
      <td>FlashRAG wiki18 FAISS flat</td>
      <td>A100 40GB</td>
    </tr>
    <tr>
      <td><b>E2</b></td>
      <td>Qwen3-8B</td>
      <td>E5-base-v2</td>
      <td>Qwen3-30B-A3B</td>
      <td>ARAG chunked index</td>
      <td>A100 (gen), H100 (eval)</td>
    </tr>
    <tr>
      <td><b>E3</b></td>
      <td>Qwen3-8B</td>
      <td>Qwen3-Embedding-0.6B</td>
      <td>Qwen3-30B-A3B</td>
      <td>ARAG chunked index</td>
      <td>A100 (gen), H100 (eval)</td>
    </tr>
    <tr>
      <td><b>E4</b></td>
      <td>Qwen3-30B-A3B</td>
      <td>E5-base-v2</td>
      <td>DeepSeek-R1-Distill-Qwen-32B</td>
      <td>ARAG chunked index</td>
      <td>H100 (gen + eval)</td>
    </tr>
    <tr>
      <td><b>Ref</b></td>
      <td>GPT-4o-mini</td>
      <td>(paper default)</td>
      <td>GPT-4o-mini</td>
      <td>(paper default)</td>
      <td>API</td>
    </tr>
  </tbody>
</table>

### Config Details

| ID    | Agent max_loops | Token budget | Temperature | max_tokens | Workers |
|-------|:--------------:|:------------:|:-----------:|:----------:|:-------:|
| **E1** |    10          |   128k       |    0.0      |   1024     |    4    |
| **E2** |    15          |   128k       |    0.0      |   8192     |    4    |
| **E3** |    15          |   128k       |    0.0      |   8192     |    4    |
| **E4** |    15          |   128k       |    0.0      |   8192     |    4    |

**Config files:**
- E1: `configs/arag_qwen25_template.yaml`
- E2: `configs/arag_qwen3_vllm_e5_{dataset}.yaml`
- E3: `configs/arag_qwen3_vllm_qwenemb_{dataset}.yaml`
- E4: `configs/arag_qwen3_30b_vllm_e5_{dataset}.yaml`

## Main Results: LLM-Accuracy (%)

| Dataset        | E1 (Q2.5-7B) | E2 (Q3-8B+E5) | E3 (Q3-8B+QEmb) | E4 (Q3-30B+E5) | Paper (GPT-4o-mini) |
|:--------------:|:------------:|:-------------:|:--------------:|:--------------:|:-------------------:|
| **HotpotQA**   | --           | 53.2          | 47.5           | **66.5**       | 77.1                |
| **MuSiQue**    | --           | 32.0          | 28.6           | **37.6**       | 46.1                |
| **2WikiMultihop**| --         | 43.1          | 36.3           | **56.9**       | 60.2                |
| **Mean**       | --           | 42.8          | 37.5           | **53.7**       | 61.1                |

## Main Results: Contain-Accuracy (%)

| Dataset        | E1 (Q2.5-7B) | E2 (Q3-8B+E5) | E3 (Q3-8B+QEmb) | E4 (Q3-30B+E5) | Paper (GPT-4o-mini) |
|:--------------:|:------------:|:-------------:|:--------------:|:--------------:|:-------------------:|
| **HotpotQA**   | 29.5         | 62.2          | 59.0           | **67.7**       | 74.0                |
| **MuSiQue**    | 5.8          | 29.8          | 24.6           | **34.4**       | 39.6                |
| **2WikiMultihop**| 15.3        | 57.1          | 52.2           | **63.9**       | 63.7                |
| **Mean**       | 16.9         | 49.7          | 45.3           | **55.3**       | 59.1                |

## Agent Behavior Metrics

| Dataset      | Metric             | E1 (Q2.5-7B) | E2 (Q3-8B+E5) | E3 (Q3-8B+QEmb) | E4 (Q3-30B+E5) |
|:------------:|:------------------:|:------------:|:-------------:|:--------------:|:--------------:|
| **HotpotQA** | Avg Loops          | 4.54         | 2.44          | 2.53           | 2.66           |
|              | Avg Retr. Tokens   | --           | 714           | 783            | 842            |
|              | Answer Rate        | 99.4%        | 100%          | 100%           | 100%           |
|              | Cost/question      | --           | bash.0115     | bash.0117      | bash.0120      |
| **MuSiQue**  | Avg Loops          | 4.91         | 2.65          | 2.63           | 2.98           |
|              | Avg Retr. Tokens   | --           | 751           | 804            | 873            |
|              | Answer Rate        | 98.6%        | 100%          | 100%           | 100%           |
|              | Cost/question      | --           | bash.0143     | bash.0142      | bash.0147      |
| **2WikiMultihop** | Avg Loops     | 4.89         | 2.78          | 2.84           | 3.05           |
|              | Avg Retr. Tokens   | --           | 811           | 935            | 800            |
|              | Answer Rate        | 99.4%        | 100%          | 100%           | 100%           |
|              | Cost/question      | --           | bash.0121     | bash.0123      | bash.0128      |

## Progression Summary

### Improvement from E1 to E4 (Contain-Accuracy)

| Dataset         | E1 → E2 | E2 → E3 | E2 → E4       | E4 vs Paper  |
|:---------------:|:-------:|:-------:|:-------------:|:------------:|
| **HotpotQA**    | +32.7pp | -3.2pp  | **+5.5pp**    | -6.3pp       |
| **MuSiQue**     | +24.0pp | -5.2pp  | **+4.6pp**    | -5.2pp       |
| **2WikiMultihop**| +41.8pp| -4.9pp  | **+6.8pp**    | **+0.2pp**   |

## Key Findings

**1. Generator quality is the dominant factor**

Upgrading from Qwen3-8B to Qwen3-30B-A3B (same embeddings) yielded the largest single improvement: +10.9pp mean LLM-Acc, +5.6pp mean Contain-Acc.

**2. E5-base-v2 > Qwen3-Embedding-0.6B for retrieval**

Switching from E5 to Qwen-Embedding consistently degraded results (-4.4pp mean Contain-Acc), suggesting E5-base-v2 is the better retriever for this task.

**3. ARAG chunked indices >> FlashRAG wiki18**

E1 used the FlashRAG 21M-passage wiki corpus. E2–E4 used ARAG per-dataset chunk indices. This alone explains most of the E1 → E2 jump (+32.8pp mean Contain-Acc), as task-specific chunking dramatically improves retrieval recall.

**4. Near paper-level on 2WikiMultihop**

E4 matches the paper on 2WikiMultihop (63.9% vs 63.7% Contain-Acc). The remaining gap is concentrated on HotpotQA (-6.3pp) and MuSiQue (-5.2pp).

**5. Independent judge validation**

E4 uses DeepSeek-R1-Distill-Qwen-32B (different model family) as judge, eliminating the self-evaluation bias present in E2/E3 where Qwen3-30B-A3B was both generator (in E4) and judge.

## Directory Structure

<pre>
results/
├── README.md                           (this file)
├── qwen25-7b-instruct/                 (E1)
│   ├── hotpotqa/eval.json
│   ├── musique/eval.json
│   └── 2wikimultihopqa/eval.json
├── qwen3-8b-vllm/                      (E2)
│   ├── hotpotqa/predictions_eval_summary.json
│   ├── musique/predictions_eval_summary.json
│   └── 2wikimultihop/predictions_eval_summary.json
├── qwen3-8b-qwen-emb-vllm/             (E3)
│   ├── hotpotqa/predictions_eval_summary.json
│   ├── musique/predictions_eval_summary.json
│   └── 2wikimultihop/predictions_eval_summary.json
└── qwen3-30b-e5-deepseekr1/            (E4)
    ├── hotpotqa/predictions_eval_summary.json
    ├── musique/predictions_eval_summary.json
    └── 2wikimultihop/predictions_eval_summary.json
</pre>
