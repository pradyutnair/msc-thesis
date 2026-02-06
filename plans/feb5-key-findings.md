# Day 1 & Day 2: Visual Summary

## Experiment Progression

```mermaid
flowchart LR
    subgraph Day1["Day 1: Standard RAG Baseline"]
        A[Query] --> B["E5-base-v2<br/>Retriever"]
        B -->|top-5| C["Qwen2.5-7B<br/>vLLM"]
        C --> D[Answer]
    end

    subgraph Day2["Day 2: + Cross-Encoder Reranker"]
        E[Query] --> F["E5-base-v2<br/>Retriever"]
        F -->|top-20| G["BGE-reranker<br/>v2-m3"]
        G -->|top-5| H["Qwen2.5-7B<br/>vLLM"]
        H --> I[Answer]
    end

    Day1 -.->|"Error analysis:<br/>retrieval is bottleneck"| Day2
```

## Answer Quality: F1 Scores

```mermaid
xychart-beta
    title "Answer F1 by Method and Dataset"
    x-axis ["HotpotQA<br/>Day 1", "HotpotQA<br/>Day 2", "MuSiQue<br/>Day 1", "MuSiQue<br/>Day 2"]
    y-axis "F1 Score (%)" 0 --> 55
    bar [42.01, 47.42, 13.03, 15.52]
```

## Answer Quality: EM Scores

```mermaid
xychart-beta
    title "Exact Match by Method and Dataset"
    x-axis ["HotpotQA<br/>Day 1", "HotpotQA<br/>Day 2", "MuSiQue<br/>Day 1", "MuSiQue<br/>Day 2"]
    y-axis "EM Score (%)" 0 --> 45
    bar [31.64, 36.41, 6.33, 7.70]
```

## Retrieval Recall@5

```mermaid
xychart-beta
    title "Retrieval Recall@5: Standard RAG vs + Reranker"
    x-axis ["HotpotQA<br/>Day 1", "HotpotQA<br/>Day 2", "MuSiQue<br/>Day 1", "MuSiQue<br/>Day 2"]
    y-axis "Avg Recall@5 (%)" 0 --> 65
    bar [50.0, 57.7, 21.4, 26.2]
```

## MuSiQue Per-Hop Retrieval Recall

```mermaid
xychart-beta
    title "MuSiQue Per-Hop Recall: Day 1 vs Day 2"
    x-axis ["Hop 1", "Hop 2", "Hop 3", "Hop 4"]
    y-axis "Recall (%)" 0 --> 45
    bar "Day 1 (Standard RAG)" [33.5, 11.6, 6.5, 3.2]
    bar "Day 2 (+ Reranker)" [39.6, 14.8, 12.4, 4.7]
```

## HotpotQA Error Categorization Shift

```mermaid
pie title "Day 1: HotpotQA Error Categories"
    "Correct" : 2343
    "Partial Retrieval" : 2698
    "Total Retrieval Miss" : 1577
    "Reasoning Failure" : 787
```

```mermaid
pie title "Day 2: HotpotQA Error Categories (+ Reranker)"
    "Correct" : 2696
    "Partial Retrieval" : 2458
    "Total Retrieval Miss" : 1193
    "Reasoning Failure" : 1058
```

## MuSiQue Error Categorization Shift

```mermaid
pie title "Day 1: MuSiQue Error Categories"
    "Correct" : 153
    "Partial Retrieval" : 910
    "Total Retrieval Miss" : 1302
    "Reasoning Failure" : 52
```

```mermaid
pie title "Day 2: MuSiQue Error Categories (+ Reranker)"
    "Correct" : 186
    "Partial Retrieval" : 1069
    "Total Retrieval Miss" : 1077
    "Reasoning Failure" : 85
```

## Key Findings Flow

```mermaid
flowchart TD
    A["Day 1 Baseline<br/>HotpotQA F1=42.0 | MuSiQue F1=13.0"] --> B{"Where do<br/>failures come from?"}
    B -->|"50% of HotpotQA GT docs<br/>not in top-5"| C["Retrieval is<br/>the bottleneck"]
    B -->|"55.8% of MuSiQue has<br/>ZERO GT docs"| C
    B -->|"Per-hop decay:<br/>33% → 12% → 7% → 3%"| D["Later hops nearly<br/>impossible to retrieve"]

    C --> E["Day 2: Add Reranker<br/>Retrieve top-20, rerank to top-5"]
    D --> E

    E --> F{"Did reranking help?"}
    F -->|"HotpotQA recall<br/>50% → 57.7%"| G["Yes: promotes relevant<br/>docs from rank 6-20"]
    F -->|"MuSiQue recall<br/>21.4% → 26.2%"| H["Partially: later-hop docs<br/>not in top-20 at all"]

    G --> I["HotpotQA F1: 42.0 → 47.4<br/>+5.4 improvement"]
    H --> J["MuSiQue F1: 13.0 → 15.5<br/>+2.5 improvement"]

    I --> K["Next: Day 3 Refiner<br/>Is noise the problem?"]
    J --> L["Next: Day 4 IRCoT<br/>Iterative retrieval for later hops"]
```

## Component Stack Progress

```mermaid
flowchart BT
    subgraph stack["Incremental Component Stack"]
        direction BT
        S1["✅ Standard RAG<br/>HQA F1=42.0 | MSQ F1=13.0"]
        S2["✅ + Reranker<br/>HQA F1=47.4 (+5.4) | MSQ F1=15.5 (+2.5)"]
        S3["⬜ + Refiner (Day 3)"]
        S4["⬜ + Iterative Retrieval (Day 4)"]
        S5["⬜ + Reasoning (Day 5)"]
        S6["⬜ + Multi-Agent (Day 7)"]

        S1 --> S2 --> S3 --> S4 --> S5 --> S6
    end
```
