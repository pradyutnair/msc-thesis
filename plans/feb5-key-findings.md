# Incremental RAG Component Analysis: Visual Summary

## Experiment Progression (Days 1-3)

```mermaid
flowchart LR
    subgraph Day1["Day 1: Standard RAG Baseline"]
        A[Query] --> B["E5-base-v2\nRetriever"]
        B -->|top-5| C["Qwen2.5-7B\nvLLM"]
        C --> D[Answer]
    end

    subgraph Day2["Day 2: + Cross-Encoder Reranker"]
        E[Query] --> F["E5-base-v2\nRetriever"]
        F -->|top-20| G["BGE-reranker\nv2-m3"]
        G -->|top-5| H["Qwen2.5-7B\nvLLM"]
        H --> I[Answer]
    end

    subgraph Day3a["Day 3a: + RECOMP Refiner"]
        J[Query] --> K["E5-base-v2\nRetriever"]
        K -->|top-5| L["RECOMP T5\nAbstractive"]
        L -->|"summary\n(8.8% retained)"| M["Qwen2.5-7B\nvLLM"]
        M --> N[Answer]
    end

    subgraph Day3b["Day 3b: + Selective-Context"]
        O[Query] --> P["E5-base-v2\nRetriever"]
        P -->|top-5| Q["GPT-2\nPerplexity Filter"]
        Q -->|"filtered\n(67% retained)"| R["Qwen2.5-7B\nvLLM"]
        R --> S[Answer]
    end

    Day1 -.->|"Error analysis:\nretrieval is bottleneck"| Day2
    Day2 -.->|"Is noise\nthe problem?"| Day3a
    Day2 -.->|"Is noise\nthe problem?"| Day3b
```

## HotpotQA F1: All Methods

```mermaid
xychart-beta
    title "HotpotQA F1 Across All Methods"
    x-axis ["Standard RAG\n(Day 1)", "+ Reranker\n(Day 2)", "+ RECOMP\n(Day 3)", "+ Selective-Ctx\n(Day 3)"]
    y-axis "F1 Score (%)" 0 --> 55
    bar [42.01, 47.42, 40.02, 36.56]
```

## MuSiQue F1: All Methods

```mermaid
xychart-beta
    title "MuSiQue F1 Across All Methods"
    x-axis ["Standard RAG\n(Day 1)", "+ Reranker\n(Day 2)", "+ RECOMP\n(Day 3)", "+ Selective-Ctx\n(Day 3)"]
    y-axis "F1 Score (%)" 0 --> 20
    bar [13.03, 15.52, 11.85, 11.26]
```

## F1 Delta from Baseline (Day 1)

```mermaid
xychart-beta
    title "F1 Change vs Standard RAG Baseline"
    x-axis ["Reranker\nHQA", "Reranker\nMSQ", "RECOMP\nHQA", "RECOMP\nMSQ", "SC\nHQA", "SC\nMSQ"]
    y-axis "Delta F1" -6 --> 6
    bar [5.41, 2.49, -1.99, -1.18, -5.45, -1.77]
```

## Compression Ratio vs F1 Impact

```mermaid
xychart-beta
    title "Day 3: More Compression = Worse Performance"
    x-axis ["SC HotpotQA\n(67% kept)", "SC MuSiQue\n(67% kept)", "RECOMP HotpotQA\n(8.8% kept)", "RECOMP MuSiQue\n(9.7% kept)"]
    y-axis "F1 Score (%)" 0 --> 45
    bar [36.56, 11.26, 40.02, 11.85]
```

## Retrieval Recall@5

```mermaid
xychart-beta
    title "Retrieval Recall@5: Standard RAG vs + Reranker"
    x-axis ["HotpotQA\nDay 1", "HotpotQA\nDay 2", "MuSiQue\nDay 1", "MuSiQue\nDay 2"]
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

## Key Findings Flow (Days 1-3)

```mermaid
flowchart TD
    A["Day 1 Baseline\nHotpotQA F1=42.0 | MuSiQue F1=13.0"] --> B{"Where do\nfailures come from?"}
    B -->|"50% of HotpotQA GT docs\nnot in top-5"| C["Retrieval is\nthe bottleneck"]
    B -->|"55.8% of MuSiQue has\nZERO GT docs"| C
    B -->|"Per-hop decay:\n33% → 12% → 7% → 3%"| D["Later hops nearly\nimpossible to retrieve"]

    C --> E["Day 2: Add Reranker\nRetrieve top-20, rerank to top-5"]
    D --> E

    E --> F{"Did reranking help?"}
    F -->|"HotpotQA F1 +5.4\nrecall 50% → 57.7%"| G["YES: promotes relevant\ndocs from rank 6-20"]
    F -->|"MuSiQue F1 +2.5\nrecall 21.4% → 26.2%"| H["PARTIALLY: later-hop docs\nnot in top-20 at all"]

    G --> I{"Is noise the\nproblem then?"}
    H --> I

    I --> J["Day 3: Add Refiners\nRECOMP + Selective-Context"]

    J --> K{"Did refining help?"}
    K -->|"RECOMP: -2.0 F1 HQA\n-1.2 F1 MSQ"| L["NO: abstractive summary\nloses critical facts"]
    K -->|"SC: -5.4 F1 HQA\n-1.8 F1 MSQ"| M["NO: perplexity filter\nremoves useful signal"]

    L --> N["CONCLUSION:\nNoise is NOT the bottleneck.\nMISSING information is."]
    M --> N

    N --> O["Next: Day 4 IRCoT\nIterative retrieval to find\nmissing later-hop docs"]
```

## Component Stack Progress

```mermaid
flowchart BT
    subgraph stack["Incremental Component Stack"]
        direction BT
        S1["✅ Standard RAG\nHQA F1=42.0 | MSQ F1=13.0"]
        S2["✅ + Reranker (+5.4 / +2.5)\nHQA F1=47.4 | MSQ F1=15.5"]
        S3["✅ + Refiner (NEGATIVE)\nRECOMP: -2.0 / -1.2 | SC: -5.4 / -1.8"]
        S4["⬜ + Iterative Retrieval (Day 4)"]
        S5["⬜ + Reasoning (Day 5)"]
        S6["⬜ + Multi-Agent (Day 7)"]

        S1 --> S2 --> S3 --> S4 --> S5 --> S6
    end

    style S3 fill:#ff6b6b,color:#fff
```

## Day 3: Refining Time Breakdown

```mermaid
xychart-beta
    title "Day 3: Time Breakdown (seconds)"
    x-axis ["RECOMP\nHotpotQA", "RECOMP\nMuSiQue", "SC\nHotpotQA", "SC\nMuSiQue"]
    y-axis "Time (seconds)" 0 --> 10500
    bar "Retrieval" [212, 175, 229, 175]
    bar "Refining" [9957, 3456, 2717, 868]
    bar "Generation" [48, 17, 228, 73]
```
