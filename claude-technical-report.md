# Multi-Agent RAG Research: Comprehensive Final Report
**Master's Thesis Reference Document**

**Author:** Pradyut Nair  
**Date:** January 25, 2026  
**Topic:** Retrieval-Augmented Generation with Multi-Agent Collaborative Search

---

# Table of Contents
1. [Minimal Requirements for RAG: Theoretical Foundation](#1-minimal-requirements-for-rag-theoretical-foundation)
2. [Benchmark Performance Comparison](#2-benchmark-performance-comparison)
3. [LLM Model Analysis for Implementation](#3-llm-model-analysis-for-implementation)
4. [Dataset and Benchmark Coverage Analysis](#4-dataset-and-benchmark-coverage-analysis)
5. [Recommended Implementation Stack](#5-recommended-implementation-stack)

---

# 1. Minimal Requirements for RAG: Theoretical Foundation

## 1.1 Core RAG Components (Essential for Any RAG System)

For your thesis on Multi-Agent Collaborative Search in RAG, these are the **absolutely essential** theoretical components:

### A. Knowledge Base & Retrieval System
**What:** External knowledge source that agents query to ground their responses.

**Minimal Requirements:**
- **Document Corpus:** Collection of text documents (e.g., Wikipedia passages)
- **Indexing Method:** Vector database OR keyword search (minimum: one retrieval method)
- **Retrieval Function:** `retrieve(query) â†’ top_k documents`

**For Multi-Agent RAG:**
- Supports concurrent queries from multiple agents
- Can handle reformulated/decomposed queries
- Returns ranked results with relevance scores

**Examples from Papers:**
- MA-RAG: Wikipedia corpus with semantic retrieval
- CoRAG: KILT Wikipedia (~36M passages)
- MMOA-RAG: Tests with Contriever, BGE, E5 retrievers

---

### B. Large Language Model (LLM)
**What:** Core reasoning engine that processes queries, retrieved context, and generates answers.

**Minimal Requirements:**
- **Input:** Question + Retrieved Documents
- **Output:** Answer (text generation)
- **Context Window:** Sufficient to hold question + top-k documents (minimum 4K tokens)

**For Multi-Agent RAG:**
- Supports agent-specific prompting/instructions
- Can perform intermediate reasoning steps
- Handles iterative/multi-turn interactions

**Examples from Papers:**
- Training-free: GPT-4o-mini (MA-RAG), Llama-3-8B (MAIN-RAG)
- Fine-tuned: Llama-3.1-8B (CoRAG), Qwen2.5-7B (DecEx-RAG)

---

### C. Multi-Agent Coordination Mechanism
**What:** Framework for orchestrating multiple agents with specialized roles.

**Minimal Requirements (Choose ONE):**
1. **Sequential Pipeline:** Agent A â†’ Agent B â†’ Agent C (simplest)
2. **Parallel Execution:** Multiple agents work simultaneously, results merged
3. **Iterative Feedback:** Agents communicate results back and forth

**For Your Three Research Questions:**
- **Q1 (Parallelism):** Need parallel execution or asynchronous task delegation
- **Q2 (Collective Intelligence):** Need inter-agent communication (voting, consensus, debate)
- **Q3 (Scaling Laws):** Need variable compute allocation (more agents = more compute)

**Examples from Papers:**
- Sequential: MA-RAG (Planner â†’ Step Definer â†’ Extractor â†’ QA)
- Parallel: HM-RAG (Multi-source retrieval agents work in parallel)
- Iterative: CoRAG (iterative retrieval chains with refinement)
- RL-based: MMOA-RAG (cooperative multi-agent RL)

---

## 1.2 Multi-Agent RAG Specific Requirements

### D. Agent Specialization
**What:** Each agent has a distinct role in the RAG pipeline.

**Minimal Agent Roles (Choose 2-4):**
1. **Query Decomposition/Planning:** Breaks complex queries into sub-questions
2. **Retrieval/Search:** Executes search and returns documents
3. **Filtering/Ranking:** Judges document relevance, filters noise
4. **Answer Generation/Synthesis:** Produces final answer from retrieved context

**Not Required But Beneficial:**
- Query Rewriting agent
- Verification/Critic agent
- Memory/Context management agent

**Examples from Papers:**
- MA-RAG: 4 agents (Planner, Step Definer, Extractor, QA)
- MAIN-RAG: 3 agents (Predictor, Judge, Final-Predictor)
- HM-RAG: 3-tier hierarchy (Decomposer, Retrievers, Decision)

---

### E. Communication Protocol
**What:** How agents share information and coordinate.

**Minimal Requirements:**
- **State Representation:** Current query state, retrieved documents, intermediate results
- **Message Passing:** Agent outputs become inputs for other agents
- **Termination Condition:** When to stop iterating and return final answer

**Options:**
1. **Shared Memory:** All agents read/write to common state
2. **Message Queue:** Agents send results to next agent in pipeline
3. **Central Coordinator:** Manager agent dispatches tasks to workers

**Examples from Papers:**
- MA-RAG: Chain-of-thought prompts passed between agents
- MMOA-RAG: Cooperative RL with shared reward signal
- HM-RAG: Hierarchical communication (top â†’ middle â†’ bottom tiers)

---

## 1.3 Training vs. Training-Free Approaches

### Training-Free (Recommended for Initial Thesis Work)
**Requirements:**
- Pre-trained LLM (no fine-tuning needed)
- Well-designed prompts for each agent role
- Retrieval system (pre-built index)

**Advantages:**
- Faster to implement
- No labeled training data required
- Easier to interpret and debug

**Papers Using This:** MA-RAG, MAIN-RAG, HM-RAG

---

### Training-Based (Advanced)
**Additional Requirements:**
- Training data: (Question, Documents, Answer) triplets
- Training method: SFT, DPO, RL, or rejection sampling
- Compute: GPUs for fine-tuning (8Ã— A100s for MMOA-RAG)

**Advantages:**
- Higher performance ceiling
- Can optimize end-to-end
- Learns optimal agent coordination

**Papers Using This:** CoRAG (SFT + sampling), DecEx-RAG (SFT + DPO), MMOA-RAG (Multi-agent RL), TeaRAG (IP-DPO)

---

## 1.4 Evaluation Framework (Essential for Thesis)

### Minimal Metrics
You **must** report these for any RAG system:

1. **Answer Quality:**
   - **Exact Match (EM):** Binary match with ground truth
   - **F1 Score:** Token-level overlap with ground truth
   
2. **Retrieval Quality:**
   - **Recall@k:** Are relevant documents in top-k?
   - **Precision@k:** What fraction of top-k are relevant?

3. **Efficiency (for Q1 and Q3):**
   - **Latency:** Time to generate answer
   - **Token Usage:** Total tokens consumed
   - **API Calls:** Number of LLM/retrieval calls

### Multi-Agent Specific Metrics
4. **Agent Contribution:**
   - Ablation studies: Remove each agent, measure performance drop
   - Inter-agent agreement: How often agents agree on decisions

5. **Scaling Metrics (for Q3):**
   - Performance vs. number of agents
   - Performance vs. retrieval rounds
   - Diminishing returns analysis

---

## 1.5 Minimal Theoretical Concepts for Your Thesis

Based on your research questions, you **must** understand:

### For Q1: Multi-Agent Collaborative Search for Parallelism
**Core Concepts:**
- **Parallel Retrieval:** Multiple agents query simultaneously
- **Task Decomposition:** Break complex queries into parallelizable sub-tasks
- **Result Aggregation:** Combine outputs from parallel agents (voting, ranking, fusion)

**Minimal Implementation:**
- 2+ retrieval agents working in parallel
- Mechanism to combine their results
- Comparison: Parallel vs. Sequential execution time

---

### For Q2: Collaboration Strategies for Collective Intelligence
**Core Concepts:**
- **Agent Communication:** How agents share intermediate findings
- **Consensus Mechanisms:** Voting, averaging, debate
- **Specialization vs. Redundancy:** Diverse agents vs. redundant agents

**Minimal Implementation:**
- 3+ agents with different roles or strategies
- Inter-agent communication mechanism
- Measure: Does collaboration > sum of individual agents?

---

### For Q3: Inference-Time Scaling Laws
**Core Concepts:**
- **Test-Time Compute:** Allocate more compute â†’ better performance
- **Diminishing Returns:** Performance plateau as agents/iterations increase
- **Scaling Functions:** Model relationship: `Performance = f(Compute)`

**Minimal Implementation:**
- Variable compute allocation (1, 2, 4, 8 agents OR 1, 2, 4, 8 retrieval rounds)
- Plot: Performance vs. Compute
- Identify: Optimal compute allocation for given query difficulty

---

## 1.6 What You DON'T Need (Scope Reduction)

To focus your thesis, you can **exclude** these:

âŒ **Multimodal RAG:** Stick to text-only (exclude HM-RAG approach)  
âŒ **Graph-Based Retrieval:** Use standard vector/keyword search (TeaRAG's knowledge graphs are optional)  
âŒ **Reinforcement Learning:** Training-free approaches are sufficient initially  
âŒ **End-to-End Training:** Fine-tuning entire pipeline adds complexity  
âŒ **Production Deployment:** Focus on research prototype, not scalable service  

---

## 1.7 Minimal Component Checklist

Before starting implementation, ensure you have:

**Data & Resources:**
- [ ] Multi-hop QA dataset (MuSiQue, HotpotQA, or 2WikiMultiHopQA)
- [ ] Document corpus (Wikipedia dump or pre-indexed)
- [ ] Pre-trained LLM access (API or local deployment)

**Core Components:**
- [ ] Retrieval system (vector DB or BM25 search)
- [ ] LLM inference pipeline
- [ ] Multi-agent coordination framework (LangGraph, CrewAI, or custom)

**Agent Definitions:**
- [ ] 2-4 specialized agents with clear roles
- [ ] Prompts/instructions for each agent
- [ ] Communication protocol between agents

**Evaluation:**
- [ ] Evaluation script (EM, F1 calculation)
- [ ] Baseline: Single-agent RAG for comparison
- [ ] Metrics aligned with research questions (latency for Q1, collaboration gain for Q2, scaling curve for Q3)

---

# 2. Benchmark Performance Comparison

## 2.1 Complete Benchmark Table

**Note**: Scores reported as EM / F1 where both available. "â€”" indicates benchmark not evaluated by paper. Best scores per benchmark in **bold**.

| Paper | Model | Training | HotpotQA (EM/F1) | 2WikiMultiHopQA (EM/F1) | NQ (EM) | TriviaQA (EM) | PopQA (Acc) | MuSiQue (EM/F1) | Bamboogle (EM/F1) |
|-------|-------|----------|------------------|-------------------------|---------|---------------|-------------|-----------------|-------------------|
| **CoRAG** | Llama-3.1-8B (fine-tuned) | SFT + best-of-N sampling (L=10, N=8) | **56.3 / 69.8** | **72.5 / 77.3** | **63.1** | **88.3** | â€” | **30.9 / 42.4** | **54.4 / 68.3** |
| **MA-RAG** | GPT-4o-mini | Training-free | 52.1 / â€” | 47.5 / â€” | 59.5 | 87.2 | â€” | â€” | â€” |
| **DecEx-RAG** | Qwen2.5-7B-Instruct | SFT + DPO | 37.7 / 49.3 | 50.0 / 55.9 | 36.0 / 47.2 | â€” | 51.3 / 53.2 | â€” | 37.6 / 49.3 |
| **MMOA-RAG** | Llama-3-8B-Instruct (E5 retriever) | Multi-agent RL | 43.7 / 57.2 | 47.5 / 53.1 | â€” | â€” | â€” | â€” | â€” |
| **MAIN-RAG** | Llama-3-8B | Training-free (adaptive filtering) | â€” | â€” | â€” | 74.1* | **64.0*** | â€” | â€” |
| **TeaRAG** | Llama3-8B / Qwen2.5-14B | IP-DPO (Iterative Process-aware DPO) | ~41.7â€  / â€” | ~54.0â€  / â€” | ~40.0â€  / â€” | â€” | ~55.3â€  / â€” | ~34.9â€  / â€” | ~41.6â€  / â€” |
| **HM-RAG** | GPT-4 (multimodal) | Training-free | â€” | â€” | â€” | â€” | â€” | â€” | â€” |

**Legend:**
- **Bold numbers**: Best score for that benchmark
- **â€”**: Benchmark not evaluated by paper
- **\***: MAIN-RAG reports Accuracy instead of EM
- **â€ **: TeaRAG exact scores estimated from baseline + reported improvements (~4% EM for Llama3-8B, ~2% for Qwen2.5-14B)
- Scores reported as EM / F1 where both metrics available

---

## 2.2 Benchmark-Specific Analysis

### HotpotQA
- **Best**: CoRAG (56.3 EM / 69.8 F1)
- **Note**: Different papers use different settings (KILT version, open-domain, fullwiki)
- MA-RAG uses KILT benchmark (5,600 dev questions)
- CoRAG uses open-domain setting with Wikipedia corpus
- **Difficulty**: Medium-Hard (2-hop questions)
- **Why Important**: Tests explicit multi-hop reasoning with supporting facts

### 2WikiMultiHopQA
- **Best**: CoRAG (72.5 EM / 77.3 F1)
- DecEx-RAG achieves 50.0 EM with smaller model (Qwen2.5-7B)
- MMOA-RAG reports 47.5 EM with multi-agent RL
- **Difficulty**: Medium-Hard (2-4 hops)
- **Why Important**: Explicit reasoning chains, evidence required

### Natural Questions (NQ)
- **Best**: CoRAG (63.1 EM) on KILT test set
- MA-RAG achieves 59.5 EM with GPT-4o-mini (training-free)
- DecEx-RAG only achieves 36.0 EM but uses much smaller model
- **Difficulty**: Medium (single-hop dominant)
- **Why Important**: Real-world open-domain QA

### TriviaQA
- **Best**: CoRAG (88.3 EM) on KILT test set
- MA-RAG achieves 87.2 EM with GPT-4o-mini (competitive, training-free)
- MAIN-RAG reports 74.1% accuracy (different metric, unfiltered version)
- **Difficulty**: Medium-Hard (1-2 hops)
- **Why Important**: Distant supervision, cross-document reasoning

### PopQA
- **Best**: MAIN-RAG (64.0% Accuracy on long-tail subset)
- DecEx-RAG reports 51.3 EM / 53.2 F1
- **Difficulty**: Variable (long-tail entities)
- **Why Important**: Tests adaptive retrieval and noise filtering

### MuSiQue â­ (RECOMMENDED PRIMARY DATASET)
- **Best**: CoRAG (30.9 EM / 42.4 F1)
- Only evaluated by CoRAG and TeaRAG among analyzed papers
- **Difficulty**: HARD (2-4 hops, anti-shortcut design)
- **Why Important**: Specifically designed to prevent single-hop shortcuts, tests TRUE multi-hop reasoning

### Bamboogle â­ (RECOMMENDED SECONDARY DATASET)
- **Best**: CoRAG (54.4 EM / 68.3 F1)
- DecEx-RAG achieves 37.6 EM / 49.3 F1
- **Difficulty**: Adversarial (compositional reasoning)
- **Why Important**: Stress test for collaborative search benefits

---

## 2.3 Key Methodological Differences

### Retrieval Corpora:
- **CoRAG**: KILT Wikipedia (~36M passages)
- **MA-RAG**: Karpukhin et al.'s preprocessed Wikipedia
- **DecEx-RAG**: 2018 Wikipedia dump
- **MAIN-RAG**: Contriever-MS MARCO (up to 20 docs/query)
- **MMOA-RAG**: Tests three retrievers (Contriever, BGE, E5)

### Base Model Capabilities:
- **GPT-4o-mini** (MA-RAG): Proprietary, strongest reasoning
- **Llama-3.1-8B** (CoRAG): Fine-tuned with best-of-N sampling
- **Qwen2.5-7B** (DecEx-RAG): Smallest model tested
- **Llama-3-8B** (MAIN-RAG, MMOA-RAG): Open-source baseline

### Training Approaches:
- **Training-free**: MA-RAG, MAIN-RAG, HM-RAG
- **Fine-tuning**: CoRAG (SFT + sampling), DecEx-RAG (SFT + DPO), TeaRAG (IP-DPO)
- **Reinforcement Learning**: MMOA-RAG (multi-agent RL)

---

## 2.4 Benchmark Coverage Analysis

| Benchmark | Papers Reporting | Difficulty | Importance for Multi-Agent RAG |
|-----------|------------------|------------|-------------------------------|
| **HotpotQA** | 5 of 7 | Medium-Hard | High (tests coordination) |
| **2WikiMultiHopQA** | 5 of 7 | Medium-Hard | High (explicit reasoning chains) |
| **Natural Questions** | 4 of 7 | Medium | Medium (single-hop dominant) |
| **TriviaQA** | 3 of 7 | Medium-Hard | Medium (useful for comparison) |
| **PopQA** | 3 of 7 | Variable | Low (noise filtering focus) |
| **MuSiQue** | 2 of 7 | **HARD** | **Very High** (anti-shortcut) |
| **Bamboogle** | 3 of 7 | **Adversarial** | **Very High** (compositional) |

**Key Insight**: Focus your thesis evaluation on **MuSiQue** and **Bamboogle**, as these are the hardest benchmarks that truly test multi-agent collaborative search benefits. HotpotQA and 2WikiMultiHopQA provide good comparability with existing work.

---

## 2.5 Recommended Baseline Comparisons for Thesis

1. **Primary Baseline**: Compare against **CoRAG** on 2WikiMultiHopQA (72.5 EM) and MuSiQue (30.9 EM)
2. **Secondary Baseline**: Compare against **MA-RAG** on HotpotQA (52.1 EM) and NQ (59.5 EM)
3. **Efficiency Baseline**: Compare token usage against **TeaRAG**'s reported 61% reduction

**Evaluation Strategy:**
- **Primary dataset**: MuSiQue (tests true multi-hop reasoning, avoids shortcuts)
- **Secondary datasets**: HotpotQA + 2WikiMultiHopQA (established benchmarks for comparability)
- **Report both EM and F1**: Follow DecEx-RAG and CoRAG's lead
- **Specify retrieval corpus and model clearly**: Critical for reproducibility

---

# 3. LLM Model Analysis for Implementation

## 3.1 Current Generation Open-Source LLMs (January 2025)

### Tier 1: Frontier Reasoning Models â­ RECOMMENDED

| Model | Sizes | Context | VRAM (Q4) | vLLM | Reasoning | Best For | License | Released |
|-------|-------|---------|-----------|------|-----------|----------|---------|----------|
| **DeepSeek-R1-Distill-Qwen-32B** â­â­ | 32B | 128K | ~20GB | âœ“ Native | Matches o1-mini | **RECOMMENDED: Best distilled reasoning** | Apache 2.0 | Jan 2025 |
| **DeepSeek-R1** | 671B (37B active) | 128K | Multi-node | âœ“ | AIME 96th percentile | Best reasoning/cost; pure RL | MIT | Jan 2025 |
| **DeepSeek-V3.2** | 671B (37B active) | 128K | Multi-node | âœ“ (SGLang) | GPT-4o level | Production reasoning + tools | MIT | Dec 2024 |
| **Qwen3-235B-A22B-Thinking** | 235B (22B active) | 256K-1M | Multi-node | âœ“ Native | AIME 100% | Multilingual reasoning (119 langs) | Apache 2.0 | Apr 2025 |
| **Qwen3-30B-A3B** â­ | 30B (3B active) | 128K | ~8GB | âœ“ Native | Rivals QwQ-32B | Consumer GPU reasoning | Apache 2.0 | Apr 2025 |
| **Phi-4** â­ | 14B | 32K | ~10GB | âœ“ | AIME 77.7% | **Best reasoning/param ratio** | MIT | Dec 2024 |

---

### Tier 2: Balanced General-Purpose Models

| Model | Sizes | Context | VRAM (Q4) | vLLM | Best For | License | Released |
|-------|-------|---------|-----------|------|----------|---------|----------|
| **Llama 4 Maverick** | 400B (17B active) | 1M | Multi-GPU | âœ“ Native | Multimodal; long context | Llama 4 | Apr 2025 |
| **Llama 4 Scout** | 109B (17B active) | 10M | ~40GB (int4) | âœ“ Native | **Extreme context** (10M tokens) | Llama 4 | Apr 2025 |
| **Qwen3-32B-Instruct** | 32B | 128K | ~20GB | âœ“ Native | Multilingual; tool use | Apache 2.0 | Apr 2025 |
| **Qwen3-14B-Instruct** | 14B | 128K | ~10GB | âœ“ Native | Balanced performance/efficiency | Apache 2.0 | Apr 2025 |

---

### Tier 3: Efficient Small Models

| Model | Sizes | Context | VRAM (Q4) | Best For | License | Released |
|-------|-------|---------|-----------|----------|---------|----------|
| **DeepSeek-R1-Distill-Qwen-7B** | 7B | 128K | ~5GB | Efficient reasoning | Apache 2.0 | Jan 2025 |
| **Qwen3-8B-Instruct** | 8B | 128K | ~6GB | Consumer hardware | Apache 2.0 | Apr 2025 |
| **Qwen3-4B-Thinking** | 4B | 128K | ~3GB | Edge reasoning | Apache 2.0 | Apr 2025 |

---

## 3.2 Hardware Recommendations by Budget (January 2025)

**Consumer GPU (24GB VRAM):**
- **Qwen3-30B-A3B** (only 3B active!) â­ Best bang-for-buck
- **DeepSeek-R1-Distill-Qwen-32B** - Top reasoning at 32B
- **Phi-4** - Best reasoning per parameter

**Workstation (48GB VRAM):**
- **Llama 4 Scout** (int4 quantization)
- **Qwen3-32B-Instruct**

**Multi-GPU (2-4Ã— A100/H100):**
- **DeepSeek-R1-Distill-Llama-70B**
- **Llama 4 Maverick**

**Cluster (8Ã— H100):**
- **DeepSeek-R1**
- **DeepSeek-V3.2**
- **Qwen3-235B**

---

## 3.3 Model Selection for Your Thesis

### Primary Recommendation: DeepSeek-R1-Distill-Qwen-32B â­â­â­

**Why:**
- Reasoning capabilities match o1-mini
- Fits on single consumer GPU (~20GB VRAM)
- Apache 2.0 license (open source requirement met)
- Native vLLM support
- Recent (Jan 2025) - cutting edge

**Use Cases:**
- Main reasoning model for all agent roles
- Supports complex multi-hop reasoning
- Handles query decomposition and synthesis

---

### Secondary Recommendation: Phi-4 â­â­

**Why:**
- Best reasoning per parameter (14B)
- Excellent on AIME (77.7%) and IFEval (92.1%)
- MIT license
- Efficient for experiments requiring multiple model instances

**Use Cases:**
- Experiments testing scaling with model size
- Running multiple parallel agents
- Ablation studies

---

### Baseline Comparisons

For fair comparison with papers, also test:
- **Llama-3-8B-Instruct** (used by MA-RAG, MAIN-RAG, MMOA-RAG)
- **Qwen2.5-7B** (used by DecEx-RAG)

---

## 3.4 Major Improvements from Previous Generation

**DeepSeek R1** (Jan 2025):
- Pure RL training without SFT
- MIT license
- 30Ã— cost reduction vs o1
- Competitive with proprietary models

**Qwen3** (Apr 2025):
- Hybrid thinking/non-thinking modes
- 36T token training (2Ã— Qwen2.5)
- 119 languages
- MoE architecture

**Llama 4** (Apr 2025):
- Native multimodal
- MoE architecture
- Extreme context windows (up to 10M tokens)

---

## 3.5 Closed-Source Baselines (For Comparison Only)

If you have API budget, compare against:
- **GPT-4o/GPT-4o-mini** (used by MA-RAG)
- **Claude 3.5 Sonnet/4 Opus**
- **Gemini 2.0 Pro**

**Note**: Your thesis requires open-source implementation, but citing closed-source performance helps contextualize results.

---

# 4. Dataset and Benchmark Coverage Analysis

## 4.1 Multi-Hop QA Datasets Comparison

| Dataset | Size | Hops | Metric | Difficulty | Best For Testing | SOTA (Jan 2025) | Use in Thesis? |
|---------|------|------|--------|------------|------------------|-----------------|----------------|
| **MuSiQue** â­â­â­ | 25K | 2-4 | F1 + Answerable | **HARD** | **Anti-shortcut; true multi-hop** | ~40% F1 | **PRIMARY** |
| **HotpotQA** â­â­ | 113K | 2 | Joint F1 + EM | Medium-Hard | Explainable multi-hop; supporting facts | ~55% F1 | **SECONDARY** |
| **2WikiMultiHopQA** â­â­ | 192K | 2-4 | F1 + Evidence | Medium-Hard | Explicit reasoning chains | ~50% EM | **SECONDARY** |
| **Bamboogle** â­ | 125 | 2 | Accuracy | **Adversarial** | Compositional search; stress test | ~50% Acc | **TERTIARY** |
| **Natural Questions** | 323K | 1 | F1 + EM | Medium | Real-world open-domain QA | ~47% F1 | Optional |
| **TriviaQA** | 95K | 1-2 | EM/F1 | Medium-Hard | Distant supervision; cross-doc | ~74% Acc | Optional |
| **PopQA** | 14K | 1 | Accuracy | Variable | Adaptive retrieval; long-tail | ~64% Acc | Optional |

---

## 4.2 Detailed Dataset Analysis

### MuSiQue (PRIMARY RECOMMENDATION) â­â­â­

**Why Best for Multi-Agent RAG Thesis:**
- **Anti-shortcut design**: Cannot be solved with single-hop retrieval
- **Requires true collaboration**: Multiple evidence pieces must be combined
- **Tests collective intelligence**: Agents must share intermediate findings
- **Hardest benchmark**: Low scores (30-40% F1) leave room for improvement

**Question Structure:**
- 2-4 hop questions requiring chaining facts
- Includes "unanswerable" questions (test retrieval quality)
- Supporting facts span multiple documents

**Example Question:**
*"What is the capital of the country where the director of film X was born?"*
- Hop 1: Retrieve director's birthplace â†’ Country Y
- Hop 2: Retrieve capital of Country Y â†’ City Z
- Cannot shortcut by directly searching "capital of [film director's birthplace]"

**Metrics:**
- Exact Match (EM)
- F1 Score
- Answerability detection

**Use in Your Thesis:**
- **Q1 (Parallelism)**: Test parallel retrieval of hop 1 vs. hop 2 documents
- **Q2 (Collective Intelligence)**: Test agent collaboration on multi-step reasoning
- **Q3 (Scaling Laws)**: Measure performance vs. number of retrieval rounds

---

### HotpotQA (SECONDARY) â­â­

**Why Important:**
- Most widely used multi-hop benchmark (113K questions)
- Requires supporting facts â†’ Tests explainability
- Good for comparison with existing papers (5 of 7 papers report on it)

**Question Structure:**
- 2-hop questions
- Must provide supporting facts (2 sentences justifying answer)
- Bridge questions + comparison questions

**Metrics:**
- Joint F1: Combined score for answer + supporting facts
- Answer EM
- Supporting Fact EM/F1

**Use in Your Thesis:**
- Baseline comparison with MA-RAG (52.1 EM), CoRAG (56.3 EM)
- Test if agents can identify correct supporting facts

---

### 2WikiMultiHopQA (SECONDARY) â­â­

**Why Important:**
- Explicit reasoning chains required
- Tests agent delegation (each hop = potential agent task)
- Evidence spans Wikipedia articles

**Question Structure:**
- 2-4 hop compositional questions
- Annotated reasoning chains
- Evidence from multiple Wikipedia articles

**Metrics:**
- EM/F1 for answer
- Evidence F1 (did system retrieve correct evidence?)

**Use in Your Thesis:**
- Test agent specialization (assign each hop to different agent)
- Compare against CoRAG (72.5 EM), DecEx-RAG (50.0 EM)

---

### Bamboogle (TERTIARY) â­

**Why Important:**
- Adversarial stress test
- Only 125 questions but very hard
- Tests compositional reasoning limits

**Question Structure:**
- Deliberately difficult compositional questions
- Requires creative retrieval strategies

**Use in Your Thesis:**
- Stress test for scaling laws (Q3)
- Show where additional compute helps most

---

## 4.3 Evaluation Metrics Checklist

### Answer Quality Metrics (REQUIRED)
- [ ] **Exact Match (EM)**: Binary match with ground truth
- [ ] **F1 Score**: Token-level overlap
- [ ] **Joint F1** (HotpotQA): Combined answer + supporting facts

### Retrieval Quality Metrics (REQUIRED for Multi-Agent RAG)
- [ ] **Recall@k**: Are relevant docs in top-k?
- [ ] **Precision@k**: What fraction of top-k are relevant?
- [ ] **MRR** (Mean Reciprocal Rank): Position of first relevant doc
- [ ] **nDCG@10**: Normalized discounted cumulative gain

### RAG-Specific Metrics
- [ ] **Supporting Facts F1** (HotpotQA): Did system retrieve correct evidence?
- [ ] **Evidence F1** (2WikiMultiHopQA): Quality of retrieved reasoning chain
- [ ] **Faithfulness**: Is answer grounded in retrieved docs?
- [ ] **Answer Relevance**: Does answer address the question?

### Efficiency Metrics (REQUIRED for Q1 and Q3)
- [ ] **Latency**: End-to-end time (seconds)
- [ ] **Token Usage**: Total tokens consumed (input + output)
- [ ] **API Calls**: Number of LLM calls
- [ ] **Retrieval Calls**: Number of search queries
- [ ] **GPU Memory**: Peak VRAM usage

### Multi-Agent Specific Metrics
- [ ] **Agent Contribution**: Ablation study (remove each agent, measure Î”EM)
- [ ] **Inter-Agent Agreement**: Consensus rate on judgments
- [ ] **Communication Overhead**: Extra tokens/calls for coordination

### Scaling Metrics (REQUIRED for Q3)
- [ ] **Performance vs. Agents**: EM(n=1, 2, 4, 8 agents)
- [ ] **Performance vs. Iterations**: EM(k=1, 2, 4, 8 rounds)
- [ ] **Compute Efficiency**: EM per token consumed
- [ ] **Diminishing Returns Point**: Where does scaling plateau?

---

## 4.4 Recommended Testing Strategy

### Phase 1: Baseline Establishment
**Dataset**: HotpotQA (well-established, many baselines)  
**Goal**: Implement single-agent RAG, measure baseline performance  
**Metrics**: EM, F1, Recall@5, Latency

### Phase 2: Multi-Agent Implementation
**Dataset**: MuSiQue (requires true multi-hop)  
**Goal**: Implement 2-3 agent system, test collaboration  
**Metrics**: EM, F1, Agent Contribution (ablation)

### Phase 3: Scaling Experiments
**Dataset**: MuSiQue + Bamboogle (hard problems benefit most from compute)  
**Goal**: Test Q3 scaling laws  
**Metrics**: Performance vs. Compute curves

### Phase 4: Comparative Analysis
**Datasets**: HotpotQA + 2WikiMultiHopQA  
**Goal**: Compare against published baselines  
**Metrics**: Report same metrics as CoRAG, MA-RAG papers

---

## 4.5 Evaluation Frameworks (Recommended Tools)

### RAGAS (v0.2+) â­ RECOMMENDED
**GitHub**: https://github.com/explodinggradients/ragas

**Features:**
- Comprehensive RAG evaluation metrics
- Faithfulness, Answer Relevance, Context Precision/Recall
- Supports custom metrics

**Installation:**
```bash
pip install ragas>=0.2.0
```

---

### RAGChecker (NeurIPS 2024)
**GitHub**: https://github.com/amazon-science/RAGChecker

**Features:**
- Fine-grained diagnostic metrics
- Claim-level verification
- Identifies specific failure modes

---

### TruLens
**GitHub**: https://github.com/truera/trulens

**Features:**
- Real-time RAG monitoring
- Interactive dashboard
- Feedback functions for custom metrics

---

### Custom Evaluation Script (REQUIRED)

You'll need to implement:

```python
def evaluate_multi_agent_rag(
    questions: List[str],
    ground_truth: List[str],
    predictions: List[str],
    retrieved_docs: List[List[str]],
    agent_outputs: List[Dict]
) -> Dict[str, float]:
    """
    Evaluate multi-agent RAG system
    
    Returns:
        {
            'exact_match': float,
            'f1_score': float,
            'retrieval_recall@5': float,
            'latency_ms': float,
            'token_usage': int,
            'agent_contribution': Dict[str, float]
        }
    """
    pass
```

---

# 5. Recommended Implementation Stack

## 5.1 Complete Stack for Your Thesis

| Component | Primary Choice | Alternative | Rationale |
|-----------|---------------|-------------|-----------|
| **Reasoning LLM** | DeepSeek-R1-Distill-Qwen-32B | Phi-4 14B | Best reasoning at 32B; Apache 2.0 license |
| **Efficient LLM** | Qwen3-30B-A3B | Qwen3-14B | Only 3B active; consumer GPU friendly |
| **Inference** | vLLM + SGLang | TensorRT-LLM | Best throughput; native MoE support |
| **Agent Framework** | LangGraph | CrewAI | Complex workflows; debugging tools; Q1+Q3 |
| **RAG Framework** | LlamaIndex | Haystack | Best retrieval; agent workflows; Q1 |
| **Optimization** | DSPy | Manual prompting | Auto-optimization; Q2+Q3 |
| **Primary Dataset** | **MuSiQue** | HotpotQA | Hardest multi-hop; tests true collaboration |
| **Secondary Dataset** | HotpotQA + 2WikiMQA | Bamboogle | Established benchmarks; comparability |
| **Evaluation** | RAGAS + Custom Joint F1 | RAGChecker | Standard + RAG-specific metrics |

---

## 5.2 Framework Selection by Research Question

### Q1: Parallel Collaborative Search for Efficiency
**Stack:**
- **LangGraph** for parallel node execution
- **LlamaIndex** for retrieval operations
- **vLLM** for efficient multi-agent inference
- **MuSiQue** for evaluation (requires parallel retrieval)

**Implementation:**
```python
from langgraph.graph import StateGraph
from llama_index import VectorStoreIndex

# Define parallel retrieval agents
graph = StateGraph()
graph.add_node("agent_1", retrieve_hop1)
graph.add_node("agent_2", retrieve_hop2)
graph.add_node("synthesizer", combine_results)

# Execute in parallel
graph.add_edge("START", "agent_1")
graph.add_edge("START", "agent_2")
graph.add_edge("agent_1", "synthesizer")
graph.add_edge("agent_2", "synthesizer")
```

---

### Q2: Collaboration Strategies for Collective Intelligence
**Stack:**
- **CrewAI** or **AutoGen** for agent communication patterns
- **DSPy** for optimizing collaboration strategies
- **2WikiMultiHopQA** for explicit reasoning chains

**Implementation:**
```python
from crewai import Agent, Task, Crew

# Define specialized agents
planner = Agent(role="Planner", goal="Decompose query")
retriever = Agent(role="Retriever", goal="Find documents")
judge = Agent(role="Judge", goal="Filter irrelevant docs")
generator = Agent(role="Generator", goal="Synthesize answer")

# Test consensus mechanisms: voting, debate, etc.
crew = Crew(agents=[planner, retriever, judge, generator])
result = crew.kickoff(task)
```

---

### Q3: Inference-Time Scaling Laws
**Stack:**
- **LangGraph** for compute graph management
- **DSPy** for scaling law experiments
- **MuSiQue + Bamboogle** for hard problems

**Implementation:**
```python
import dspy

# Test scaling: 1, 2, 4, 8 agents
for n_agents in [1, 2, 4, 8]:
    performance = run_multi_agent_rag(n_agents=n_agents)
    plot(n_agents, performance)  # Scaling curve
```

---

## 5.3 Critical Dependencies

```python
# Core ML Stack
transformers>=4.51.0      # Latest for Qwen3 support
vllm>=0.6.0               # MoE support, paged attention
torch>=2.4.0              # Flash attention 2
flash-attn>=2.7.0         # Efficient attention

# Agent Frameworks
langgraph>=0.2.0          # Stateful agents, parallel execution
llama-index>=0.12.0       # Advanced RAG, agent workflows
dspy-ai>=2.5.0            # Prompt optimization, scaling experiments
crewai>=0.80.0            # Multi-agent coordination (alternative)

# Evaluation
ragas>=0.2.0              # RAG metrics
datasets>=3.0.0           # Benchmark loading (HuggingFace)

# Retrieval
faiss-gpu>=1.7.0          # Vector search (GPU accelerated)
rank-bm25>=0.2.2          # Keyword search baseline
```

---

## 5.4 Implementation Roadmap

### Week 1-2: Setup & Baselines
- [ ] Set up DeepSeek-R1-Distill-Qwen-32B with vLLM
- [ ] Implement single-agent RAG baseline
- [ ] Evaluate on HotpotQA (establish baseline: ~45-50% EM expected)

### Week 3-4: Multi-Agent Implementation
- [ ] Implement 2-3 specialized agents (Planner, Retriever, Generator)
- [ ] Test on MuSiQue
- [ ] Measure improvement over single-agent baseline

### Week 5-6: Research Question Focus
- [ ] Q1: Implement parallel retrieval, measure latency reduction
- [ ] Q2: Test collaboration strategies (voting, debate)
- [ ] Q3: Run scaling experiments (1, 2, 4, 8 agents)

### Week 7-8: Evaluation & Analysis
- [ ] Comprehensive evaluation on MuSiQue, HotpotQA, 2WikiMultiHopQA
- [ ] Ablation studies (remove each agent, measure contribution)
- [ ] Compare against CoRAG, MA-RAG baselines

### Week 9-10: Thesis Writing
- [ ] Document methodology, results, analysis
- [ ] Create visualizations (scaling curves, ablation plots)
- [ ] Discuss limitations and future work

---

## 5.5 Key Papers to Cite

### For Background & Related Work:
- **RAG Foundations**: Lewis et al. (2020) - Retrieval-Augmented Generation for Knowledge-Intensive NLP
- **Multi-Agent AI**: AgentVerse, MetaGPT (2023) - Multi-agent coordination

### For Q1 (Collaborative Search):
- **MA-RAG**: Training-free collaborative CoT
- **MAIN-RAG**: Adaptive filtering in multi-agent setting
- **HM-RAG**: Hierarchical parallel retrieval

### For Q2 (Collaboration Strategies):
- **MMOA-RAG**: Multi-agent RL for coordination
- **MA-RAG**: Sequential collaboration with CoT
- **AgentsNet**: Large-scale multi-agent coordination

### For Q3 (Inference-Time Scaling):
- **CoRAG**: Test-time compute scaling with rejection sampling
- **DecEx-RAG**: MDP-based tree search scaling
- **DeepSeek-R1**: Pure RL reasoning (cite for scaling insights)

---

# Summary and Next Steps

## What You Now Have:

âœ… **Minimal RAG Requirements**: Core components, multi-agent specifics, evaluation framework  
âœ… **Benchmark Performance Table**: Exact scores from 7 leading papers  
âœ… **LLM Model Analysis**: Recommended models (DeepSeek-R1-Distill-Qwen-32B, Phi-4)  
âœ… **Dataset Coverage**: MuSiQue (primary), HotpotQA + 2WikiMultiHopQA (secondary)  
âœ… **Implementation Stack**: LangGraph + LlamaIndex + vLLM + DSPy  

## Your Thesis in Three Research Questions:

### Q1: Can multi-agent collaborative search increase parallelism?
- **Test**: Parallel vs. sequential retrieval on MuSiQue
- **Measure**: Latency reduction, EM improvement
- **Baseline**: Single-agent RAG (sequential retrieval)

### Q2: What collaboration strategies maximize collective intelligence?
- **Test**: Voting, debate, consensus mechanisms on 2WikiMultiHopQA
- **Measure**: EM with collaboration vs. sum of individual agents
- **Baseline**: Independent agents (no communication)

### Q3: Can inference-time scaling laws be observed?
- **Test**: Performance vs. number of agents/retrieval rounds on MuSiQue + Bamboogle
- **Measure**: EM(n), token efficiency, diminishing returns point
- **Baseline**: Single retrieval round, single agent

## Immediate Next Steps:

1. **Set up environment**: Install vLLM, LangGraph, LlamaIndex
2. **Download MuSiQue dataset**: Start with dev set (~2K questions)
3. **Deploy DeepSeek-R1-Distill-Qwen-32B**: Test inference speed
4. **Implement single-agent baseline**: Measure performance (expect ~25-30% EM on MuSiQue)
5. **Design your first multi-agent system**: 2-3 agents, test on 100 MuSiQue questions

## Success Criteria:

Your thesis will be successful if you can demonstrate:
- [ ] Multi-agent RAG outperforms single-agent baseline by â‰¥5% EM on MuSiQue
- [ ] Parallel retrieval reduces latency by â‰¥30% (Q1)
- [ ] Collaboration provides measurable benefit (Q2): Multi-agent EM > Average(individual agents)
- [ ] Scaling curve identified (Q3): Performance vs. compute with diminishing returns point

Good luck with your thesis! ðŸš€