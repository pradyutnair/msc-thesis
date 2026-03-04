"""MA²RAG: Multi-Agent Agentic RAG with Shared Evidence Caching."""

from multi_agent.types import (
    SubQuestion,
    DecompositionPlan,
    AgentResult,
    CachedDocument,
    PipelineResult,
)
from multi_agent.decomposer import Decomposer
from multi_agent.evidence_cache import EvidenceCache
from multi_agent.search_agent import SearchAgent
from multi_agent.dispatcher import Dispatcher
from multi_agent.aggregator import Aggregator
from multi_agent.pipeline import MultiAgentPipeline

__all__ = [
    "SubQuestion",
    "DecompositionPlan",
    "AgentResult",
    "CachedDocument",
    "PipelineResult",
    "Decomposer",
    "EvidenceCache",
    "SearchAgent",
    "Dispatcher",
    "Aggregator",
    "MultiAgentPipeline",
]
