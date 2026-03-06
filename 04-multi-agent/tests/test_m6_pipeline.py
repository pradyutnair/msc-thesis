from arag.core.config import Config
from arag.tools.registry import ToolRegistry
from multi_agent.m6_pipeline import M6LitCorePipeline
from multi_agent.m6_types import BlackboardState, ClaimRecord, FrontierItem


class DummyLLM:
    pass


def _make_pipeline() -> M6LitCorePipeline:
    cfg = Config.from_dict(
        {
            "multi_agent": {
                "m6_max_manager_steps": 4,
            }
        }
    )
    return M6LitCorePipeline(llm_client=DummyLLM(), tools=ToolRegistry(), config=cfg)


def test_dependencies_satisfied_uses_supported_claims() -> None:
    pipeline = _make_pipeline()
    board = BlackboardState(claims=[ClaimRecord(id=1, entity="A", relation="bridge", value="B", status="supported")])
    frontier = FrontierItem(id=0, role_hint="attribute", goal="Find birthplace", depends_on_claim_ids=[1])
    task = pipeline._fallback_decision(BlackboardState(frontier=[frontier]), "question")
    assert task.action == "spawn_attribute_worker"


def test_fallback_prefers_merge_claim() -> None:
    pipeline = _make_pipeline()
    board = BlackboardState(claims=[ClaimRecord(id=2, entity="A", relation="fact", value="B", status="proposed")])
    decision = pipeline._fallback_decision(board, "question")
    assert decision.action == "merge_claim"
    assert decision.claim_ids == [2]


def test_expand_queries_adds_role_specific_variants() -> None:
    pipeline = _make_pipeline()
    queries = pipeline._expand_queries("bridge", "Who founded the city", ["city founder"])
    assert any("wikipedia" in query for query in queries)
    assert queries[0] == "city founder"
