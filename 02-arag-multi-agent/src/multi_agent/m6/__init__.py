"""M6: Blackboard-coordinated multi-agent RAG pipeline.

Architecture (AgentFlow-inspired):
  - PlannerAgent: decompose → monitor → synthesize lifecycle
  - WorkerAgent: plan → execute → verify loop per sub-question
  - Coordinator: concurrent async loops + watchdog
  - Blackboard: shared state for emergent coordination
"""
