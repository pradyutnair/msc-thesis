"""M6: Blackboard-coordinated multi-agent RAG pipeline.

Architecture:
  - PlannerAgent: decompose -> monitor -> signal synthesis
  - WorkerAgent: plan -> execute loop per sub-question
  - SynthesizerAgent: aggregate evidence into final answer
  - Coordinator: concurrent async loops + watchdog
  - Blackboard: shared state for emergent coordination
"""
