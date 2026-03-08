"""Base class for M6 autonomous agents.

Each agent follows the observe → should_act → act cycle, triggered by the
coordinator's tick loop. Agents are autonomous (decide *whether* to act)
but not polling (coordinator triggers checks).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from multi_agent.m6.blackboard import Blackboard

logger = logging.getLogger(__name__)


class AutonomousAgent(ABC):
    """Abstract base for blackboard-coordinated agents.

    Subclasses implement observe/should_act/act. The tick() method
    orchestrates the full cycle and returns whether the agent acted.
    """

    def __init__(self, agent_id: str, agent_type: str):
        self.agent_id = agent_id
        self.agent_type = agent_type

    async def tick(self, blackboard: Blackboard) -> bool:
        """Run one observe/should_act/act cycle.

        Returns True if the agent performed an action this tick.
        """
        obs = await self.observe(blackboard)
        if not self.should_act(obs):
            return False

        logger.debug("%s (%s) acting on tick %d", self.agent_id, self.agent_type,
                     blackboard.current_tick)
        tokens = await self.act(obs, blackboard)
        if tokens > 0:
            await blackboard.add_tokens(tokens)
        return True

    @abstractmethod
    async def observe(self, blackboard: Blackboard) -> dict[str, Any]:
        """Read relevant state from the blackboard."""
        ...

    @abstractmethod
    def should_act(self, observation: dict[str, Any]) -> bool:
        """Decide whether to act based on current observation."""
        ...

    @abstractmethod
    async def act(self, observation: dict[str, Any], blackboard: Blackboard) -> int:
        """Perform the agent's action. Returns approximate tokens used."""
        ...
