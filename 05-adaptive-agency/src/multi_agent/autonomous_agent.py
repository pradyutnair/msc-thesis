"""Base class for M6 autonomous agents.

Each agent follows the observe → should_act → act cycle in an independent
async loop. Agents are fully autonomous: they decide *whether* and *when*
to act based on blackboard state, running concurrently with other agents.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from multi_agent.blackboard import Blackboard

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

        logger.debug("%s (%s) acting (action %d)", self.agent_id, self.agent_type,
                     blackboard.current_tick)
        tokens = await self.act(obs, blackboard)
        await blackboard.record_action(tokens)
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
