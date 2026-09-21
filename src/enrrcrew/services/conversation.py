from __future__ import annotations

from enrrcrew.agents import AgentManager, AgentRequest, AgentRunResult, AgentToolContext


class ConversationService:
    """UI-independent entry point for one bounded conversational run."""

    def __init__(self, manager: AgentManager | None = None):
        self.manager = manager or AgentManager()

    def answer(
        self,
        request: AgentRequest,
        context: AgentToolContext,
    ) -> AgentRunResult:
        return self.manager.answer(request, context)
