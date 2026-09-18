from __future__ import annotations

from dataclasses import dataclass

from enrrcrew.services.rag import RagService


@dataclass(frozen=True, slots=True)
class ManagerAnswer:
    content: str
    debug_messages: tuple[str, ...]


class AgentManager:
    """Coordinate open-ended questions while keeping tool details out of the answer."""

    def __init__(self, rag: RagService):
        self.rag = rag

    def answer(
        self,
        question: str,
        mode: str,
        community: int,
        response_type: str,
    ) -> ManagerAnswer:
        debug = (
            "Route: open-ended retrieval",
            f"Tool: RagService.search(mode={mode}, community={community}, "
            f"response_type={response_type!r})",
        )
        content = self.rag.search(question, mode, community, response_type)
        return ManagerAnswer(content=content, debug_messages=debug)
