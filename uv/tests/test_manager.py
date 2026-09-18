from enrrcrew.agents import AgentManager


class FakeRag:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    def search(self, *args: object) -> str:
        self.calls.append(args)
        return "grounded answer"


def test_agent_manager_coordinates_open_retrieval() -> None:
    rag = FakeRag()
    result = AgentManager(rag).answer("question", "local", 2, "prioritized list")

    assert result.content == "grounded answer"
    assert rag.calls == [("question", "local", 2, "prioritized list")]
    assert result.debug_messages[0] == "Route: open-ended retrieval"
    assert "RagService.search" in result.debug_messages[1]
