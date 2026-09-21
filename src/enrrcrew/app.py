from __future__ import annotations

import logging
from pathlib import Path

import streamlit as st

from enrrcrew.agents import (
    AgentRequest,
    AgentToolContext,
    ChatTurn,
    ConversationRoute,
    CsvAnalysisAgent,
    RouteStrategy,
    decide_route,
)
from enrrcrew.config import AppSettings, SessionWorkspace
from enrrcrew.predictors import FEPredictor, YieldPredictor
from enrrcrew.schemas import PredictionInput, PredictionResult, PredictionType
from enrrcrew.services import (
    PredictionTextExtractor,
    RagService,
    RecommendationTextExtractor,
    SandboxRunner,
)
from enrrcrew.services.conversation import ConversationService
from enrrcrew.ui import (
    apply_prediction_defaults,
    apply_theme,
    get_recommendation_service,
    render_knowledge_update,
    render_prediction_form,
    render_recommendation,
)

LOGGER = logging.getLogger(__name__)


@st.cache_resource(show_spinner=False)
def get_predictor(kind: str, models_dir: str) -> YieldPredictor | FEPredictor:
    path = Path(models_dir)
    return YieldPredictor(path) if kind == "yield" else FEPredictor(path)


def initialize_session(settings: AppSettings) -> SessionWorkspace:
    if "session_id" not in st.session_state:
        workspace = SessionWorkspace.create(settings)
        st.session_state.session_id = workspace.session_id
    return SessionWorkspace.create(settings, st.session_state.session_id)


def render_sidebar_guide() -> None:
    with st.expander("How to use", expanded=False):
        st.markdown(
            """
**Choose a workspace based on your goal**

1. **Dialogue & retrieval** — Search the literature or combine retrieval, prediction, and interpretation.
2. **Yield prediction** — Validate an ammonia-yield classification with structured conditions.
3. **FE prediction** — Validate a Faradaic-efficiency classification with structured conditions.
4. **CSV analysis** — Generate code, review it, then run it in the Docker sandbox.
5. **Catalyst recommendation** — Rank strict Yield/FE candidates within explicit constraints.
6. **Knowledge base update** — Upload abstracts, review a batch, build an index, or switch versions.

**Operating rules**

- Predictions and recommendations prioritize experiments; they are not experimental conclusions.
- Review generated CSV code before execution.
- Keep the launch terminal open during indexing and approve each batch only once.
- Use `Activate` to switch to a specific version; use `Rollback` to undo the latest switch.
"""
        )


def render_sidebar(settings: AppSettings, workspace: SessionWorkspace) -> AppSettings:
    with st.sidebar:
        st.markdown("## Control rack")
        st.caption(f"SESSION · {workspace.session_id[:10]}")
        api_key = st.text_input(
            "API key",
            type="password",
            value="",
            placeholder="Using environment" if settings.api_key else "Required for LLM/RAG",
            help="Kept only in this browser session and never written to disk.",
        )
        base_url = st.text_input("Base URL", value=settings.base_url)
        configured = bool(api_key.strip() or settings.api_key)
        st.markdown(
            f"**LLM link:** {'ONLINE' if configured else 'NOT CONFIGURED'}  \n"
            f"**Asset root:** `{settings.asset_root}`  \n"
            f"**Runtime:** `{workspace.root}`"
        )
        missing = settings.validate_assets()
        if missing:
            st.error("Missing assets:\n" + "\n".join(missing))
        render_sidebar_guide()
    return settings.with_session_credentials(api_key, base_url)


def render_header() -> None:
    st.markdown(
        """
        <section class="hero">
          <div class="eyebrow">ELECTROCATALYSIS · DECISION SYSTEM / 02</div>
          <h1>eNRR<span style="color:#b8f34a">Crew</span></h1>
          <p>Literature retrieval, model-backed catalyst screening, and isolated data analysis—
          rebuilt around reproducible scientific inputs and session-safe execution.</p>
          <div class="status-strip">
            <span class="status-pill">PYTHON 3.12.7</span>
            <span class="status-pill">MODEL PIPELINES LOCKED</span>
            <span class="status-pill">DOCKER-ONLY CODE EXECUTION</span>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _render_chat_artifacts(artifacts: dict[str, object]) -> None:
    for key in ("yield_result", "fe_result"):
        if value := artifacts.get(key):
            render_result(PredictionResult.model_validate(value))
    recommendation = artifacts.get("recommendation_result")
    if recommendation:
        from enrrcrew.schemas import RecommendationResult

        result = RecommendationResult.model_validate(recommendation)
        st.caption(
            f"Recommendation run `{result.run_id}` · "
            f"{len(result.recommendations)} strict matches · "
            f"{len(result.near_misses)} near misses"
        )
    if artifacts.get("csv_code"):
        st.info("CSV code is ready for review in the CSV analysis tab.")


def _render_agent_trace(debug: list[dict[str, object]]) -> None:
    if not debug:
        return
    with st.expander("Agent trace"):
        for event in debug:
            tool = f" · {event['tool']}" if event.get("tool") else ""
            elapsed = (
                f" · {event['elapsed_ms']} ms"
                if event.get("elapsed_ms") is not None
                else ""
            )
            st.code(
                f"R{event['round_index']} · {event['agent']}{tool} · "
                f"{event['status']}{elapsed}\n{event['summary']}",
                language="text",
            )


def _active_datasets(settings: AppSettings, workspace: SessionWorkspace) -> dict[str, Path]:
    datasets = {
        "curated": settings.input_dir / "data_include_morphology_electrocatalyst.csv"
    }
    uploaded = st.session_state.get("active_session_csv")
    if uploaded:
        path = Path(uploaded).resolve()
        if path.is_file() and path.is_relative_to(workspace.root.resolve()):
            datasets["session_upload"] = path
    return datasets


def _agent_context(
    settings: AppSettings,
    workspace: SessionWorkspace,
) -> AgentToolContext:
    dataset = settings.input_dir / "data_include_morphology_electrocatalyst.csv"
    return AgentToolContext(
        settings=settings,
        workspace=workspace,
        rag=RagService(
            settings.rag_service_url,
            settings.rag_service_token,
            settings.api_key,
            settings.base_url,
        ),
        yield_predictor=get_predictor("yield", str(settings.models_dir)),
        fe_predictor=get_predictor("fe", str(settings.models_dir)),
        recommendation_service=get_recommendation_service(
            str(dataset), str(settings.models_dir), str(workspace.root)
        ),
        prediction_extractor=PredictionTextExtractor(
            settings.api_key, settings.base_url, settings.chat_model
        ),
        recommendation_extractor=RecommendationTextExtractor(
            settings.api_key, settings.base_url, settings.chat_model
        ),
        csv_agent=CsvAnalysisAgent(
            settings.api_key, settings.base_url, settings.chat_model
        ),
        datasets=_active_datasets(settings, workspace),
    )


def _offline_chat_guidance(prompt: str) -> str:
    decision = decide_route(prompt)
    if decision.strategy is not RouteStrategy.FAST_PATH:
        return (
            "Configure an API key to use open-ended conversational coordination and retrieval. "
            "The structured prediction and recommendation tabs remain available offline."
        )
    guidance = {
        ConversationRoute.YIELD_PREDICTION: "Open the Yield prediction tab to run the local model.",
        ConversationRoute.FE_PREDICTION: "Open the FE prediction tab to run the local model.",
        ConversationRoute.CATALYST_RECOMMENDATION: (
            "Open the Catalyst recommendation tab to run deterministic offline screening."
        ),
        ConversationRoute.CSV: (
            "Open the CSV analysis tab to review code and use the Docker sandbox. "
            "An API key is required to generate a new code draft."
        ),
        ConversationRoute.RETRIEVAL: (
            "Configure an API key to search the local GraphRAG index from the dialogue tab."
        ),
    }
    return guidance[decision.detected_routes[0]]


def render_chat(settings: AppSettings, workspace: SessionWorkspace) -> None:
    st.subheader("Literature console")
    control_a, control_b, control_c = st.columns([1, 1, 2])
    selected_mode = control_a.selectbox("Search mode", ["local", "global"])
    mode = selected_mode if selected_mode in {"local", "global"} else "local"
    selected_community = control_b.selectbox("Community", [0, 1, 2])
    community = selected_community if isinstance(selected_community, int) else 0
    selected_response_type = control_c.selectbox(
        "Response format",
        ["single paragraph", "prioritized list", "multiple paragraphs", "multiple-page report"],
    )
    response_type = (
        selected_response_type
        if isinstance(selected_response_type, str)
        else "single paragraph"
    )
    messages = st.session_state.setdefault("chat_messages", [])
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            _render_chat_artifacts(message.get("artifacts", {}))
            _render_agent_trace(message.get("debug", []))
    prompt = st.chat_input("Ask about electrocatalytic nitrogen reduction…")
    if not prompt:
        return
    history = [
        ChatTurn(role=message["role"], content=message["content"])
        for message in messages[-12:]
        if message.get("role") in {"user", "assistant"} and message.get("content")
    ]
    messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    if not settings.api_key:
        answer = _offline_chat_guidance(prompt)
        debug: list[dict[str, object]] = []
        artifacts: dict[str, object] = {}
    else:
        try:
            with st.spinner("Coordinating the eNRR specialist crew…"):
                request = AgentRequest(
                    question=prompt,
                    history=history,
                    rag_mode=mode,
                    community=community,
                    response_type=response_type,
                    active_dataset=(
                        "session_upload"
                        if "session_upload" in _active_datasets(settings, workspace)
                        else "curated"
                    ),
                )
                managed = ConversationService().answer(
                    request,
                    _agent_context(settings, workspace),
                )
                answer = managed.answer
                debug = [event.model_dump(mode="json") for event in managed.trace]
                artifacts = managed.artifacts.model_dump(mode="json")
        except Exception as exc:
            answer = f"Coordination failed: {type(exc).__name__}"
            debug = []
            artifacts = {}
    if result := artifacts.get("yield_result"):
        st.session_state.result_yield = result
    if result := artifacts.get("fe_result"):
        st.session_state.result_fe = result
    if result := artifacts.get("recommendation_result"):
        st.session_state.recommendation_result = result
    if code := artifacts.get("csv_code"):
        st.session_state.csv_code = code
        st.session_state.csv_code_editor = code
    assistant_message = {
        "role": "assistant",
        "content": answer,
        "debug": debug,
        "artifacts": artifacts,
    }
    messages.append(assistant_message)
    with st.chat_message("assistant"):
        st.markdown(answer)
        _render_chat_artifacts(artifacts)
        _render_agent_trace(debug)


def render_result(result: PredictionResult) -> None:
    left, middle, right = st.columns(3)
    left.metric("Classification", result.category)
    middle.metric("Probability", "—" if result.probability is None else f"{result.probability:.1%}")
    right.metric("Cluster", "—" if result.cluster is None else str(result.cluster))
    st.caption(f"Model · {result.model_version}")
    if result.warnings:
        with st.expander("Prediction notes"):
            for warning in result.warnings:
                st.write(f"• {warning}")
    with st.expander("Validated model input"):
        st.json(result.input_data)


def render_prediction_tab(settings: AppSettings, prediction_type: PredictionType) -> None:
    label = "Yield" if prediction_type is PredictionType.YIELD else "FE"
    st.subheader(f"{label} screening")
    with st.expander("Extract input from a natural-language description"):
        description = st.text_area(
            "Experimental description",
            key=f"natural_{prediction_type.value}",
            height=140,
        )
        if st.button("Extract & review", key=f"extract_{prediction_type.value}"):
            try:
                with st.spinner("Structuring catalyst parameters…"):
                    extracted = PredictionTextExtractor(
                        settings.api_key, settings.base_url, settings.chat_model
                    ).extract(description, prediction_type)
                apply_prediction_defaults(extracted)
                st.success("Input extracted. Review every field before prediction.")
            except Exception as exc:
                st.error(f"Extraction failed: {exc}")
    submitted, value = render_prediction_form(prediction_type)
    if submitted and value is not None:
        try:
            with st.spinner("Generating composition descriptors and running the model…"):
                predictor = get_predictor(prediction_type.value, str(settings.models_dir))
                result = predictor.predict(value)
            st.session_state[f"result_{prediction_type.value}"] = result.model_dump(mode="json")
        except Exception as exc:
            LOGGER.exception("Prediction failed for %s", prediction_type.value)
            st.error(f"Prediction failed: {exc}")
    saved_result = st.session_state.get(f"result_{prediction_type.value}")
    if saved_result:
        render_result(PredictionResult.model_validate(saved_result))


def _save_upload(workspace: SessionWorkspace, upload: object) -> Path:
    filename = Path(getattr(upload, "name", "uploaded.csv")).name
    if not filename.lower().endswith(".csv"):
        raise ValueError("Only CSV files are accepted")
    destination = workspace.root / filename
    destination.write_bytes(upload.getvalue())
    return destination


def render_csv_analysis(settings: AppSettings, workspace: SessionWorkspace) -> None:
    st.subheader("Isolated CSV workbench")
    st.info("Generated Python runs only in the restricted Docker sandbox—never on the host.")
    source_mode = st.radio("Dataset", ["Curated eNRR dataset", "Upload CSV"], horizontal=True)
    if source_mode == "Upload CSV":
        upload = st.file_uploader("CSV file", type=["csv"])
        csv_path = _save_upload(workspace, upload) if upload is not None else None
        if csv_path is not None:
            st.session_state.active_session_csv = str(csv_path)
    else:
        csv_path = settings.input_dir / "data_include_morphology_electrocatalyst.csv"
    question = st.text_area(
        "Analysis request",
        placeholder="Compare median Faradaic efficiency by electrolyte and save a chart.",
    )
    if st.button("Generate analysis code", disabled=csv_path is None):
        try:
            alias = "dataset.csv"
            with st.spinner("Drafting constrained analysis…"):
                code = CsvAnalysisAgent(
                    settings.api_key, settings.base_url, settings.chat_model
                ).generate_code(question, csv_path, alias)
            st.session_state.csv_code = code
            st.session_state.csv_code_editor = code
        except Exception as exc:
            st.error(f"Code generation failed: {exc}")
    if "csv_code_editor" not in st.session_state:
        st.session_state.csv_code_editor = st.session_state.get("csv_code", "")
    code = st.text_area(
        "Sandbox code — review before execution",
        height=280,
        key="csv_code_editor",
    )
    if st.button("Run in Docker sandbox", disabled=not code or csv_path is None):
        try:
            with st.spinner("Running isolated container…"):
                result = SandboxRunner(
                    settings.sandbox_image, settings.sandbox_timeout
                ).run(code, {"dataset.csv": csv_path}, workspace.root)
            st.session_state.sandbox_result = result.model_dump(mode="json")
        except Exception as exc:
            st.error(str(exc))
    saved = st.session_state.get("sandbox_result")
    if saved:
        from enrrcrew.schemas import SandboxResult

        result = SandboxResult.model_validate(saved)
        st.code(result.stdout or result.stderr or "No textual output", language="text")
        st.caption(f"Exit code: {result.exit_code} · timed out: {result.timed_out}")
        for output in result.output_files:
            if output.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                st.image(str(output), caption=output.name)


def main() -> None:
    st.set_page_config(page_title="eNRRCrew", page_icon="⚗️", layout="wide")
    apply_theme()
    base_settings = AppSettings.from_environment()
    workspace = initialize_session(base_settings)
    settings = render_sidebar(base_settings, workspace)
    pending_prediction = st.session_state.pop("pending_prediction", None)
    if pending_prediction:
        apply_prediction_defaults(PredictionInput.model_validate(pending_prediction))
    render_header()
    chat_tab, yield_tab, fe_tab, csv_tab, recommendation_tab, knowledge_tab = st.tabs(
        [
            "⌁ Dialogue & retrieval",
            "↗ Yield prediction",
            "ϕ FE prediction",
            "▦ CSV analysis",
            "◇ Catalyst recommendation",
            "⌘ Knowledge base update",
        ]
    )
    with chat_tab:
        render_chat(settings, workspace)
    with yield_tab:
        render_prediction_tab(settings, PredictionType.YIELD)
    with fe_tab:
        render_prediction_tab(settings, PredictionType.FE)
    with csv_tab:
        render_csv_analysis(settings, workspace)
    with recommendation_tab:
        render_recommendation(settings, workspace)
    with knowledge_tab:
        render_knowledge_update(settings, workspace)


if __name__ == "__main__":
    main()
