from __future__ import annotations

import logging
from pathlib import Path

import streamlit as st

from enrrcrew.agents import AgentManager, ConversationRoute, CsvAnalysisAgent, route_message
from enrrcrew.config import AppSettings, SessionWorkspace
from enrrcrew.predictors import FEPredictor, YieldPredictor
from enrrcrew.schemas import PredictionInput, PredictionResult, PredictionType
from enrrcrew.services import PredictionTextExtractor, RagService, SandboxRunner
from enrrcrew.ui import (
    apply_prediction_defaults,
    apply_theme,
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


def render_chat(settings: AppSettings) -> None:
    st.subheader("Literature console")
    control_a, control_b, control_c = st.columns([1, 1, 2])
    mode = control_a.selectbox("Search mode", ["local", "global"])
    community = control_b.selectbox("Community", [0, 1, 2])
    response_type = control_c.selectbox(
        "Response format",
        ["single paragraph", "prioritized list", "multiple paragraphs", "multiple-page report"],
    )
    messages = st.session_state.setdefault("chat_messages", [])
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("debug"):
                with st.expander("Agent trace"):
                    for item in message["debug"]:
                        st.code(item, language="text")
    prompt = st.chat_input("Ask about electrocatalytic nitrogen reduction…")
    if not prompt:
        return
    messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    route = route_message(prompt)
    if route is ConversationRoute.YIELD_PREDICTION:
        answer = "Use the **Yield prediction** tab so the scientific inputs can be validated."
    elif route is ConversationRoute.FE_PREDICTION:
        answer = "Use the **FE prediction** tab so the scientific inputs can be validated."
    elif route is ConversationRoute.CSV:
        answer = "Use the **CSV analysis** tab to select a dataset and run isolated analysis."
    elif route is ConversationRoute.CATALYST_RECOMMENDATION:
        answer = (
            "Use the **Catalyst recommendation** tab for constrained, model-backed screening."
        )
    else:
        try:
            with st.spinner("Traversing the eNRR knowledge graph…"):
                managed = AgentManager(
                    RagService(
                        settings.asset_root,
                        settings.graphrag_config,
                        settings.api_key,
                        settings.base_url,
                    )
                ).answer(prompt, mode, community, response_type)
                answer = managed.content
                debug = list(managed.debug_messages)
        except Exception as exc:
            answer = f"Retrieval failed: {exc}"
            debug = [f"Manager error: {type(exc).__name__}"]
    assistant_message = {"role": "assistant", "content": answer}
    if route is ConversationRoute.RETRIEVAL:
        assistant_message["debug"] = debug
    messages.append(assistant_message)
    with st.chat_message("assistant"):
        st.markdown(answer)
        if assistant_message.get("debug"):
            with st.expander("Agent trace"):
                for item in assistant_message["debug"]:
                    st.code(item, language="text")


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
        except Exception as exc:
            st.error(f"Code generation failed: {exc}")
    code = st.text_area(
        "Sandbox code — review before execution",
        value=st.session_state.get("csv_code", ""),
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
    chat_tab, yield_tab, fe_tab, csv_tab, recommendation_tab = st.tabs(
        [
            "⌁ Dialogue & retrieval",
            "↗ Yield prediction",
            "ϕ FE prediction",
            "▦ CSV analysis",
            "◇ Catalyst recommendation",
        ]
    )
    with chat_tab:
        render_chat(settings)
    with yield_tab:
        render_prediction_tab(settings, PredictionType.YIELD)
    with fe_tab:
        render_prediction_tab(settings, PredictionType.FE)
    with csv_tab:
        render_csv_analysis(settings, workspace)
    with recommendation_tab:
        render_recommendation(settings, workspace)


if __name__ == "__main__":
    main()
