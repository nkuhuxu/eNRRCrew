from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
from pydantic import ValidationError

from enrrcrew.config import AppSettings, SessionWorkspace
from enrrcrew.predictors.common import STRUCTURE_TYPES
from enrrcrew.recommendation import RecommendationService
from enrrcrew.schemas import (
    ELECTROLYTES,
    ELEMENT_SYMBOLS,
    PH_CATEGORIES,
    PredictionInput,
    PredictionType,
    RecommendationItem,
    RecommendationMode,
    RecommendationRequest,
    RecommendationResult,
)
from enrrcrew.services import RagService


@st.cache_resource(show_spinner=False)
def get_recommendation_service(
    dataset_path: str, models_dir: str, session_root: str
) -> RecommendationService:
    return RecommendationService(
        Path(dataset_path), Path(models_dir), Path(session_root)
    )


def _item_row(item: RecommendationItem) -> dict[str, object]:
    return {
        "ID": item.candidate.candidate_id,
        "Candidate": item.candidate.display_name,
        "Origin": item.candidate.origin.value,
        "Elements": ", ".join(item.candidate.elements),
        "Potential (V)": item.candidate.applied_potential,
        "Yield": item.yield_result.category,
        "Yield High score": item.yield_high_score,
        "FE": item.fe_result.category,
        "FE centroid margin": item.fe_centroid_margin,
        "Combined signal": item.combined_signal,
        "Domain": item.applicability_status,
        "Novelty (local)": item.novelty_score,
    }


def _prediction_from_candidate(
    item: RecommendationItem, prediction_type: PredictionType
) -> PredictionInput:
    candidate = item.candidate
    return PredictionInput(
        prediction_type=prediction_type,
        applied_potential=candidate.applied_potential,
        electrocatalyst=candidate.display_name,
        elements=candidate.elements,
        morphology=candidate.morphology,
        ph_categories=candidate.ph_categories,
        electrolytes=candidate.electrolytes,
        n15_labeling=False,
    )


def _render_details(
    item: RecommendationItem, settings: AppSettings, result: RecommendationResult
) -> None:
    st.markdown("### Candidate detail")
    left, middle, right = st.columns(3)
    left.metric("Combined signal", f"{item.combined_signal:.3f}")
    middle.metric("Applicability", item.applicability_status)
    right.metric("Local novelty", f"{item.novelty_score:.3f}")
    st.json(item.candidate.model_dump(mode="json"))
    if item.warnings:
        with st.expander("Warnings", expanded=True):
            for warning in item.warnings:
                st.write(f"- {warning}")
    with st.expander("Nearest local systems"):
        st.dataframe(pd.DataFrame(item.nearest_known_candidates), use_container_width=True)

    send_yield, send_fe = st.columns(2)
    if send_yield.button("Send to Yield form", use_container_width=True):
        st.session_state.pending_prediction = _prediction_from_candidate(
            item, PredictionType.YIELD
        ).model_dump(mode="json")
        st.rerun()
    if send_fe.button("Send to FE form", use_container_width=True):
        st.session_state.pending_prediction = _prediction_from_candidate(
            item, PredictionType.FE
        ).model_dump(mode="json")
        st.rerun()

    evidence_key = f"recommendation_evidence_{result.run_id}_{item.candidate.candidate_id}"
    if st.button(
        "Retrieve GraphRAG evidence",
        disabled=not bool(settings.api_key),
        help="Configure an API key in the sidebar to enable local GraphRAG evidence.",
    ):
        candidate = item.candidate
        question = (
            "Summarize direct evidence and close analogues in the local eNRR knowledge graph for "
            f"a catalyst containing {', '.join(candidate.elements)}, with "
            f"{candidate.morphology or 'unspecified morphology'}, at "
            f"{candidate.applied_potential:.3f} V, pH categories "
            f"{sorted(candidate.ph_categories)}, and electrolytes "
            f"{sorted(candidate.electrolytes)}. Clearly distinguish direct matches from analogues."
        )
        try:
            with st.spinner("Searching the local knowledge graph…"):
                st.session_state[evidence_key] = RagService(
                    settings.asset_root,
                    settings.graphrag_config,
                    settings.api_key,
                    settings.base_url,
                ).search(question, "local", 0, "multiple paragraphs")
        except Exception as exc:
            st.session_state[evidence_key] = f"Evidence retrieval failed: {exc}"
    if evidence := st.session_state.get(evidence_key):
        st.markdown(evidence)
    st.caption(
        "Not found in the current CSV or local knowledge graph does not mean unreported "
        "in global literature or patents."
    )


def render_recommendation(
    settings: AppSettings, workspace: SessionWorkspace
) -> None:
    st.subheader("Catalyst recommendation")
    st.info(
        "Ranks experiment candidates for validation. Generated entries are hypotheses, "
        "not confirmed catalyst discoveries."
    )
    service = get_recommendation_service(
        str(settings.input_dir / "data_include_morphology_electrocatalyst.csv"),
        str(settings.models_dir),
        str(workspace.root),
    )
    default_elements = service.default_allowed_elements()
    with st.form("recommendation_form", border=True):
        st.markdown("### Candidate constraints")
        first, second = st.columns(2)
        with first:
            mode = st.selectbox(
                "Candidate mode",
                list(RecommendationMode),
                index=2,
                format_func=lambda value: value.value,
            )
            allowed = st.multiselect(
                "Allowed elements",
                sorted(ELEMENT_SYMBOLS),
                default=default_elements,
            )
            forbidden = st.multiselect("Forbidden elements", sorted(ELEMENT_SYMBOLS))
            max_elements = st.number_input(
                "Maximum elements per candidate", min_value=1, max_value=7, value=4
            )
            morphologies = st.multiselect("Morphology filter", STRUCTURE_TYPES)
        with second:
            potential_min = st.number_input(
                "Minimum potential (V)", value=-0.8, min_value=-10.0, max_value=10.0
            )
            potential_max = st.number_input(
                "Maximum potential (V)", value=0.0, min_value=-10.0, max_value=10.0
            )
            potential_step = st.number_input(
                "Potential step (V)", value=0.1, min_value=0.05, max_value=0.5
            )
            ph_categories = st.multiselect("pH filter", PH_CATEGORIES)
            electrolytes = st.multiselect("Electrolyte filter", ELECTROLYTES)
            generated_limit = st.number_input(
                "Generated candidate limit", min_value=0, max_value=2000, value=500
            )
            result_limit = st.number_input(
                "Result limit", min_value=1, max_value=100, value=20
            )
        submitted = st.form_submit_button("Run recommendation", use_container_width=True)

    if submitted:
        try:
            request = RecommendationRequest(
                mode=mode,
                allowed_elements=set(allowed),
                forbidden_elements=set(forbidden),
                max_elements=int(max_elements),
                morphologies=set(morphologies),
                ph_categories=set(ph_categories),
                electrolytes=set(electrolytes),
                potential_min=potential_min,
                potential_max=potential_max,
                potential_step=potential_step,
                generated_candidate_limit=int(generated_limit),
                result_limit=int(result_limit),
            )
            with st.spinner("Screening known and generated catalyst systems…"):
                result = service.recommend(request)
            st.session_state.recommendation_result = result.model_dump(mode="json")
        except ValidationError as exc:
            st.error("Please correct the recommendation constraints.")
            for error in exc.errors():
                st.caption(f"{'.'.join(map(str, error['loc']))}: {error['msg']}")
        except Exception as exc:
            st.error(f"Recommendation failed: {exc}")

    saved = st.session_state.get("recommendation_result")
    if not saved:
        return
    result = RecommendationResult.model_validate(saved)
    counts = result.rejected_counts
    columns = st.columns(4)
    columns[0].metric("Known evaluated", counts.get("evaluated_known", 0))
    columns[1].metric("Generated", counts.get("generated", 0))
    columns[2].metric("Strict double-high", counts.get("strict_double_high", 0))
    columns[3].metric("Out of domain", counts.get("out_of_domain", 0))

    st.markdown("### Strict recommendations")
    if result.recommendations:
        st.dataframe(
            pd.DataFrame(_item_row(item) for item in result.recommendations),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.warning("No candidate satisfies the strict double-high and applicability rules.")
    st.markdown("### Near misses")
    if result.near_misses:
        st.dataframe(
            pd.DataFrame(_item_row(item) for item in result.near_misses),
            use_container_width=True,
            hide_index=True,
        )

    rows = [_item_row(item) for item in result.recommendations]
    download_left, download_right = st.columns(2)
    download_left.download_button(
        "Download recommendations CSV",
        pd.DataFrame(rows).to_csv(index=False).encode("utf-8"),
        file_name=f"recommendations_{result.run_id}.csv",
        mime="text/csv",
        use_container_width=True,
    )
    download_right.download_button(
        "Download audit JSON",
        result.model_dump_json(indent=2).encode("utf-8"),
        file_name=f"recommendation_{result.run_id}.json",
        mime="application/json",
        use_container_width=True,
    )

    selectable = result.recommendations + result.near_misses
    if selectable:
        by_id = {item.candidate.candidate_id: item for item in selectable}
        selected_id = st.selectbox(
            "Inspect candidate",
            list(by_id),
            format_func=lambda candidate_key: by_id[candidate_key].candidate.display_name,
        )
        _render_details(by_id[selected_id], settings, result)
