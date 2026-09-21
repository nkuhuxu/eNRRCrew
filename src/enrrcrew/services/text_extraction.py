from __future__ import annotations

import json

from openai import OpenAI

from enrrcrew.schemas import PredictionInput, PredictionType, RecommendationRequest


class MissingPredictionFields(ValueError):
    def __init__(self, fields: list[str]):
        self.fields = fields
        super().__init__(f"Missing required prediction fields: {', '.join(fields)}")


class PredictionTextExtractor:
    def __init__(self, api_key: str, base_url: str, model: str):
        if not api_key:
            raise ValueError("An API key is required for natural-language extraction")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def extract(self, text: str, prediction_type: PredictionType) -> PredictionInput:
        if not text.strip():
            raise ValueError("Description cannot be empty")
        prompt = f"""
Extract an electrocatalytic nitrogen-reduction prediction input from the text.
Return only a JSON object with this exact shape:
{{
  "prediction_type": "{prediction_type.value}",
  "applied_potential": null,
  "electrocatalyst": null,
  "elements": [],
  "morphology": "",
  "ph_categories": [],
  "electrolytes": [],
  "n15_labeling": false
}}
Use only chemical element symbols in elements. Applied potential must be numeric in volts.
Use null or an empty array when a required value is not stated. Never invent a
catalyst name, element, potential, morphology, pH category, or electrolyte.
Allowed pH categories: acidic, alkaline, ionic liquid, khco3, li tfsi, nabf4,
neutral, weak acid. Allowed electrolytes: h2so4, hcl, k2so4, kclo4, koh,
li2so4, licl, liclo4, lioh, na2so4, naoh, pbs. Use empty arrays for
unmentioned categories. Do not infer N-15 labeling unless explicitly stated.

Text:
{text}
"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You extract validated scientific input as strict JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("The extraction model returned an empty response")
        payload = json.loads(content)
        payload["prediction_type"] = prediction_type.value
        missing: list[str] = []
        if payload.get("applied_potential") is None:
            missing.append("applied_potential")
        if not str(payload.get("electrocatalyst") or "").strip():
            missing.append("electrocatalyst")
        if not payload.get("elements"):
            missing.append("elements")
        if missing:
            raise MissingPredictionFields(missing)
        return PredictionInput.model_validate(payload)


class RecommendationTextExtractor:
    def __init__(self, api_key: str, base_url: str, model: str):
        if not api_key:
            raise ValueError("An API key is required for recommendation extraction")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def extract(
        self,
        text: str,
        default_allowed_elements: list[str],
    ) -> RecommendationRequest:
        if not text.strip():
            raise ValueError("Recommendation request cannot be empty")
        prompt = f"""
Extract deterministic catalyst-recommendation constraints from the request.
Return only JSON with these keys:
{{
  "mode": "hybrid",
  "allowed_elements": [],
  "forbidden_elements": [],
  "max_elements": 4,
  "morphologies": [],
  "ph_categories": [],
  "electrolytes": [],
  "potential_min": -0.8,
  "potential_max": 0.0,
  "potential_step": 0.1,
  "generated_candidate_limit": 500,
  "result_limit": 20
}}
Only include constraints explicitly stated by the user. Use empty arrays for
unmentioned category filters. Do not invent elements or experimental conditions.
Allowed recommendation modes are known, explore, and hybrid.

Request:
{text}
"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You extract validated recommendation constraints as strict JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("The recommendation model returned an empty response")
        payload = json.loads(content)
        assumptions: dict[str, object] = {
            "mode": "hybrid",
            "allowed_elements": default_allowed_elements,
            "forbidden_elements": [],
            "max_elements": 4,
            "morphologies": [],
            "ph_categories": [],
            "electrolytes": [],
            "potential_min": -0.8,
            "potential_max": 0.0,
            "potential_step": 0.1,
            "generated_candidate_limit": 500,
            "result_limit": 20,
        }
        assumptions.update({key: value for key, value in payload.items() if value is not None})
        normalized_text = text.casefold()
        explicit_modes = {
            "known": any(term in normalized_text for term in ("known", "已知")),
            "explore": any(
                term in normalized_text for term in ("explore", "exploratory", "探索")
            ),
            "hybrid": any(term in normalized_text for term in ("hybrid", "混合")),
        }
        selected_modes = [mode for mode, selected in explicit_modes.items() if selected]
        if len(selected_modes) == 1:
            assumptions["mode"] = selected_modes[0]
        if not assumptions["allowed_elements"]:
            assumptions["allowed_elements"] = default_allowed_elements
        return RecommendationRequest.model_validate(assumptions)
