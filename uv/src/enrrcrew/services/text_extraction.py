from __future__ import annotations

import json

from openai import OpenAI

from enrrcrew.schemas import PredictionInput, PredictionType


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
  "applied_potential": -0.3,
  "electrocatalyst": "catalyst name",
  "elements": ["Co", "Mo", "C", "N"],
  "morphology": "porous nanofibers",
  "ph_categories": ["acidic"],
  "electrolytes": ["pbs"],
  "n15_labeling": false
}}
Use only chemical element symbols in elements. Applied potential must be numeric in volts.
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
        return PredictionInput.model_validate(payload)

