from __future__ import annotations

import csv
from pathlib import Path

from openai import OpenAI


class CsvAnalysisAgent:
    def __init__(self, api_key: str, base_url: str, model: str):
        if not api_key:
            raise ValueError("An API key is required to generate CSV analysis code")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    @staticmethod
    def read_columns(csv_path: Path) -> list[str]:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            return next(reader)

    def generate_code(self, question: str, csv_path: Path, container_name: str) -> str:
        if not question.strip():
            raise ValueError("CSV question cannot be empty")
        columns = self.read_columns(csv_path)
        prompt = f"""
Write one complete Python script that answers the question using pandas.
The CSV is read-only at /data/{container_name}.
Columns: {columns}
Question: {question}

Rules:
- Return Python source only, with no Markdown fence.
- Allowed imports: pandas, numpy, matplotlib, seaborn, math, statistics, json, csv.
- Print the concise answer to stdout.
- Save charts under /output using a descriptive .png filename.
- Never use network access, subprocesses, environment variables, or arbitrary file access.
"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You write small, deterministic, sandboxed data-analysis scripts.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        code = response.choices[0].message.content or ""
        code = code.strip()
        if code.startswith("```python"):
            code = code.removeprefix("```python").removesuffix("```").strip()
        return code

