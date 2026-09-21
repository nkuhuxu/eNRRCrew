from getpass import getpass

from enrrcrew.agents import CsvAnalysisAgent
from enrrcrew.config import AppSettings
from enrrcrew.schemas import PredictionType
from enrrcrew.services import PredictionTextExtractor, RagService, validate_code

settings = AppSettings.from_environment()
api_key = getpass("Session API key: ").strip()
if not api_key:
    raise SystemExit("An API key is required")
base_url = "https://api.chatanywhere.tech/v1"

extracted = PredictionTextExtractor(api_key, base_url, settings.chat_model).extract(
    "Fe-N-C porous nanosheets containing Fe, N and C were tested at -0.45 V "
    "in neutral PBS with explicit N-15 isotope labeling.",
    PredictionType.YIELD,
)
assert extracted.prediction_type is PredictionType.YIELD
assert extracted.electrocatalyst
assert 1 <= len(extracted.elements) <= 7
print("Natural-language extraction passed")

dataset = settings.input_dir / "data_include_morphology_electrocatalyst.csv"
code = CsvAnalysisAgent(api_key, base_url, settings.chat_model).generate_code(
    "Print the number of rows in the dataset.", dataset, "dataset.csv"
)
validate_code(code)
assert "/data/dataset.csv" in code
print("CSV analysis code generation passed")

answer = RagService(
    settings.rag_service_url,
    settings.rag_service_token,
    api_key,
    base_url,
).search(
    "What catalyst characteristics are associated with improved nitrogen reduction?",
    mode="local",
    community=0,
    response_type="single paragraph",
)
assert answer.strip()
print(f"GraphRAG local search passed ({len(answer)} characters)")
