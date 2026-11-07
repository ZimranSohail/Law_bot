import os
from dataclasses import dataclass
from urllib.parse import urlparse

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)


def _temp():
    v = os.getenv("TEMPERATURE", "").strip().lower()
    return None if v in ("", "none") else float(v)


@dataclass(frozen=True)
class Settings:
    endpoint: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    api_key: str = os.getenv("AZURE_OPENAI_API_KEY", "")
    deployment: str = os.getenv("AZURE_OPENAI_DEPLOYMENT", "")   # the one line to swap models
    temperature: float | None = _temp()
    max_output_tokens: int = int(os.getenv("MAX_OUTPUT_TOKENS", "1200"))


def make_base_url(endpoint: str) -> str:
    """Keep only scheme + host, then append /openai/v1/ (tolerates a pasted path)."""
    p = urlparse(endpoint.strip())
    return f"{p.scheme}://{p.netloc}/openai/v1/"


settings = Settings()

_missing = [k for k, v in {"AZURE_OPENAI_ENDPOINT": settings.endpoint,
                           "AZURE_OPENAI_API_KEY": settings.api_key,
                           "AZURE_OPENAI_DEPLOYMENT": settings.deployment}.items() if not v]
if _missing:
    raise RuntimeError(f"Missing in .env: {', '.join(_missing)}")

client = OpenAI(base_url=make_base_url(settings.endpoint), api_key=settings.api_key, max_retries=8)