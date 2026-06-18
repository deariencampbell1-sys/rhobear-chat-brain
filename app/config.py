import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    # LLM provider chain
    minimax_api_key: str
    minimax_base_url: str
    minimax_fast_model: str
    minimax_thinker_model: str
    llama_upstream: str  # local llama.cpp fallback (corrected to 2019)
    # Cache + admin
    cache_threshold: float
    admin_token: str
    # Service
    port: int
    data_dir: Path
    seeds_path: Path
    embedding_model: str
    embedding_dim: int
    onnx_model_dir: Path | None

    @property
    def cache_db_path(self) -> Path:
        return self.data_dir / "cache.db"

    @property
    def requests_db_path(self) -> Path:
        return self.data_dir / "requests.db"

    @property
    def chat_completions_url(self) -> str:
        """Kept for backward compat with the healthz endpoint; uses the
        local llama fallback since that's what we want to report as
        'upstream' for health purposes."""
        base = self.llama_upstream.rstrip("/")
        return f"{base}/v1/chat/completions"


def load_settings() -> Settings:
    data_dir = Path(os.environ.get("DATA_DIR", "./data"))
    seeds_path = Path(os.environ.get("SEEDS_PATH", "./seeds/sales-faq.jsonl"))
    onnx_dir = os.environ.get("ONNX_MODEL_DIR", "").strip()
    return Settings(
        # Provider chain
        minimax_api_key=os.environ.get("MINIMAX_API_KEY", ""),
        # Canonical MiniMax base URL — same default Apollo and Hermes use.
        # Trailing /v1 is NOT included; the MiniMaxProvider appends
        # /v1/text/chatcompletion_v2 itself.
        minimax_base_url=os.environ.get(
            "MINIMAX_BASE_URL", "https://api.minimax.io"
        ),
        minimax_fast_model=os.environ.get(
            "MINIMAX_FAST_MODEL", "MiniMax-M2.7-highspeed"
        ),
        minimax_thinker_model=os.environ.get(
            "MINIMAX_THINKER_MODEL", "MiniMax-M3"
        ),
        llama_upstream=os.environ.get("LLAMA_UPSTREAM", "http://localhost:2019"),
        # Cache + admin
        cache_threshold=float(os.environ.get("CACHE_THRESHOLD", "0.86")),
        admin_token=os.environ.get("ADMIN_TOKEN", ""),
        # Service
        port=int(os.environ.get("PORT", "8000")),
        data_dir=data_dir,
        seeds_path=seeds_path,
        embedding_model=os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
        embedding_dim=int(os.environ.get("EMBEDDING_DIM", "384")),
        onnx_model_dir=Path(onnx_dir) if onnx_dir else None,
    )
