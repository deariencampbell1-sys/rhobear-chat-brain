import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    llama_upstream: str
    cache_threshold: float
    admin_token: str
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
        base = self.llama_upstream.rstrip("/")
        return f"{base}/v1/chat/completions"


def load_settings() -> Settings:
    data_dir = Path(os.environ.get("DATA_DIR", "./data"))
    seeds_path = Path(os.environ.get("SEEDS_PATH", "./seeds/sales-faq.jsonl"))
    return Settings(
        llama_upstream=os.environ.get("LLAMA_UPSTREAM", "http://localhost:8080"),
        cache_threshold=float(os.environ.get("CACHE_THRESHOLD", "0.86")),
        admin_token=os.environ.get("ADMIN_TOKEN", ""),
        port=int(os.environ.get("PORT", "8000")),
        data_dir=data_dir,
        seeds_path=seeds_path,
        embedding_model=os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
        embedding_dim=int(os.environ.get("EMBEDDING_DIM", "384")),
        onnx_model_dir=(
            Path(onnx_dir)
            if (onnx_dir := os.environ.get("ONNX_MODEL_DIR", "")).strip()
            else None
        ),
    )