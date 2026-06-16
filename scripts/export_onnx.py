"""Export BAAI/bge-small-en-v1.5 to ONNX for the slim Docker runtime."""

from pathlib import Path

from sentence_transformers import SentenceTransformer


def main() -> None:
    out = Path("/build/onnx-model")
    out.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
    model.save_pretrained(
        str(out),
        create_model_card=False,
        backend="onnx",
        model_kwargs={"file_name": "model.onnx"},
    )
    print(f"Exported ONNX model to {out}")


if __name__ == "__main__":
    main()