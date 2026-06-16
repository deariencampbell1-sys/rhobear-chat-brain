"""Export BAAI/bge-small-en-v1.5 to ONNX for the slim Docker runtime."""

from pathlib import Path

from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer


def main() -> None:
    model_name = "BAAI/bge-small-en-v1.5"
    out = Path("/build/onnx-model")
    out.mkdir(parents=True, exist_ok=True)

    model = ORTModelForFeatureExtraction.from_pretrained(model_name, export=True)
    model.save_pretrained(out)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(out)

    onnx_files = list(out.glob("*.onnx"))
    if not onnx_files:
        raise RuntimeError(f"No ONNX files exported to {out}")
    onnx_files[0].rename(out / "model.onnx")
    print(f"Exported ONNX model to {out / 'model.onnx'}")


if __name__ == "__main__":
    main()