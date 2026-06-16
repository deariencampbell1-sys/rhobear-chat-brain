"""Export BAAI/bge-small-en-v1.5 to ONNX for the slim Docker runtime."""

import shutil
import subprocess
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic


def main() -> None:
    out = Path("/build/onnx-model")
    out.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            "optimum-cli",
            "export",
            "onnx",
            "--model",
            "BAAI/bge-small-en-v1.5",
            "--task",
            "feature-extraction",
            str(out),
        ],
        check=True,
    )

    onnx_files = list(out.glob("*.onnx"))
    if not onnx_files:
        raise RuntimeError(f"No ONNX files exported to {out}")

    raw_model = out / "model.raw.onnx"
    shutil.move(str(onnx_files[0]), raw_model)

    target = out / "model.onnx"
    quantize_dynamic(
        str(raw_model),
        str(target),
        weight_type=QuantType.QInt8,
    )
    raw_model.unlink(missing_ok=True)

    print(f"Exported quantized ONNX model to {target}")


if __name__ == "__main__":
    main()