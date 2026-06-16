"""Export BAAI/bge-small-en-v1.5 to ONNX for the slim Docker runtime."""

import shutil
import subprocess
from pathlib import Path


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

    target = out / "model.onnx"
    if onnx_files[0] != target:
        shutil.move(str(onnx_files[0]), target)

    print(f"Exported ONNX model to {target}")


if __name__ == "__main__":
    main()