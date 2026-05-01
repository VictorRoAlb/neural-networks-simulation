from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path


TARGETS = {
    15: "training_history_reference.png",
    54: "final_confusion_matrix.png",
}


def decode_png(payload: str, destination: Path) -> None:
    if isinstance(payload, list):
        payload = "".join(payload)
    destination.write_bytes(base64.b64decode(payload))


def extract_images(notebook_path: Path, output_dir: Path) -> list[Path]:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for cell_index, filename in TARGETS.items():
        if cell_index >= len(notebook.get("cells", [])):
            continue

        cell = notebook["cells"][cell_index]
        for output in cell.get("outputs", []):
            data = output.get("data") if isinstance(output, dict) else None
            if not data or "image/png" not in data:
                continue

            destination = output_dir / filename
            decode_png(data["image/png"], destination)
            written.append(destination)
            break

    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract representative PNG figures from a Jupyter notebook with embedded outputs."
    )
    parser.add_argument("notebook_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    written = extract_images(args.notebook_path, args.output_dir)
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
