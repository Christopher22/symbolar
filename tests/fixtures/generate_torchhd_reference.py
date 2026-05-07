#!/usr/bin/env python3
"""Generate TorchHD compatibility fixtures for Rust integration tests."""

from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass
import dataclasses

class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)


@dataclass
class Fixture:
    v1: list[float]
    v2: list[float]
    bundled: list[float]
    bound: list[float]
    permuted: list[float]
    similarity: float

    @classmethod
    def calculate(cls, v1: "torchhd.VSATensor", v2: "torchhd.VSATensor") -> Fixture:
        return cls(
            v1=v1.tolist(),
            v2=v2.tolist(),
            bundled=v1.bundle(v2).tolist(),
            bound=v1.bind(v2).tolist(),
            permuted=v1.permute(3).tolist(),
            similarity=v1.cosine_similarity(v2).item(),
        )

def main() -> None:
    try:
        import torch
        import torchhd
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency. Install torch and torchhd first, for example with:\n"
            "  python3 -m pip install torch torchhd"
        ) from exc

    torch.manual_seed(0)

    # Load input data from JSON file.
    input_path = Path(__file__).parent / "input.json"
    with input_path.open() as f:
        input_data = json.load(f)

    # Generate reference data for BSC and MAP architectures.
    reference_data = {}
    for arch_name, vectors in input_data.items():
        if arch_name == "BSC":
            v1 = torchhd.BSCTensor(torch.tensor(vectors["v1"]))
            v2 = torchhd.BSCTensor(torch.tensor(vectors["v2"]))
        elif arch_name == "MAP":
            v1 = torchhd.MAPTensor(torch.tensor(vectors["v1"]))
            v2 = torchhd.MAPTensor(torch.tensor(vectors["v2"]))
        else:
            raise ValueError(f"Unknown architecture: {arch_name}")
        reference_data[arch_name] = Fixture.calculate(v1, v2)

    # Save reference data to JSON file.
    reference_path = Path(__file__).parent / "reference.json"
    with reference_path.open("w") as f:
        json.dump(reference_data, f, cls=EnhancedJSONEncoder)

if __name__ == "__main__":
    main()
