#!/usr/bin/env python3
"""Generate TorchHD compatibility fixtures for Rust integration tests."""

from __future__ import annotations

import json
from pathlib import Path


def bools_to_numeric_list(values) -> list[float]:
    return [1.0 if bool(v) else 0.0 for v in values]


def numeric_list(values) -> list[float]:
    return [float(v) for v in values]


def make_vsa_tensor(torch, tensor_cls, values, dtype):
    return torch.tensor(values, dtype=dtype).as_subclass(tensor_cls)


def make_architecture_reference(bundle_a, bundle_b, bind_a, bind_b, permute_by: int, normalize_bundle: bool, to_numeric):
    bundled = bundle_a.bundle(bundle_b)
    if normalize_bundle:
        bundled = bundled.normalize()

    return {
        "bundle_input_a": to_numeric(bundle_a),
        "bundle_input_b": to_numeric(bundle_b),
        "bind_input_a": to_numeric(bind_a),
        "bind_input_b": to_numeric(bind_b),
        "bundle": to_numeric(bundled),
        "bind": to_numeric(bind_a.bind(bind_b)),
        "permute_by": permute_by,
        "permute": to_numeric(bind_a.permute(shifts=permute_by)),
        "inverse": to_numeric(bind_a.inverse()),
        "cosine_similarity": float(bind_a.cosine_similarity(bind_b).item()),
    }


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

    bsc_bundle_a = [1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    bsc_bundle_b = [1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    bsc_bind_a = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    bsc_bind_b = [1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]
    bsc_shift = 3

    bsc_a_bundle = make_vsa_tensor(torch, torchhd.BSCTensor, bsc_bundle_a, torch.bool)
    bsc_b_bundle = make_vsa_tensor(torch, torchhd.BSCTensor, bsc_bundle_b, torch.bool)
    bsc_a_bind = make_vsa_tensor(torch, torchhd.BSCTensor, bsc_bind_a, torch.bool)
    bsc_b_bind = make_vsa_tensor(torch, torchhd.BSCTensor, bsc_bind_b, torch.bool)

    bsc = make_architecture_reference(
        bundle_a=bsc_a_bundle,
        bundle_b=bsc_b_bundle,
        bind_a=bsc_a_bind,
        bind_b=bsc_b_bind,
        permute_by=bsc_shift,
        normalize_bundle=False,
        to_numeric=bools_to_numeric_list,
    )

    map_bundle_a = [1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0]
    map_bundle_b = [1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0]
    map_bind_a = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]
    map_bind_b = [1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0]
    map_shift = 3

    map_a_bundle = make_vsa_tensor(torch, torchhd.MAPTensor, map_bundle_a, torch.float32)
    map_b_bundle = make_vsa_tensor(torch, torchhd.MAPTensor, map_bundle_b, torch.float32)
    map_a_bind = make_vsa_tensor(torch, torchhd.MAPTensor, map_bind_a, torch.float32)
    map_b_bind = make_vsa_tensor(torch, torchhd.MAPTensor, map_bind_b, torch.float32)

    map_data = make_architecture_reference(
        bundle_a=map_a_bundle,
        bundle_b=map_b_bundle,
        bind_a=map_a_bind,
        bind_b=map_b_bind,
        permute_by=map_shift,
        normalize_bundle=True,
        to_numeric=numeric_list,
    )

    out = {
        "torchhd_version": torchhd.__version__,
        "architectures": {
            "bsc": bsc,
            "map": map_data,
        },
    }

    output_path = Path(__file__).resolve().parent / "torchhd_reference.json"
    output_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
