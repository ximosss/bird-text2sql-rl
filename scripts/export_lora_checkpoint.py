#!/usr/bin/env python3
"""Recover a vLLM/PEFT LoRA adapter from a Prime-RL trainer checkpoint."""

from __future__ import annotations

import argparse
import json
import shutil
import tomllib
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.format_utils import (
    _EmptyStateDictLoadPlanner,
    _load_state_dict,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--reference-adapter", type=Path)
    source.add_argument("--training-config", type=Path)
    return parser.parse_args()


def adapter_config_from_training_config(path: Path) -> dict:
    config = tomllib.loads(path.read_text(encoding="utf-8"))
    model = config["model"]
    # Prime-RL v1 keeps the trainable model implementation and LoRA settings
    # under [trainer.model], while older project configs placed LoRA directly
    # under [model]. Accept both layouts so checkpoint recovery remains tied to
    # the resolved training config instead of a hand-written adapter schema.
    lora = model.get("lora") or config.get("trainer", {}).get("model", {}).get("lora")
    if not isinstance(lora, dict):
        raise KeyError("LoRA settings not found under [model.lora] or [trainer.model.lora]")
    return {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": model["name"],
        "r": int(lora["rank"]),
        "lora_alpha": float(lora["alpha"]),
        "lora_dropout": float(lora.get("dropout", 0.0)),
        "bias": "none",
        "target_modules": sorted(lora["target_modules"]),
        "modules_to_save": None,
    }


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    reference = args.reference_adapter.resolve() if args.reference_adapter else None
    output_weights = output / "adapter_model.safetensors"

    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    required_paths = [checkpoint / ".metadata"]
    if reference is not None:
        required_paths.extend(
            [reference / "adapter_config.json", reference / "adapter_model.safetensors"]
        )
    else:
        required_paths.append(args.training_config.resolve())
    for required in required_paths:
        if not required.is_file():
            raise FileNotFoundError(required)

    reader = FileSystemReader(checkpoint)
    metadata = reader.read_metadata()
    dcp_keys = {
        key
        for key in metadata.state_dict_metadata
        if key.startswith("app.model.") and ".lora_" in key
    }
    if not dcp_keys:
        raise RuntimeError(f"No LoRA model tensors found in {checkpoint}")

    state: dict[str, object] = {}
    _load_state_dict(
        state,
        storage_reader=reader,
        planner=_EmptyStateDictLoadPlanner(keys=dcp_keys),
        no_dist=True,
    )
    model_state = state["app"]["model"]  # type: ignore[index]
    adapter_state = {}
    for key, tensor in model_state.items():  # type: ignore[union-attr]
        if not key.endswith(".0"):
            raise RuntimeError(f"Unexpected Prime-RL LoRA key: {key}")
        adapter_state[f"{key[:-2]}.weight"] = tensor.contiguous()

    exported_shapes = {key: tuple(tensor.shape) for key, tensor in adapter_state.items()}
    if reference is not None:
        with safe_open(reference / "adapter_model.safetensors", framework="pt") as handle:
            reference_shapes = {key: tuple(handle.get_slice(key).get_shape()) for key in handle.keys()}
        if exported_shapes != reference_shapes:
            missing = sorted(set(reference_shapes) - set(exported_shapes))
            extra = sorted(set(exported_shapes) - set(reference_shapes))
            wrong_shape = sorted(
                key
                for key in set(reference_shapes) & set(exported_shapes)
                if reference_shapes[key] != exported_shapes[key]
            )
            raise RuntimeError(
                f"Adapter schema mismatch: missing={missing[:5]}, extra={extra[:5]}, "
                f"wrong_shape={wrong_shape[:5]}"
            )
        adapter_config = json.loads((reference / "adapter_config.json").read_text(encoding="utf-8"))
    else:
        adapter_config = adapter_config_from_training_config(args.training_config.resolve())
        rank = adapter_config["r"]
        a_keys = sorted(key for key in exported_shapes if key.endswith(".lora_A.weight"))
        if not a_keys:
            raise RuntimeError("Checkpoint contains no exported LoRA A tensors")
        for a_key in a_keys:
            b_key = a_key.replace(".lora_A.weight", ".lora_B.weight")
            if b_key not in exported_shapes:
                raise RuntimeError(f"Missing paired tensor: {b_key}")
            if exported_shapes[a_key][0] != rank or exported_shapes[b_key][1] != rank:
                raise RuntimeError(
                    f"LoRA rank mismatch for {a_key}: "
                    f"A={exported_shapes[a_key]} B={exported_shapes[b_key]} expected={rank}"
                )

    output.mkdir(parents=True)
    try:
        save_file(adapter_state, output_weights, metadata={"format": "pt"})
        if reference is not None:
            shutil.copy2(reference / "adapter_config.json", output / "adapter_config.json")
        else:
            (output / "adapter_config.json").write_text(
                json.dumps(adapter_config, indent=2) + "\n", encoding="utf-8"
            )
        manifest = {
            "source_checkpoint": str(checkpoint),
            "reference_adapter_schema": str(reference) if reference is not None else None,
            "training_config": str(args.training_config.resolve()) if args.training_config else None,
            "tensor_count": len(adapter_state),
        }
        (output / "export_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except Exception:
        if output_weights.exists():
            output_weights.unlink()
        raise

    print(f"Exported {len(adapter_state)} LoRA tensors to {output}")


if __name__ == "__main__":
    main()
