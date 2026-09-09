#!/usr/bin/env python3
"""Merge the selected cumulative Stage-B LoRA into a Qwen3-4B HF model."""

from __future__ import annotations

import argparse
import hashlib
import json
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path, default=Path("/data/qwen3-4b-instruct-2507"))
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-shard-size", default="5GB")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def validate_layout(
    base_model: Path,
    adapter_weights: Path,
    adapter_config: Path,
) -> tuple[dict, list[str], float]:
    config = json.loads(adapter_config.read_text(encoding="utf-8"))
    if config.get("bias", "none") != "none" or config.get("modules_to_save"):
        raise ValueError("only bias=none LoRA adapters without modules_to_save are supported")
    rank = int(config["r"])
    scale = float(config["lora_alpha"]) / rank

    index_path = base_model / "model.safetensors.index.json"
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    elif (base_model / "model.safetensors").is_file():
        with safe_open(base_model / "model.safetensors", framework="pt", device="cpu") as reader:
            weight_map = {key: "model.safetensors" for key in reader.keys()}
    else:
        raise FileNotFoundError("base model has no safetensors weights or index")

    with safe_open(adapter_weights, framework="pt", device="cpu") as adapter_reader:
        keys = set(adapter_reader.keys())
        unexpected = sorted(
            key for key in keys if not key.endswith((".lora_A.weight", ".lora_B.weight"))
        )
        if unexpected:
            raise ValueError(f"unsupported adapter tensors: {unexpected[:3]}")
        a_keys = sorted(key for key in keys if key.endswith(".lora_A.weight"))
        if not a_keys:
            raise ValueError("adapter contains no LoRA A tensors")
        with ExitStack() as stack:
            readers = {
                shard: stack.enter_context(
                    safe_open(base_model / shard, framework="pt", device="cpu")
                )
                for shard in sorted(set(weight_map.values()))
            }
            for a_key in a_keys:
                b_key = a_key.replace(".lora_A.weight", ".lora_B.weight")
                parameter_name = a_key.replace(".lora_A.weight", ".weight")
                if b_key not in keys:
                    raise ValueError(f"missing paired tensor: {b_key}")
                if parameter_name not in weight_map:
                    raise ValueError(f"base parameter is missing: {parameter_name}")
                a_shape = tuple(adapter_reader.get_slice(a_key).get_shape())
                b_shape = tuple(adapter_reader.get_slice(b_key).get_shape())
                base_shape = tuple(
                    readers[weight_map[parameter_name]].get_slice(parameter_name).get_shape()
                )
                if a_shape[0] != rank or b_shape[1] != rank or (b_shape[0], a_shape[1]) != base_shape:
                    raise ValueError(
                        f"shape mismatch for {parameter_name}: A={a_shape} B={b_shape} base={base_shape}"
                    )
    return config, a_keys, scale


def main() -> None:
    args = parse_args()
    adapter_weights = args.adapter / "adapter_model.safetensors"
    adapter_config = args.adapter / "adapter_config.json"
    for required in (args.base_model / "config.json", adapter_weights, adapter_config):
        if not required.is_file():
            raise FileNotFoundError(required)
    config, validated_a_keys, scale = validate_layout(
        args.base_model, adapter_weights, adapter_config
    )
    if args.validate_only:
        print(json.dumps({"validated_modules": len(validated_a_keys), "lora_scale": scale}))
        return
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rank = int(config["r"])

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    merged_modules = 0
    with safe_open(adapter_weights, framework="pt", device="cpu") as tensors:
        keys = set(tensors.keys())
        a_keys = validated_a_keys
        with torch.no_grad():
            for a_key in a_keys:
                b_key = a_key.replace(".lora_A.weight", ".lora_B.weight")
                if b_key not in keys:
                    raise ValueError(f"missing paired tensor: {b_key}")
                parameter_name = a_key.replace(".lora_A.weight", ".weight")
                parameter = model.get_parameter(parameter_name)
                a = tensors.get_tensor(a_key)
                b = tensors.get_tensor(b_key)
                if a.shape[0] != rank or b.shape[1] != rank:
                    raise ValueError(
                        f"rank mismatch for {parameter_name}: A={tuple(a.shape)} B={tuple(b.shape)}"
                    )
                delta = torch.matmul(b.float(), a.float()).mul_(scale)
                if delta.shape != parameter.shape or not torch.isfinite(delta).all():
                    raise ValueError(
                        f"unsafe merge for {parameter_name}: delta={tuple(delta.shape)} "
                        f"parameter={tuple(parameter.shape)}"
                    )
                parameter.add_(delta.to(dtype=parameter.dtype))
                merged_modules += 1

    model.save_pretrained(
        args.output_dir,
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.save_pretrained(args.output_dir)

    provenance = {
        "contract_version": "bird-sft-merged-v2",
        "base_model": str(args.base_model.resolve()),
        "adapter": str(args.adapter.resolve()),
        "adapter_model_sha256": sha256(adapter_weights),
        "adapter_config_sha256": sha256(adapter_config),
        "dtype": "bfloat16",
        "merged_modules": merged_modules,
        "lora_scale": scale,
    }
    (args.output_dir / "bird_merge_manifest.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(provenance, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
