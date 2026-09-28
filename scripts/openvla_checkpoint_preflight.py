#!/usr/bin/env python3
import argparse
import json

from transformers import AutoConfig, AutoProcessor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="openvla/openvla-7b-finetuned-libero-spatial")
    p.add_argument("--suite", default="libero_spatial")
    args = p.parse_args()

    cfg = AutoConfig.from_pretrained(args.checkpoint, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.checkpoint, trust_remote_code=True)

    stats = getattr(cfg, "norm_stats", None)
    if not stats:
        raise RuntimeError("Checkpoint config does not expose norm_stats")

    key = args.suite
    if key not in stats and f"{key}_no_noops" in stats:
        key = f"{key}_no_noops"
    if key not in stats:
        raise RuntimeError(f"No action stats for {args.suite}; available={list(stats)}")

    action_stats = stats[key]["action"]
    q01 = action_stats["q01"]
    q99 = action_stats["q99"]
    payload = {
        "checkpoint": args.checkpoint,
        "model_type": getattr(cfg, "model_type", None),
        "processor_class": type(processor).__name__,
        "unnorm_key": key,
        "action_dim": len(q01),
        "q01": q01,
        "q99": q99,
    }
    if len(q01) != 7 or len(q99) != 7:
        raise RuntimeError(f"Expected 7-DoF action stats, got {len(q01)}")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
