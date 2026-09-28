"""OpenVLA inference helpers used by CAPE-TTA experiments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor


@dataclass
class OpenVLAHandle:
    model: object
    processor: object
    checkpoint: str
    unnorm_key: str
    device: torch.device


def load_openvla(checkpoint: str, suite: str, *, use_flash_attention: bool = True) -> OpenVLAHandle:
    if not torch.cuda.is_available():
        raise RuntimeError("The OpenVLA experiment requires a CUDA GPU.")
    device = torch.device("cuda:0")
    attn = "flash_attention_2" if use_flash_attention else "sdpa"
    model = AutoModelForVision2Seq.from_pretrained(
        checkpoint,
        attn_implementation=attn,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)
    processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)

    key = suite
    if key not in model.norm_stats and f"{key}_no_noops" in model.norm_stats:
        key = f"{key}_no_noops"
    if key not in model.norm_stats:
        raise RuntimeError(f"No norm_stats for {suite}; available={list(model.norm_stats)}")
    return OpenVLAHandle(model, processor, checkpoint, key, device)


def official_prompt(checkpoint: str, task: str) -> str:
    if "openvla-v01" in checkpoint:
        system = (
            "A chat between a curious user and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the user's questions."
        )
        return f"{system} USER: What action should the robot take to {task.lower()}? ASSISTANT:"
    return f"In: What action should the robot take to {task.lower()}?\nOut:"


def preprocess_libero_image(image: np.ndarray, *, center_crop: bool = True) -> np.ndarray:
    """Match OpenVLA's official LIBERO image preprocessing."""
    import tensorflow as tf

    img = np.asarray(image)[::-1, ::-1]
    img = tf.image.encode_jpeg(img)
    img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)
    img = tf.image.resize(img, (224, 224), method="lanczos3", antialias=True)
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)

    if center_crop:
        x = tf.image.convert_image_dtype(img, tf.float32)
        scale = tf.sqrt(tf.constant(0.9, dtype=tf.float32))
        offset = (1.0 - scale) / 2.0
        boxes = tf.reshape(tf.stack([offset, offset, offset + scale, offset + scale]), (1, 4))
        x = tf.image.crop_and_resize(tf.expand_dims(x, 0), boxes, tf.constant([0]), (224, 224))[0]
        x = tf.clip_by_value(x, 0.0, 1.0)
        img = tf.image.convert_image_dtype(x, tf.uint8, saturate=True)

    return img.numpy()


def prepare_inputs(handle: OpenVLAHandle, image: np.ndarray, task: str):
    image = preprocess_libero_image(image, center_crop=True)
    pil = Image.fromarray(image).convert("RGB")
    inputs = handle.processor(official_prompt(handle.checkpoint, task), pil)
    return inputs.to(handle.device, dtype=torch.bfloat16)


def _decode_action(handle: OpenVLAHandle, token_ids: np.ndarray) -> np.ndarray:
    model = handle.model
    discretized = model.vocab_size - token_ids
    discretized = np.clip(discretized - 1, 0, model.bin_centers.shape[0] - 1)
    normalized = model.bin_centers[discretized]

    stats = model.get_action_stats(handle.unnorm_key)
    mask = stats.get("mask", np.ones_like(stats["q01"], dtype=bool))
    high = np.asarray(stats["q99"])
    low = np.asarray(stats["q01"])
    return np.where(mask, 0.5 * (normalized + 1) * (high - low) + low, normalized)


@torch.inference_mode()
def generate_action(
    handle: OpenVLAHandle,
    image: np.ndarray,
    task: str,
    *,
    do_sample: bool,
    temperature: float = 0.7,
    top_p: float = 0.95,
    seed: Optional[int] = None,
    restrict_to_action_tokens: bool = True,
) -> tuple[np.ndarray, float, list[int]]:
    """Generate one 7-DoF action and its sequence log-probability."""
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    inputs = prepare_inputs(handle, image, task)
    input_ids = inputs["input_ids"]
    if not torch.all(input_ids[:, -1] == 29871):
        empty = torch.tensor([[29871]], device=input_ids.device, dtype=input_ids.dtype)
        input_ids = torch.cat((input_ids, empty), dim=1)

    kwargs = dict(inputs)
    kwargs["input_ids"] = input_ids
    action_dim = handle.model.get_action_dim(handle.unnorm_key)

    generation_kwargs = {
        "max_new_tokens": action_dim,
        "do_sample": do_sample,
        "return_dict_in_generate": True,
        "output_scores": True,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p

    if restrict_to_action_tokens:
        low = int(handle.model.vocab_size - 256)
        high = int(handle.model.vocab_size)
        allowed = list(range(low, high))
        generation_kwargs["prefix_allowed_tokens_fn"] = lambda _batch, _ids: allowed

    out = handle.model.generate(**kwargs, **generation_kwargs)
    ids = out.sequences[0, -action_dim:]
    logprob = 0.0
    for score, tok in zip(out.scores, ids):
        logprob += float(torch.log_softmax(score[0], dim=-1)[tok].item())

    ids_np = ids.detach().cpu().numpy()
    return _decode_action(handle, ids_np), logprob, ids_np.astype(int).tolist()


def to_libero_action(action: np.ndarray) -> np.ndarray:
    """Match the official OpenVLA LIBERO gripper post-processing."""
    action = np.asarray(action, dtype=np.float32).copy()
    action[-1] = 2.0 * action[-1] - 1.0
    action[-1] = np.sign(action[-1])
    action[-1] *= -1.0
    return action
