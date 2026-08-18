from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

# Safer setting under Windows / multiprocessing environments
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


@dataclass
class VLLMEngineConfig:
    model_name_or_path: str = "Qwen/Qwen3-VL-8B-Instruct"
    tensor_parallel_size: int = 2
    gpu_memory_utilization: float = 0.92
    max_num_batched_tokens: int = 16384
    max_model_len: int = 16384
    trust_remote_code: bool = True
    allowed_local_media_path: str | None = None


@dataclass
class VLLMSamplingConfig:
    temperature: float = 0.0
    max_tokens: int = 1024
    top_k: int = -1
    top_p: float = 0.9


class Qwen3VLVLLM:
    """vLLM wrapper for Qwen3-VL-8B (text + image inputs only)."""

    def __init__(
        self,
        engine_config: VLLMEngineConfig | None = None,
        sampling_config: VLLMSamplingConfig | None = None,
    ) -> None:
        self.engine_config = engine_config or VLLMEngineConfig()
        self.sampling_config = sampling_config or VLLMSamplingConfig()

        model_path = Path(self.engine_config.model_name_or_path)
        if model_path.exists():
            model_id = str(model_path.resolve())
        else:
            model_id = self.engine_config.model_name_or_path

        self.processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=self.engine_config.trust_remote_code,
        )

        llm_kwargs = {
            "model": model_id,
            "tensor_parallel_size": self.engine_config.tensor_parallel_size,
            "gpu_memory_utilization": self.engine_config.gpu_memory_utilization,
            "max_num_batched_tokens": self.engine_config.max_num_batched_tokens,
            "max_model_len": self.engine_config.max_model_len,
            "trust_remote_code": self.engine_config.trust_remote_code,
            # Explicitly disallow video in a single request
            "limit_mm_per_prompt": {"video": 0},
        }
        # Newer vLLM versions expect a string when this key is present.
        if self.engine_config.allowed_local_media_path is not None:
            llm_kwargs["allowed_local_media_path"] = (
                self.engine_config.allowed_local_media_path
            )
        self.llm = LLM(**llm_kwargs)

    @staticmethod
    def _to_image_list(images: str | Sequence[str] | None) -> list[str]:
        if images is None:
            return []
        if isinstance(images, str):
            return [images]
        return [img for img in images if img]

    def _build_messages(
        self,
        prompt: str,
        images: str | Sequence[str] | None = None,
        system_prompt: str | None = None,
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}],
                }
            )

        user_content: list[dict[str, str]] = []
        for image in self._to_image_list(images):
            user_content.append({"type": "image", "image": image})
        user_content.append({"type": "text", "text": prompt})
        messages.append({"role": "user", "content": user_content})
        return messages

    @staticmethod
    def _assert_no_video(messages: Sequence[dict[str, Any]]) -> None:
        for message in messages:
            content = message.get("content", [])
            if not isinstance(content, list):
                continue
            for item in content:
                if isinstance(item, dict) and item.get("type") == "video":
                    raise ValueError(
                        "This wrapper only supports text+image input; video is not supported."
                    )

    def prepare_inputs_for_vllm(
        self, messages: Sequence[dict[str, Any]]
    ) -> dict[str, Any]:
        """Convert Qwen message format into inputs ready for vLLM inference."""
        self._assert_no_video(messages)
        text = self.processor.apply_chat_template(
            list(messages),
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            list(messages),
            image_patch_size=self.processor.image_processor.patch_size,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        if video_inputs is not None:
            raise ValueError(
                "Video input detected; this wrapper only supports text+image."
            )

        mm_data: dict[str, Any] = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs

        return {
            "prompt": text,
            "multi_modal_data": mm_data,
            "mm_processor_kwargs": video_kwargs or {},
        }

    def _build_sampling_params(
        self, sampling_overrides: dict[str, Any] | None = None
    ) -> SamplingParams:
        cfg = {
            "temperature": self.sampling_config.temperature,
            "max_tokens": self.sampling_config.max_tokens,
            "top_k": self.sampling_config.top_k,
            "top_p": self.sampling_config.top_p,
        }
        if sampling_overrides:
            cfg.update({k: v for k, v in sampling_overrides.items() if v is not None})
        return SamplingParams(**cfg)

    def chat(
        self,
        prompt: str,
        images: str | Sequence[str] | None = None,
        system_prompt: str | None = None,
        sampling_overrides: dict[str, Any] | None = None,
    ) -> str:
        """Single-turn inference: text + optional image(s) in, string out."""
        messages = self._build_messages(prompt, images, system_prompt)
        model_input = self.prepare_inputs_for_vllm(messages)
        try:
            outputs = self.llm.generate(
                [model_input],
                sampling_params=self._build_sampling_params(sampling_overrides),
                use_tqdm=False,
            )
        except TypeError:
            outputs = self.llm.generate(
                [model_input],
                sampling_params=self._build_sampling_params(sampling_overrides),
            )
        if not outputs or not outputs[0].outputs:
            return ""
        return outputs[0].outputs[0].text.strip()

    def batch_chat(
        self,
        requests: Sequence[dict[str, Any]],
        sampling_overrides: dict[str, Any] | None = None,
    ) -> list[str]:
        """
        Batch inference interface.
        Each item in requests looks like:
            {"prompt": "...", "images": "a.jpg" or ["a.jpg"], "system_prompt": "..."}
        """
        model_inputs: list[dict[str, Any]] = []
        for req in requests:
            messages = self._build_messages(
                prompt=req["prompt"],
                images=req.get("images"),
                system_prompt=req.get("system_prompt"),
            )
            model_inputs.append(self.prepare_inputs_for_vllm(messages))

        try:
            outputs = self.llm.generate(
                model_inputs,
                sampling_params=self._build_sampling_params(sampling_overrides),
                use_tqdm=False,
            )
        except TypeError:
            outputs = self.llm.generate(
                model_inputs,
                sampling_params=self._build_sampling_params(sampling_overrides),
            )

        results: list[str] = []
        for output in outputs:
            if output.outputs:
                results.append(output.outputs[0].text.strip())
            else:
                results.append("")
        return results


_QWEN_VL_CLIENT_CACHE: dict[str, Qwen3VLVLLM] = {}


def get_or_create_qwen_vl_client(
    model_name_or_path: str,
    tensor_parallel_size: int = 2,
    gpu_memory_utilization: float = 0.92,
    max_num_batched_tokens: int = 4096,
    max_model_len: int = 4096,
    allowed_local_media_path: str | None = None,
) -> Qwen3VLVLLM:
    """Process-wide singleton cache to avoid multiple vLLM engines."""
    cache_key = (
        f"{model_name_or_path}|tp={tensor_parallel_size}|gpu={gpu_memory_utilization}|"
        f"maxbt={max_num_batched_tokens}|maxlen={max_model_len}|media={allowed_local_media_path}"
    )
    if cache_key not in _QWEN_VL_CLIENT_CACHE:
        _QWEN_VL_CLIENT_CACHE[cache_key] = Qwen3VLVLLM(
            engine_config=VLLMEngineConfig(
                model_name_or_path=model_name_or_path,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                max_num_batched_tokens=max_num_batched_tokens,
                max_model_len=max_model_len,
                allowed_local_media_path=allowed_local_media_path,
            )
        )
    return _QWEN_VL_CLIENT_CACHE[cache_key]


if __name__ == "__main__":
    client = Qwen3VLVLLM(
        engine_config=VLLMEngineConfig(
            model_name_or_path="Qwen/Qwen3-VL-8B-Instruct",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.92,
            max_num_batched_tokens=4096,
            max_model_len=4096,
        )
    )

    answer = client.chat(
        prompt="Briefly summarize the core content of the image.",
        images="your/path/to/image.jpg",
        sampling_overrides={"max_tokens": 256},
    )
    print(answer)
