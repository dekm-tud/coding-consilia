"""
Shared vLLM loading/inference helpers for the GPU-server scripts in this
directory (qwen_process.py, name_clusters.py). Not part of the main
pipeline's requirements.txt — install on the GPU server separately:

    pip install vllm

Default model: Qwen/Qwen3.6-27B (https://huggingface.co/Qwen/Qwen3.6-27B),
overridable via --model on either script.
"""

from __future__ import annotations

import logging
from typing import Optional

log = logging.getLogger(__name__)

DEFAULT_MODEL = "Qwen/Qwen3.6-27B"


def load_llm(
    model_id: str,
    tensor_parallel_size: int = 1,
    dtype: str = "bfloat16",
    gpu_memory_utilization: float = 0.9,
    max_model_len: Optional[int] = None,
    max_num_seqs: Optional[int] = 256,
):
    """
    max_num_seqs caps how many sequences vLLM schedules concurrently. Hybrid
    Mamba/attention models (e.g. Qwen3.6) need one Mamba cache block per
    concurrent decode sequence — vLLM's default max_num_seqs (1024) can
    exceed the cache blocks your GPU/gpu_memory_utilization actually allow,
    which fails CUDA graph capture at startup with an error like:
        "max_num_seqs (1024) exceeds available Mamba cache blocks (614)"
    If you hit that, either lower --max-num-seqs below the reported cache
    block count, or raise --gpu-memory-utilization to get more blocks.
    """
    from vllm import LLM

    log.info("Loading %s via vLLM (tp=%d, dtype=%s, max_num_seqs=%s) …",
              model_id, tensor_parallel_size, dtype, max_num_seqs)
    llm = LLM(
        model=model_id,
        tensor_parallel_size=tensor_parallel_size,
        dtype=dtype,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
    )
    log.info("Model loaded.")
    return llm


def chat_batch(llm, conversations: list[list[dict]], max_tokens: list[int]) -> list[str]:
    """Run a batch of independent chat conversations through vLLM in one call.

    `conversations[i]` is a list of {"role", "content"} messages for request i;
    `max_tokens[i]` is that request's generation budget. Returns the generated
    text for each request, in the same order (greedy decoding — temperature 0,
    since this is extraction/translation, not creative generation).

    Thinking mode is explicitly disabled (enable_thinking=False): Qwen3-family
    models emit a <think>...</think> reasoning block before the answer when
    thinking is on, which our callers don't strip — it would land inside
    translation output and break the keyword-JSON parsing — and it isn't
    needed for straightforward translation/extraction anyway.
    """
    from vllm import SamplingParams

    if not conversations:
        return []
    sampling = [SamplingParams(temperature=0, max_tokens=n) for n in max_tokens]
    outputs = llm.chat(
        conversations,
        sampling_params=sampling,
        use_tqdm=len(conversations) > 20,
        chat_template_kwargs={"enable_thinking": False},
    )
    return [out.outputs[0].text.strip() for out in outputs]
