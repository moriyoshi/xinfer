"""Opt-in vLLM half of the Gemma 4 E4B NVFP4 GB10 comparison.

Run inside a vLLM image with the prepared checkpoint mounted at /model and
tokens-{40,2048,8192}.json in --input. See docs/gemma4-e4b-nvfp4-vllm-gb10.md.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

from vllm import LLM, SamplingParams


def generate(llm, ids, count):
    params = SamplingParams(
        max_tokens=count, temperature=0.0, ignore_eos=True, detokenize=False
    )
    started = time.perf_counter()
    result = llm.generate([{"prompt_token_ids": ids}], params, use_tqdm=False)[0]
    wall_ms = (time.perf_counter() - started) * 1000
    return wall_ms, list(result.outputs[0].token_ids), result.metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/model")
    parser.add_argument("--input", default="/bench")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--decode-steps", type=int, default=48)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.35)
    parser.add_argument("--eager", action="store_true")
    args = parser.parse_args()
    root = Path(args.input)
    prompts = {
        size: json.loads((root / f"tokens-{size}.json").read_text())
        for size in (40, 2048, 8192)
    }
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        dtype="bfloat16",
        quantization="modelopt_fp4",
        kv_cache_dtype="auto",
        tensor_parallel_size=1,
        max_model_len=8256,
        max_num_batched_tokens=8256,
        max_num_seqs=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=False,
        enable_chunked_prefill=True,
        language_model_only=True,
        enforce_eager=args.eager,
        disable_log_stats=False,
    )
    generate(llm, prompts[40], 1)

    rows = []
    for size, ids in prompts.items():
        for sample in range(args.samples):
            first_wall_ms, first, _ = generate(llm, ids, 1)
            full_wall_ms, generated, metrics = generate(llm, ids, args.decode_steps + 1)
            row = {
                "tokens": size,
                "sample": sample,
                "prefill_ms": first_wall_ms,
                "full_ms": full_wall_ms,
                "decode_post_first_wall_ms_per_step": (
                    full_wall_ms - metrics.first_token_latency * 1000
                )
                / args.decode_steps,
                "decode_engine_ms_per_step": (
                    metrics.last_token_ts - metrics.first_token_ts
                )
                * 1000
                / args.decode_steps,
                "first_token": first[0],
                "generated": generated,
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
            (root / "vllm-results.json").write_text(
                json.dumps({"args": vars(args), "rows": rows}, indent=2)
            )
    print(
        "medians",
        json.dumps(
            {
                size: {
                    "prefill_ms": statistics.median(
                        row["prefill_ms"] for row in rows if row["tokens"] == size
                    ),
                    "decode_ms_per_step": statistics.median(
                        row["decode_post_first_wall_ms_per_step"]
                        for row in rows
                        if row["tokens"] == size
                    ),
                }
                for size in prompts
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
