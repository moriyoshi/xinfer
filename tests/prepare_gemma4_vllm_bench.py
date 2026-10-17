"""Prepare matching token IDs and BF16-KV ModelOpt metadata for the vLLM probe.

The output model directory contains symlinks to /weights, which must be the
read-only mount of the original checkpoint inside the vLLM container.
"""

import argparse
import hashlib
import json
from pathlib import Path


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")
    print(path, hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-tokens", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.checkpoint.resolve() == args.output.resolve():
        raise ValueError("output must be separate from checkpoint")
    ids = json.loads(args.base_tokens.read_text())
    if len(ids) != 40:
        raise ValueError("expected the 40-token Gemma 4 reference prompt")
    args.output.mkdir(parents=True, exist_ok=True)
    head, body, tail = ids[:15], ids[15:35], ids[35:]
    for size in (40, 2048, 8192):
        count = size - len(head) - len(tail)
        repeats = (count + len(body) - 1) // len(body)
        tokens = ids if size == 40 else head + (body * repeats)[:count] + tail
        path = args.output / f"tokens-{size}.json"
        path.write_text(json.dumps(tokens, separators=(",", ":")))
        print(path, hashlib.sha256(path.read_bytes()).hexdigest())

    model = args.output / "model"
    model.mkdir(exist_ok=True)
    for original in args.checkpoint.iterdir():
        if not original.is_file() or original.name in ("config.json", "hf_quant_config.json"):
            continue
        link = model / original.name
        target = Path("/weights") / original.name
        if link.exists() or link.is_symlink():
            if not link.is_symlink() or link.readlink() != target:
                raise ValueError(f"unexpected existing file: {link}")
        else:
            link.symlink_to(target)

    config = json.loads((args.checkpoint / "config.json").read_text())
    config["quantization_config"]["kv_cache_scheme"] = None
    save_json(model / "config.json", config)
    quant_config = json.loads((args.checkpoint / "hf_quant_config.json").read_text())
    quant_config["quantization"]["kv_cache_quant_algo"] = None
    save_json(model / "hf_quant_config.json", quant_config)


if __name__ == "__main__":
    main()
