#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
import subprocess
import time


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark embedding throughput by timing etl.embed_pgvector")
    ap.add_argument("--batch", type=int, default=1500)
    ap.add_argument("--encode-batch", type=int, default=512)
    ap.add_argument("--model", default=os.getenv("RAG_EMBED_MODEL_PATH", "sentence-transformers/all-MiniLM-L6-v2"))
    ap.add_argument("--normalize", action="store_true")
    ap.add_argument("--only-missing", action="store_true")
    ap.add_argument("--index", choices=["none", "hnsw", "ivfflat"], default="none")
    args = ap.parse_args()

    cmd = [
        "python",
        "-m",
        "etl.embed_pgvector",
        "--batch",
        str(args.batch),
        "--encode-batch",
        str(args.encode_batch),
        "--model",
        args.model,
        "--index",
        args.index,
    ]
    if args.normalize:
        cmd.append("--normalize")
    if args.only_missing:
        cmd.append("--only-missing")

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0

    print("=== bench_embed ===")
    print(f"command={' '.join(cmd)}")
    print(f"elapsed_seconds={elapsed:.2f}")
    print(f"exit_code={proc.returncode}")
    if proc.stdout:
        print("--- stdout (tail) ---")
        print("\n".join(proc.stdout.strip().splitlines()[-20:]))
    if proc.stderr:
        print("--- stderr (tail) ---")
        print("\n".join(proc.stderr.strip().splitlines()[-20:]))


if __name__ == "__main__":
    main()
