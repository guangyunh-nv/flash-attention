"""CUDA-event benchmark for dense Blackwell preferred/fallback clusters."""

import argparse
import json
import os
import statistics
import subprocess
import threading
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0, help="Physical GPU index")
    parser.add_argument("--head-dim", type=int, choices=(128, 256), default=128)
    parser.add_argument("--seqlen", type=int, default=200000)
    parser.add_argument("--dtype", choices=("fp8", "bf16"), default="fp8")
    parser.add_argument("--mode", choices=("2cta", "8cta+2cta/2cta"), default="8cta+2cta/2cta")
    parser.add_argument("--kv-stages", type=int, help="D256 only; D128 sizes its pipeline automatically")
    parser.add_argument("--warmup", type=int, default=40)
    parser.add_argument("--repeats", type=int, default=80)
    parser.add_argument("--output", type=Path, help="Write event samples and frequency telemetry as JSON")
    args = parser.parse_args()
    if args.seqlen <= 0 or args.warmup < 0 or args.repeats < 1:
        parser.error("seqlen/repeats must be positive and warmup must be nonnegative")
    if args.kv_stages is not None and (args.head_dim != 256 or not 2 <= args.kv_stages <= 12):
        parser.error("--kv-stages requires D256 and a value between 2 and 12")

    # These specialization options are read at module import time.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["FA_FWD_CLUSTER_MODE"] = args.mode
    os.environ["FA_CLC"] = "0"
    os.environ["FA_DISABLE_2CTA"] = "0"
    if args.kv_stages is not None:
        os.environ["FA_HD256_KV_STAGE"] = str(args.kv_stages)

    import torch
    from flash_attn.cute.interface import _flash_attn_fwd

    if torch.cuda.get_device_capability() != (10, 3):
        parser.error("This benchmark targets the SM103 preferred-cluster specialization")
    dtype = torch.float8_e4m3fn if args.dtype == "fp8" else torch.bfloat16
    torch.manual_seed(0)
    shape = (1, args.seqlen, 24, args.head_dim)
    buffers = []
    for _ in range(2):
        q, k, v = [torch.randn(shape, device="cuda", dtype=torch.bfloat16).to(dtype) for _ in range(3)]
        out = torch.empty(shape, device="cuda", dtype=torch.bfloat16)
        lse = torch.empty((1, 24, args.seqlen), device="cuda", dtype=torch.float32)
        buffers.append((q, k, v, out, lse))

    def launch(index):
        q, k, v, out, lse = buffers[index % 2]
        _flash_attn_fwd(q, k, v, out=out, lse=lse, return_lse=True, num_splits=1)

    with torch.inference_mode():
        launch(0)
        launch(1)
        for i in range(args.warmup):
            launch(i)
        torch.cuda.synchronize()

        monitor = subprocess.Popen(
            ["nvidia-smi", "-i", str(args.gpu),
             "--query-gpu=clocks.sm,power.draw,temperature.gpu,utilization.gpu",
             "--format=csv,noheader,nounits", "--loop-ms=20"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1,
        )
        readings = []
        ready = threading.Event()

        def read_monitor():
            for line in monitor.stdout:
                try:
                    clock, power, temperature, utilization = map(float, line.split(","))
                except ValueError:
                    continue
                readings.append(dict(t=time.perf_counter(), sm_mhz=clock, power_w=power,
                                     temperature_c=temperature, utilization_pct=utilization))
                ready.set()

        reader = threading.Thread(target=read_monitor, daemon=True)
        reader.start()
        try:
            if not ready.wait(10):
                raise RuntimeError("nvidia-smi did not return frequency samples")
            starts = [torch.cuda.Event(enable_timing=True) for _ in range(args.repeats)]
            ends = [torch.cuda.Event(enable_timing=True) for _ in range(args.repeats)]
            begin = time.perf_counter()
            for i in range(args.repeats):
                starts[i].record()
                launch(i)
                ends[i].record()
            torch.cuda.synchronize()
            end = time.perf_counter()
        finally:
            monitor.terminate()
            monitor.wait(timeout=5)
            reader.join(timeout=1)

    samples = [start.elapsed_time(stop) for start, stop in zip(starts, ends)]
    telemetry = [dict(r, t=r["t"] - begin) for r in readings if begin <= r["t"] <= end]
    clocks = sorted(r["sm_mhz"] for r in telemetry if r["t"] >= 0.2 * (end - begin))
    result = dict(mode=args.mode, head_dim=args.head_dim, seqlen=args.seqlen, dtype=args.dtype,
                  gpu=args.gpu, warmup=args.warmup, samples_ms=samples,
                  median_ms=statistics.median(samples), frequency_samples=telemetry)
    result["tflops"] = 4 * 24 * args.seqlen ** 2 * args.head_dim / (result["median_ms"] * 1e9)
    if clocks:
        result.update(stable_sm_mhz=statistics.median(clocks),
                      sm_mhz_p10=clocks[round(0.1 * (len(clocks) - 1))],
                      sm_mhz_p90=clocks[round(0.9 * (len(clocks) - 1))])
    print(json.dumps({k: v for k, v in result.items() if k not in ("samples_ms", "frequency_samples")}, indent=2))
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
