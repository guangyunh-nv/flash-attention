# Preferred 8CTA / fallback 2CTA forward attention

This opt-in Blackwell forward path lets hardware place either eight-CTA or
two-CTA clusters in a single kernel launch. All matrix operations use 2CTA MMA.
The preferred cluster shares K/V loads across four MMA pairs; fallback clusters
load K/V for their own pair. Dense work launches a complete nonpersistent grid
and does not require Cluster Launch Control (CLC) work stealing.

The change is based on FlashAttention commit `a369df7` and was validated on
GB300 (SM103), CUDA 13, and CuTeDSL 4.7.1. It is an experimental, opt-in
specialization; it does not change the default forward mode.

## Enable

Install this checkout's CuTe package in an environment with a compatible
PyTorch/CUDA and CuTeDSL installation:

```bash
pip install -e flash_attn/cute
```

Set the mode before importing FlashAttention:

```bash
export FA_FWD_CLUSTER_MODE='8cta+2cta/2cta'
```

Use the usual `flash_attn.cute.flash_attn_func` API. `FA_FWD_CLUSTER_MODE=2cta`
selects the original dispatch policy. No separate 1CTA-MMA modes are included.
The existing dense CLC suppression is preserved: `FA_CLC=1` does not turn on
work stealing for these dense kernels.

The preferred path is selected only for the validated configuration family:

- SM103, batch size 1, equal query and KV sequence lengths.
- BF16, FP8 E4M3, or FP8 E5M2 inputs.
- Dqk=Dv=128 or 256 with Hq=Hkv=24; additionally, D256 with Hq=128/Hkv=1.
- Dense noncausal attention, without local masking, variable lengths, paged KV,
  block sparsity, or SplitKV, and with 2CTA MMA available.

Other configurations retain the original dispatcher. Selecting this environment
option does not expand the dedicated D256 kernel's existing feature support.

## Scheduling and pipeline depth

D128 flattens whole 2CTA query tasks across heads and batches. Each head does
not need padding to eight CTAs. At a head boundary the preferred cluster uses
pair-local K/V loads, avoiding multicast between different heads. The complete
grid is rounded only once to a multiple of the preferred cluster size. Ordinary
query-tail masking and 2CTA task granularity still apply.

D256 retains its existing query/head grid mapping and adds preferred/fallback
placement. Its default KV pipeline depth remains 4. Set `FA_HD256_KV_STAGE`
before import to specialize it to another depth; the depth is part of the JIT
cache key. Values 2–12 are accepted. For D256 FP8, 12 fits; BF16 requires more
shared memory and depth 5 was validated. Larger BF16 depths can exceed hardware
shared-memory limits. D128 continues to size its KV pipeline automatically.

Maximum depth is not necessarily fastest. Choose a depth using measurements for
the target workload and hardware.

## Reproduce event timing and sustained frequency

From the repository root, run these in separate processes on the same GPU:

```bash
python benchmarks/benchmark_preferred_clusters.py --gpu 0 --head-dim 128 --mode 2cta --output baseline.json
python benchmarks/benchmark_preferred_clusters.py --gpu 0 --head-dim 128 --mode '8cta+2cta/2cta' --output preferred.json
python benchmarks/benchmark_preferred_clusters.py --gpu 0 --head-dim 256 --mode '8cta+2cta/2cta' --kv-stages 5 --output d256.json
```

The default workload is B1 H24 Q=KV=200000 with FP8 E4M3 inputs and BF16 output.
Use `--dtype bf16` for BF16 inputs. The benchmark alternates two independent
Q/K/V buffer sets, excludes compilation and allocation from timing, warms up
for 40 launches, and collects 80 CUDA-event samples. It samples the selected
GPU's SM clock, power, and temperature through `nvidia-smi` every 20 ms.
Frequency summaries use the latter 80% of the timed window. GPU clocks and
power limits are not modified.

JSON output includes all event samples and timestamped telemetry. Report
latency and observed frequency together. TFLOP/s counts QK and PV only:
`4 * B * H * Q * KV * D / elapsed_seconds / 1e12`.

## Validate

```bash
FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=0 pytest -q tests/cute/test_mixed_cluster_scheduling.py
```

The tests cover query tails and head boundaries, FP8/BF16, D128/D256, both CLC
request settings, D256 depth specialization and cache separation, and ordinary
configurations outside the preferred-path selection guard. D256 BF16 at depth
12 is intentionally skipped because it exceeds shared-memory capacity.
