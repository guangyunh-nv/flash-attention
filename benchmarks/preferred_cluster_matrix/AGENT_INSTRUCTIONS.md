# Agent task: canonical preferred-cluster measurements on a 1000 W B300

Run the complete matrix below, summarize results, and retain every raw event/telemetry JSON and log. This branch adds measurement tooling on top of the clean feature tip; do not change the attention implementation to make a result look better.

## Revisions and workload

- Base: `a369df707e1980fb328abcc1733e3457ec10155f`, ordinary 2CTA/2CTA.
- Feature implementation: `ed916d78bc3b863cd463db5b9bac61f0e3227589`.
- Compare base 2CTA/2CTA, tip 2CTA/2CTA, and tip preferred 8CTA/fallback 2CTA with 2CTA MMA (`8+2/2`). No 1CTA-MMA variants.
- B=1, Hq=Hkv=24, Q=KV=200000, Dqk=Dv in {128,256}, noncausal dense attention.
- Inputs: BF16 and FP8 E4M3. Output BF16; LSE FP32.
- Depth: default and maximum shared-memory KV stages. Q stages=2 throughout.
- D128 BF16=6, FP8=16: default already equals maximum; explicitly reuse measurements.
- D256 default=4, maximum BF16=5 and FP8=12. The base hardcodes depth4: the harness applies a clearly labeled benchmark-only `_setup_attributes` override for its maximum-depth cases. Source files remain unchanged.
- Both modes run with CLC off. Plain D128 is statically persistent; preferred D128 is nonpersistent and uses flattened whole-2CTA scheduling without per-head 8CTA padding.

## Setup

Use an exclusive GPU allocation. Inspect `nvidia-smi -q`, confirm SM103 B300/GB300 and the expected 1000 W enforced power limit, and record any existing clock locks. Do not change power limits, clock locks, Tensor Core boost policy, or the allocation's GPU visibility without understanding the site's allocation rules. GPU arguments to this harness are physical nvidia-smi indices. It replaces CUDA_VISIBLE_DEVICES per worker; when a scheduler restricts visible devices, select only GPUs allocated to this job.

Use a CUDA13-compatible PyTorch environment with NVIDIA CuTeDSL 4.7.1 and the project's dependencies. Record actual versions. The original environment used Python3.12. Do not silently upgrade the DSL for the comparison. Check available memory: the harness uses two rotating full-size Q/K/V/O sets; D256 BF16 requires roughly 20 GB plus runtime overhead. It must not silently shrink the workload if allocation fails.

From the measurement branch checkout:

```bash
# Use the Python environment prepared for CUDA13 and CuTeDSL4.7.1.
python -m pip install -e flash_attn/cute
# Verify the resolved DSL version after installation.
python -m pip show nvidia-cutlass-dsl torch

git worktree add --detach ../flash-attention-preferred-base a369df707e1980fb328abcc1733e3457ec10155f
python benchmarks/preferred_cluster_matrix/run.py \
  --base-repo ../flash-attention-preferred-base --gpus 0 1 2 3 \
  --output agent_space/b300_1000w_matrix --dry-run
```

The base worktree must be pristine. The feature kernel and standalone benchmark must match the validated tip; the launcher checks this. Source namespaces are selected explicitly in each child process, so an installed editable package cannot accidentally substitute tip code for the base.

Before timing, run the branch's focused correctness tests on one allocated GPU:

```bash
CUDA_VISIBLE_DEVICES=0 FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=0 \
  pytest -q tests/cute/test_mixed_cluster_scheduling.py
```

If a test fails, preserve the failure and investigate before accepting performance. Expected BF16 D256 depth12 capacity skips are not failures.

## Measure

```bash
python benchmarks/preferred_cluster_matrix/run.py \
  --base-repo ../flash-attention-preferred-base --gpus 0 1 2 3 \
  --output agent_space/b300_1000w_matrix
```

One GPU is also supported: use `--gpus 0`. Two or three GPUs are supported as well. Cases are assigned round-robin to GPU lanes, with no concurrent processes on one GPU. Do not run other tests/profilers on those GPUs during the event matrix.

The launcher executes 36 processes: two independent processes for each of 18 distinct configurations. Each process compiles/warms two rotating buffers, executes32 additional sustained warmups, then40 CUDA-event samples. Pool the80 samples for the median. Allocation/JIT are excluded; persistent disk compile cache is disabled. For each case/depth, the order is base–mixed–plain–plain–mixed–base. This is more consistently interleaved than the original1400 W default-depth collection, where plain controls were added afterward.

Telemetry comes from nvidia-smi every20 ms. Report median and P10–P90 SM clocks in the latter80% of the timed window, plus board power and temperature. The launcher saves the initial GPU configuration and package versions in manifest.json. It does not set power or clocks. Use a fresh empty output directory each time; failed/partial runs remain available for diagnosis.

## Deliver

The launcher generates `base-tip-events.md`, `base-tip-events.json`, and `summary_three.json` in the output directory. Summarization can be rerun:

```bash
python benchmarks/preferred_cluster_matrix/summarize.py agent_space/b300_1000w_matrix
```

Report both depth tables, with all three paths, median event ms, achieved TFLOP/s, stable SM clocks, board power, and run spread. Count FLOPs as `4*B*H*Q*KV*D` (QK and PV only). Explicitly distinguish:

1. Plain regression: `(tip_plain/base - 1)*100`; positive means slower.
2. Preferred improvement: `(1 - tip_mixed/tip_plain)*100`; positive means lower latency.
3. Comparison to the1400 W reference, which also changes machine/power environment and is not a controlled within-GPU A/B.

Do not dismiss a repeatable regression as noise. Maximum depth is not necessarily fastest. Preserve all36 JSON/log pairs, manifest, and summaries outside Git unless asked to commit results. Keep the feature branch clean.

## Prior observations and optional profiling

`reference_1400w.md` and `.json` contain the previous event measurements. Default plain base→tip changes were <=0.07%; maximum-depth BF16 D256 plain regressed1.46%. D128 BF16 preferred improved1.51%, FP8 improved8.74% versus tip plain.

Profiling on1400 W GB300 found BF16 D128 Tensor Core active time about99% in both paths, while LRC read traffic fell927→394 GB and L2 reads956→416 GB. FP8 Tensor Core active time rose50.1→56.6%. These are active-time percentages, not peak-FLOP percentages. BF16 event-run clocks rose1500–1507→1522–1530 MHz, with board power about1377→1372 W. Reduced memory-system power enabling SM frequency is consistent with these observations; LRC-specific power was not directly measured. Frequency gains are real performance gains.

If the1000 W results warrant profiling, keep it separate from event timing. Use the ncu-profiler skill if available. Explicitly set `--clock-control none --pipeline-boost-state dynamic --cache-control none`, warm up before selecting one kernel, and use CUTE_DSL_LINEINFO=1 for source correlation. Start with a one-pass collection of duration, SM clock, DRAM bytes, L2 sectors, LRC sectors; collect Tensor Core active cycles separately if needed to remain one-pass. Verify supported metric names locally. Use detailed replay for diagnosis only, never as the event latency result. Do not infer memory bandwidth saturation from long-scoreboard stalls alone: the previous BF16 samples concentrated on barrier-predicate consumers while Tensor Cores remained busy.
