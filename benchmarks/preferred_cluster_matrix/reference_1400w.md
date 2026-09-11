# Canonical CUDA-event matrix: base, tip plain, tip preferred

B=1, Hq=Hkv=24, Q=KV=200000, Dqk=Dv=128 or 256, dense noncausal attention. FP8 means E4M3 inputs; every output is BF16. Measured on four NVIDIA GB300 GPUs (152 SMs each, 1400 W power limits) on 2026-09-11.

- Base 2CTA/2CTA: `a369df707e1980fb328abcc1733e3457ec10155f`.
- Tip 2CTA/2CTA: `ed916d78bc3b863cd463db5b9bac61f0e3227589`, `FA_FWD_CLUSTER_MODE=2cta`.
- Tip 8+2/2: the same tip, `FA_FWD_CLUSTER_MODE=8cta+2cta/2cta`.
- CLC is off throughout. Each dtype/dimension case uses one GPU for all paths and depths. The source checkouts and branch tip are unchanged.

## Default KV depth

| Input | D | KV depth | Base 2/2 ms | Tip 2/2 ms | Tip 8+2/2 ms | Plain change % | Mixed reduction % |
|---|---:|---:|---:|---:|---:|---:|---:|
| bf16 | 128 | 6 | 265.417 | 265.605 | 261.598 | +0.07% | +1.51% |
| bf16 | 256 | 4 | 569.593 | 569.609 | 571.843 | +0.00% | -0.39% |
| fp8 | 128 | 16 | 208.503 | 208.546 | 190.325 | +0.02% | +8.74% |
| fp8 | 256 | 4 | 427.352 | 427.471 | 393.121 | +0.03% | +8.04% |

## Maximum KV depth

| Input | D | KV depth | Base 2/2 ms | Tip 2/2 ms | Tip 8+2/2 ms | Plain change % | Mixed reduction % |
|---|---:|---:|---:|---:|---:|---:|---:|
| bf16 | 128 | 6 | 265.417 | 265.605 | 261.598 | +0.07% | +1.51% |
| bf16 | 256 | 5 | 597.598 | 606.327 | 574.261 | +1.46% | +5.29% |
| fp8 | 128 | 16 | 208.503 | 208.546 | 190.325 | +0.02% | +8.74% |
| fp8 | 256 | 12 | 387.206 | 387.584 | 378.642 | +0.10% | +2.31% |

Plain change = (tip plain / base − 1) × 100: positive is slower. Mixed reduction = (1 − tip mixed / tip plain) × 100: positive is faster.

D128 automatically selects the maximum KV depth in both revisions: BF16 depth 6 and FP8 depth 16. Its default and maximum rows therefore reuse the same measurements. D256 default is depth 4. Its maximum is BF16 depth 5 or FP8 depth 12, with Q depth 2 throughout. The base hardcodes depth 4; only its maximum-depth cases use a benchmark-only `_setup_attributes` override. The tip uses `FA_HD256_KV_STAGE`. Maximum capacity does not imply optimal latency.

## Sustained SM frequency

Values are median [P10–P90] MHz over the latter 80% of the timed windows. These are measured frequencies, not clock locks.

| Depth | Input | D | GPU | Base 2/2 MHz | Tip 2/2 MHz | Tip 8+2/2 MHz |
|---|---|---:|---:|---:|---:|---:|
| default | bf16 | 128 | 0 | 1507 [1492–1545] | 1507 [1492–1545] | 1530 [1522–1567] |
| default | bf16 | 256 | 2 | 1537 [1530–1537] | 1537 [1530–1537] | 1545 [1537–1560] |
| default | fp8 | 128 | 1 | 2002 [2002–2032] | 2002 [2002–2025] | 1965 [1965–1972] |
| default | fp8 | 256 | 3 | 1987 [1972–1995] | 1987 [1972–1995] | 1950 [1942–1957] |
| maximum | bf16 | 128 | 0 | 1507 [1492–1545] | 1507 [1492–1545] | 1530 [1522–1567] |
| maximum | bf16 | 256 | 2 | 1365 [1320–1447] | 1342 [1297–1417] | 1440 [1417–1477] |
| maximum | fp8 | 128 | 1 | 2002 [2002–2032] | 2002 [2002–2025] | 1965 [1965–1972] |
| maximum | fp8 | 256 | 3 | 1710 [1695–1800] | 1710 [1695–1800] | 1755 [1732–1852] |

## Method

Every configuration has two independent processes, 40 CUDA-event samples per process (80 total). Each process compiles and warms two independent rotating Q/K/V sets, then executes 32 sustained warmup launches before timing. Compilation/allocation are excluded. Persistent compilation cache is disabled. GPU-specific nvidia-smi SM frequency/power/temperature telemetry is sampled every 20 ms. Clocks and power limits are not modified.

The original default base/mixed comparison used base–mixed–mixed–base order. Plain controls were added afterward in two runs. For D256 maximum depth, the order was base–mixed–plain–plain–mixed–base. No GPU ran concurrent benchmark processes. This is natural-DVFS testing; assess small differences together with per-run medians and frequency ranges. Different dtype/dimension cases are on different GPUs, so only within-row comparisons are controlled.

## Per-configuration details

| Depth | Input | D | Path | Round medians ms | Min–max ms | TFLOP/s |
|---|---|---:|---|---:|---:|---:|
| default | bf16 | 128 | base | 265.337, 265.449 | 258.114–268.367 | 1851.9 |
| default | bf16 | 128 | tip_plain | 265.570, 265.697 | 258.139–269.498 | 1850.6 |
| default | bf16 | 128 | tip | 261.037, 261.943 | 253.349–263.855 | 1878.9 |
| default | bf16 | 256 | base | 569.534, 569.613 | 562.465–572.677 | 1725.9 |
| default | bf16 | 256 | tip_plain | 569.595, 569.609 | 558.239–574.915 | 1725.8 |
| default | bf16 | 256 | tip | 571.821, 571.855 | 561.257–575.896 | 1719.1 |
| default | fp8 | 128 | base | 208.415, 208.645 | 205.623–211.295 | 2357.4 |
| default | fp8 | 128 | tip_plain | 208.520, 208.611 | 205.116–209.901 | 2356.9 |
| default | fp8 | 128 | tip | 190.329, 190.325 | 185.821–193.567 | 2582.5 |
| default | fp8 | 256 | base | 427.457, 427.265 | 426.965–433.343 | 2300.3 |
| default | fp8 | 256 | tip_plain | 427.503, 427.441 | 426.985–428.464 | 2299.7 |
| default | fp8 | 256 | tip | 393.242, 392.347 | 389.378–396.471 | 2500.6 |
| maximum | bf16 | 128 | base | 265.337, 265.449 | 258.114–268.367 | 1851.9 |
| maximum | bf16 | 128 | tip_plain | 265.570, 265.697 | 258.139–269.498 | 1850.6 |
| maximum | bf16 | 128 | tip | 261.037, 261.943 | 253.349–263.855 | 1878.9 |
| maximum | bf16 | 256 | base | 596.118, 599.788 | 584.717–603.777 | 1645.0 |
| maximum | bf16 | 256 | tip_plain | 606.268, 606.374 | 589.923–612.516 | 1621.3 |
| maximum | bf16 | 256 | tip | 574.311, 574.248 | 560.623–580.635 | 1711.8 |
| maximum | fp8 | 128 | base | 208.415, 208.645 | 205.623–211.295 | 2357.4 |
| maximum | fp8 | 128 | tip_plain | 208.520, 208.611 | 205.116–209.901 | 2356.9 |
| maximum | fp8 | 128 | tip | 190.329, 190.325 | 185.821–193.567 | 2582.5 |
| maximum | fp8 | 256 | base | 387.187, 387.249 | 375.734–389.931 | 2538.8 |
| maximum | fp8 | 256 | tip_plain | 387.154, 388.145 | 379.234–390.072 | 2536.3 |
| maximum | fp8 | 256 | tip | 378.327, 378.806 | 373.895–381.429 | 2596.2 |

TFLOP/s counts QK and PV only: 4*B*H*Q*KV*D / seconds / 1e12.

This reference was collected before the portable measurement branch. The adjacent `reference_1400w.json` contains the summarized results. See `AGENT_INSTRUCTIONS.md` to reproduce on the new cluster.
