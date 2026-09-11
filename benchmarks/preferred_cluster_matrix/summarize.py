import json,statistics,sys
from pathlib import Path
out=Path(sys.argv[1]).resolve();release=out
rows=[]
revisions=['base','tip_plain','tip']
for depth in ['default','maximum']:
 for dtype in ['bf16','fp8']:
  for dim in [128,256]:
   policy='matched_max' if depth=='maximum' and dim==256 else 'default'
   row=dict(depth=depth,dtype=dtype,head_dim=dim,reuses_default=(depth=='maximum' and dim==128))
   for rev in revisions:
    files=sorted(out.glob(f'{dtype}_d{dim}_{policy}_{rev}_r*.json'))
    assert len(files)==2,(depth,dtype,dim,rev,len(files))
    records=[json.loads(p.read_text()) for p in files]
    samples=[v for r in records for v in r['samples_ms']]
    clocks=[];powers=[];temperatures=[]
    for r in records:
     end=max(x['t'] for x in r['frequency_samples'])
     stable=[x for x in r['frequency_samples'] if x['t']>=.2*end]
     clocks += [x['sm_mhz'] for x in stable]
     powers += [x['power_w'] for x in stable]
     temperatures += [x['temperature_c'] for x in stable]
    clocks.sort();spec=records[0]['specializations'][0]
    assert all(r['specializations'][0]==spec for r in records)
    row[rev]=dict(median_ms=statistics.median(samples),min_ms=min(samples),max_ms=max(samples),n=len(samples),sm_mhz=statistics.median(clocks),sm_mhz_p10=clocks[round(.1*(len(clocks)-1))],sm_mhz_p90=clocks[round(.9*(len(clocks)-1))],kv_stage=spec['kv_stage'],gpu=records[0]['gpu'],power_w=statistics.median(powers),temperature_c=statistics.median(temperatures),tflops=(491520 if dim==128 else 983040)/statistics.median(samples),round_medians_ms=[r['median_ms'] for r in records])
   row['plain_regression_pct']=100*(row['tip_plain']['median_ms']/row['base']['median_ms']-1)
   row['mixed_latency_reduction_pct']=100*(1-row['tip']['median_ms']/row['tip_plain']['median_ms'])
   rows.append(row)
(release/'base-tip-events.json').write_text(json.dumps(rows,indent=2)+'\n')
(out/'summary_three.json').write_text(json.dumps(rows,indent=2)+'\n')
lines=['# Canonical CUDA-event matrix: base, tip plain, tip preferred','',
'B=1, Hq=Hkv=24, Q=KV=200000, Dqk=Dv=128 or 256, dense noncausal attention. FP8 means E4M3 inputs; every output is BF16. GPU identity, configured power limits, driver, package versions, and timestamp are recorded in manifest.json.','',
'- Base 2CTA/2CTA: `a369df707e1980fb328abcc1733e3457ec10155f`.',
'- Tip 2CTA/2CTA: `ed916d78bc3b863cd463db5b9bac61f0e3227589`, `FA_FWD_CLUSTER_MODE=2cta`.',
'- Tip 8+2/2: the same tip, `FA_FWD_CLUSTER_MODE=8cta+2cta/2cta`.',
'- CLC is off throughout. Each dtype/dimension case uses one GPU for all paths and depths. The source checkouts and branch tip are unchanged.','']
for depth in ['default','maximum']:
 lines += [f'## {depth.capitalize()} KV depth','',
 '| Input | D | KV depth | Base 2/2 ms | Tip 2/2 ms | Tip 8+2/2 ms | Plain change % | Mixed reduction % |',
 '|---|---:|---:|---:|---:|---:|---:|---:|']
 for r in rows:
  if r['depth']!=depth:continue
  a,b,c=r['base'],r['tip_plain'],r['tip']
  lines.append(f"| {r['dtype']} | {r['head_dim']} | {a['kv_stage']} | {a['median_ms']:.3f} | {b['median_ms']:.3f} | {c['median_ms']:.3f} | {r['plain_regression_pct']:+.2f}% | {r['mixed_latency_reduction_pct']:+.2f}% |")
 lines+=['']
lines += ['Plain change = (tip plain / base − 1) × 100: positive is slower. Mixed reduction = (1 − tip mixed / tip plain) × 100: positive is faster.','',
'D128 automatically selects the maximum KV depth in both revisions: BF16 depth 6 and FP8 depth 16. Its default and maximum rows therefore reuse the same measurements. D256 default is depth 4. Its maximum is BF16 depth 5 or FP8 depth 12, with Q depth 2 throughout. The base hardcodes depth 4; only its maximum-depth cases use a benchmark-only `_setup_attributes` override. The tip uses `FA_HD256_KV_STAGE`. Maximum capacity does not imply optimal latency.','',
'## Sustained SM frequency','',
'Values are median [P10–P90] MHz over the latter 80% of the timed windows. These are measured frequencies, not clock locks.','',
'| Depth | Input | D | GPU | Base 2/2 MHz | Tip 2/2 MHz | Tip 8+2/2 MHz |',
'|---|---|---:|---:|---:|---:|---:|']
for r in rows:
 freq=lambda x:f"{x['sm_mhz']:.0f} [{x['sm_mhz_p10']:.0f}–{x['sm_mhz_p90']:.0f}]"
 lines.append(f"| {r['depth']} | {r['dtype']} | {r['head_dim']} | {r['base']['gpu']} | {freq(r['base'])} | {freq(r['tip_plain'])} | {freq(r['tip'])} |")
lines += ['','## Method','',
'Every configuration has two independent processes, 40 CUDA-event samples per process (80 total). Each process compiles and warms two independent rotating Q/K/V sets, then executes 32 sustained warmup launches before timing. Compilation/allocation are excluded. Persistent compilation cache is disabled. GPU-specific nvidia-smi SM frequency/power/temperature telemetry is sampled every 20 ms. Clocks and power limits are not modified.','',
'For every case and distinct depth, the order is base–mixed–plain–plain–mixed–base. No GPU ran concurrent benchmark processes. This is natural-DVFS testing; assess small differences together with per-run medians and frequency ranges. Different dtype/dimension cases are on different GPUs, so only within-row comparisons are controlled.','',
'## Per-configuration details','',
'| Depth | Input | D | Path | Round medians ms | Min–max ms | TFLOP/s |',
'|---|---|---:|---|---:|---:|---:|']
for r in rows:
 for rev in revisions:
  x=r[rev];med=', '.join(f'{v:.3f}' for v in x['round_medians_ms'])
  lines.append(f"| {r['depth']} | {r['dtype']} | {r['head_dim']} | {rev} | {med} | {x['min_ms']:.3f}–{x['max_ms']:.3f} | {x['tflops']:.1f} |")
lines += ['','TFLOP/s counts QK and PV only: 4*B*H*Q*KV*D / seconds / 1e12.','',
'The branch benchmark is `benchmarks/benchmark_preferred_clusters.py`. Source-tree selection, depth override, runners, specialization logs, and every event/telemetry JSON are in the selected output directory; scripts are in `benchmarks/preferred_cluster_matrix/`. The machine-readable result is `base-tip-events.json` next to this report.']
(release/'base-tip-events.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines[:30]))

with (release/'base-tip-events.md').open('a') as f:
 f.write('\n## Stable board power and temperature\n\n| Depth | Input | D | Path | Power W | Temperature C |\n|---|---|---:|---|---:|---:|\n')
 for r in rows:
  for rev in revisions:
   x=r[rev]
   f.write(f"| {r['depth']} | {r['dtype']} | {r['head_dim']} | {rev} | {x['power_w']:.2f} | {x['temperature_c']:.1f} |\n")
