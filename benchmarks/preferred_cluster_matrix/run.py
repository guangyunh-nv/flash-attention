"""Run the canonical base/plain/preferred CUDA-event matrix on 1–4 GPUs."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time

BASE = 'a369df707e1980fb328abcc1733e3457ec10155f'
FEATURE = 'ed916d78bc3b863cd463db5b9bac61f0e3227589'
ROOT = Path(__file__).resolve().parents[2]
CASES = [('bf16', 128), ('fp8', 128), ('bf16', 256), ('fp8', 256)]
ORDER = [('base', 0), ('tip', 0), ('tip_plain', 0), ('tip_plain', 1), ('tip', 1), ('base', 1)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-repo', type=Path, required=True)
    parser.add_argument('--gpus', type=int, nargs='+', default=[0])
    parser.add_argument('--output', type=Path, default=Path('agent_space/preferred_matrix'))
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if not 1 <= len(args.gpus) <= 4 or len(set(args.gpus)) != len(args.gpus) or min(args.gpus) < 0:
        parser.error('Provide 1–4 distinct nonnegative physical GPU indices')
    args.base_repo = args.base_repo.resolve()
    args.output = args.output.resolve()
    git = lambda repo, *a: subprocess.check_output(['git', '-C', str(repo), *a], text=True).strip()
    if git(args.base_repo, 'rev-parse', 'HEAD') != BASE:
        parser.error('Base checkout must be exactly ' + BASE)
    for repo in [args.base_repo, ROOT]:
        if git(repo, 'status', '--porcelain', '--untracked-files=no'):
            parser.error(f'Tracked modifications in {repo}')
    if git(ROOT, 'diff', FEATURE, '--', 'flash_attn', 'benchmarks/benchmark_preferred_clusters.py'):
        parser.error('Feature implementation differs from the validated tip')
    jobs = []
    for index, (dtype, dim) in enumerate(CASES):
        gpu = args.gpus[index % len(args.gpus)]
        for policy in (['default', 'matched_max'] if dim == 256 else ['default']):
            for rev, round_ in ORDER:
                stage = (5 if dtype == 'bf16' else 12) if policy == 'matched_max' else 4
                tag = f'{dtype}_d{dim}_{policy}_{rev}_r{round_}'
                jobs.append(dict(gpu=gpu, dtype=dtype, dim=dim, policy=policy, revision=rev, round=round_, stage=stage, tag=tag))
    if args.dry_run:
        print(json.dumps(jobs, indent=2))
        return
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Output directory must be empty; choose a new directory for each run')
    args.output.mkdir(parents=True, exist_ok=True)
    def capture(cmd):
        p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        return dict(command=cmd, returncode=p.returncode, output=p.stdout)
    manifest = dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), base=BASE,
                    feature=FEATURE, harness=git(ROOT, 'rev-parse', 'HEAD'), jobs=jobs,
                    gpu=capture(['nvidia-smi', '-q']), packages=capture([sys.executable, '-m', 'pip', 'freeze']),
                    python=sys.version, order=ORDER, original_cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    def lane(gpu):
        for j in (j for j in jobs if j['gpu'] == gpu):
            rev = j['revision']; env = os.environ.copy()
            # Remove inherited tuning overrides. Set only the audited options below.
            for key in list(env):
                if key.startswith('FA_'):
                    del env[key]
            env.update(CUDA_VISIBLE_DEVICES=str(gpu), FA_FWD_CLUSTER_MODE='8cta+2cta/2cta' if rev == 'tip' else '2cta',
                       FA_CLC='0', FA_DISABLE_2CTA='0', FA_HD256_KV_STAGE=str(j['stage']),
                       FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED='0', CUTE_DSL_LINEINFO='0',
                       MATRIX_REPO=str(args.base_repo if rev == 'base' else ROOT), MATRIX_GPU=str(gpu),
                       MATRIX_DTYPE=j['dtype'], MATRIX_DIM=str(j['dim']), MATRIX_REVISION=rev,
                       MATRIX_COMMIT=BASE if rev == 'base' else FEATURE, MATRIX_DEPTH_POLICY=j['policy'],
                       MATRIX_OUTPUT=str(args.output / (j['tag'] + '.json')),
                       MATRIX_BENCHMARK=str(ROOT / 'benchmarks/benchmark_preferred_clusters.py'))
            cmd = [sys.executable, str(Path(__file__).with_name('run_one.py'))]
            with (args.output / (j['tag'] + '.log')).open('w') as log:
                result = subprocess.run(cmd, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=900)
            print(j['tag'], 'exit', result.returncode, flush=True)
            if result.returncode:
                raise RuntimeError('Benchmark failed: ' + j['tag'])
    with concurrent.futures.ThreadPoolExecutor(len(args.gpus)) as pool:
        list(pool.map(lane, args.gpus))
    subprocess.run([sys.executable, str(Path(__file__).with_name('summarize.py')), str(args.output)], check=True)


if __name__ == '__main__':
    main()
