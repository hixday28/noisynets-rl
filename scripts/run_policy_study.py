"""Запуск A3C/PPO и шумовых вариантов по фиксированному манифесту."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,default=ROOT/'configs/policy_study.json')
    parser.add_argument('--out',type=Path,default=ROOT/'results/policy')
    parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args()
    if args.workers<1:parser.error('workers должен быть положительным')
    manifest=json.loads(args.manifest.read_text())
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/'configs').mkdir(exist_ok=True);(args.out/'logs').mkdir(exist_ok=True)
    (args.out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    jobs=[]
    for env in manifest['environments']:
        for condition in env['conditions']:
            values=env['config'] | {k:v for k,v in condition.items() if k!='key'}
            config=args.out/'configs'/f"{env['key']}_{condition['key']}.json"
            text=json.dumps(values,indent=2)+'\n'
            if config.exists() and config.read_text()!=text:raise ValueError(config)
            config.write_text(text)
            for seed in manifest['seeds']:
                jobs.append((config,values|{'seed':seed},args.out/env['key']/condition['key']/f'seed_{seed}'))
    def launch(job):
        config,values,out=job
        if (out/'COMPLETE').exists():
            stored=json.loads((out/'config.json').read_text())
            if any(stored.get(k)!=v for k,v in values.items()):raise ValueError(out)
            print('Пропущен',out.relative_to(args.out),flush=True);return
        if out.exists():raise FileExistsError(out)
        log=args.out/'logs'/('_'.join(out.relative_to(args.out).parts)+'.log')
        with log.open('w') as f:
            subprocess.run([sys.executable,'-m','noisynets.actorcritic','--config',str(config),
                            '--seed',str(values['seed']),'--out',str(out)],cwd=ROOT,
                           stdout=f,stderr=subprocess.STDOUT,check=True)
        print('Готово:',out.relative_to(args.out),flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:list(pool.map(launch,jobs))


if __name__=='__main__':main()
