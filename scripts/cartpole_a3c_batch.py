"""Новые A3C-запуски CartPole из отдельного фиксированного манифеста."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import threading

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,default=ROOT/'configs/cartpole_a3c_final_manifest.json')
    p.add_argument('--out',type=Path,default=ROOT/'results/cartpole_a3c_final')
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--max-workers',type=int,default=4)
    args=p.parse_args()
    if args.workers<1 or args.max_workers<args.workers:p.error('Некорректное число процессов')
    manifest=json.loads(args.manifest.read_text())
    args.out.mkdir(parents=True,exist_ok=True)
    saved=args.out/'manifest.json'
    serialized=json.dumps(manifest,ensure_ascii=False,indent=2)+'\n'
    if saved.exists() and saved.read_text()!=serialized:
        raise ValueError('Манифест существующей серии отличается')
    saved.write_text(serialized)
    (args.out/'configs').mkdir(exist_ok=True)
    (args.out/'logs').mkdir(exist_ok=True)
    capacity=args.out/'execution_capacity.json'
    if not capacity.exists():capacity.write_text(json.dumps({'workers':args.workers})+'\n')
    files=['src/noisynets/actorcritic.py','src/noisynets/policy_networks.py',
           'src/noisynets/networks.py','scripts/cartpole_a3c_study.py']
    provenance={'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                'git_status':subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True),
                'source_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files}}
    record=args.out/'provenance.json'
    if record.exists():
        if json.loads(record.read_text())['source_sha256']!=provenance['source_sha256']:
            raise ValueError('Исходники существующей серии отличаются')
    else:
        record.write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n')
        for name in files:
            destination=args.out/'source'/name
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/name,destination)
    jobs=[]
    for environment in manifest['environments']:
        for condition in environment['conditions']:
            values=environment['config'] | {k:v for k,v in condition.items() if k!='key'}
            values['anneal_lr']=bool(manifest['optimization_schedule']['anneal_lr'])
            config=args.out/'configs'/f"{environment['key']}_{condition['key']}.json"
            contents=json.dumps(values,indent=2)+'\n'
            if config.exists() and config.read_text()!=contents:raise ValueError(config)
            config.write_text(contents)
            for seed in manifest['seeds']:
                jobs.append((config,values|{'seed':seed},
                             args.out/environment['key']/condition['key']/f'seed_{seed}'))
    jobs.sort(key=lambda job: (job[1]['seed'], job[2].parent.name))

    gate=threading.Condition()
    active=0
    def launch(job):
        nonlocal active
        config,values,out=job
        if (out/'COMPLETE').exists():
            stored=json.loads((out/'config.json').read_text())
            treatment=json.loads((out/'treatment.json').read_text())
            expected={k:v for k,v in values.items() if k!='anneal_lr'}
            if any(stored.get(k)!=v for k,v in expected.items()) or treatment['anneal_lr']!=values['anneal_lr']:
                raise ValueError(out)
            print('Пропущен готовый запуск',out.relative_to(args.out),flush=True)
            return
        if out.exists():raise FileExistsError(out)
        with gate:
            while True:
                try:limit=int(json.loads(capacity.read_text())['workers'])
                except (ValueError,KeyError):limit=args.workers
                limit=max(1,min(args.max_workers,limit))
                if active<limit:
                    active+=1
                    break
                gate.wait(timeout=2)
        try:
            log=args.out/'logs'/('_'.join(out.relative_to(args.out).parts)+'.log')
            with log.open('w') as f:
                subprocess.run([sys.executable,'scripts/cartpole_a3c_study.py','--config',str(config),
                                '--out',str(out),'--seed',str(values['seed'])],cwd=ROOT,
                               stdout=f,stderr=subprocess.STDOUT,check=True)
        finally:
            with gate:
                active-=1
                gate.notify_all()
        print('Готово:',out.relative_to(args.out),flush=True)

    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:list(pool.map(launch,jobs))


if __name__=='__main__':main()
