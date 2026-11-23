"""Проверка происхождения, бюджета и расписания изолированной итоговой серии A3C."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def read_json(path):return json.loads(path.read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,default=ROOT/'results/cartpole_a3c_final')
    args=p.parse_args();root=args.results
    manifest=read_json(root/'manifest.json')
    assert manifest==read_json(ROOT/'configs/cartpole_a3c_final_manifest.json')
    provenance=read_json(root/'provenance.json')
    for name,digest in provenance['source_sha256'].items():
        assert hashlib.sha256((root/'source'/name).read_bytes()).hexdigest()==digest,name
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    rows=[]
    for env in manifest['environments']:
        for condition in env['conditions']:
            prescribed=env['config']|{k:v for k,v in condition.items() if k!='key'}
            for seed in manifest['seeds']:
                folder=root/env['key']/condition['key']/f'seed_{seed}'
                assert (folder/'COMPLETE').exists(),folder
                c=read_json(folder/'config.json')
                assert c==prescribed|{'seed':seed},folder
                meta=read_json(folder/'metadata.json')
                assert meta['training_source_sha256']==provenance['source_sha256'],folder
                assert read_json(folder/'treatment.json')['anneal_lr'] is True,folder
                assert meta['actor_workers']==4 and meta['actor_learner_type']=='asynchronous_threads',folder
                assert meta['stale_gradient_updates']>0 and meta['max_staleness']>0,folder
                history=list(csv.DictReader((folder/'evaluations.csv').open()))
                steps=[int(r['step']) for r in history]
                assert steps==list(range(0,500001,50000)),folder
                for r in history:
                    step=int(r['step']);previous=max(0,step-c['eval_interval'])
                    expected=c['learning_rate']*max(.05,1-previous/c['steps'])
                    assert np.isclose(float(r['learning_rate']),expected,rtol=0,atol=1e-12),folder
                updates=list(csv.DictReader((folder/'updates.csv').open()))
                assert len(updates)==meta['gradient_updates'],folder
                assert [int(r['update']) for r in updates]==list(range(1,len(updates)+1)),folder
                assert int(updates[-1]['step'])==500000,folder
                scores=read_json(folder/'final_returns.json')
                assert set(scores)=={'behaviour','mean_greedy','mean_sampled'},folder
                assert all(len(v)==30 for v in scores.values()),folder
                row={'condition':condition['key'],'seed':seed,'steps':c['steps'],
                     'gradient_updates':meta['gradient_updates'],'seconds':meta['elapsed_seconds'],
                     'max_staleness':meta['max_staleness'],
                     'auc_behaviour':float(np.trapezoid([float(r['return_mean']) for r in history],steps)/c['steps'])}
                for mode,values in scores.items():
                    row[mode]=float(np.mean(values))
                    row[mode+'_success_rate']=float(np.mean(np.asarray(values)==500))
                row['meets_475_both']=row['behaviour']>=475 and row['mean_greedy']>=475
                rows.append(row)
    assert len(rows)==30
    conditions=[]
    for key in [c['key'] for e in manifest['environments'] for c in e['conditions']]:
        selected=[r for r in rows if r['condition']==key]
        group={'condition':key,'n_seeds':len(selected),
               'meets_475_both':sum(r['meets_475_both'] for r in selected)}
        for metric in ['behaviour','mean_greedy','mean_sampled','auc_behaviour']:
            values=[r[metric] for r in selected]
            group[metric+'_mean']=float(np.mean(values))
            group[metric+'_seed_sd']=float(np.std(values,ddof=1))
            group[metric+'_min']=float(np.min(values))
        conditions.append(group)
    summary={'all_provenance_and_schedule_checks_passed':True,
             'n_models':len(rows),'total_training_steps':sum(r['steps'] for r in rows),
             'total_gradient_updates':sum(r['gradient_updates'] for r in rows),
             'total_final_episodes':30*3*len(rows),
             'all_30_meet_475_both':all(r['meets_475_both'] for r in rows),
             'source_sha256':provenance['source_sha256'],
             'manifest_sha256':hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest(),
             'actual_learning_rates_used':[.003*(1-step/500000) for step in range(0,500000,50000)],
             'conditions':conditions,'per_run':rows}
    out=root/'cartpole_a3c_audit.json'
    out.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ['per_run','source_sha256']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
