"""Проверка и анализ серии A3C/PPO с двумя генераторами параметрического шума."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .actorcritic import evaluate_policy, load_policy_checkpoint
from .final_analysis import paired_interval, read_json

LABELS={'a3c':'A3C','noisy_a3c_independent':'Noisy-A3C независимый',
        'noisy_a3c_factorized':'Noisy-A3C факторизованный','ppo':'PPO',
        'noisy_ppo_factorized':'Noisy-PPO факторизованный',
        'noisy_ppo_independent':'Noisy-PPO независимый'}


def load_policy_study(root=Path('results/policy'), manifest_path=Path('configs/policy_study.json')):
    root=Path(root); manifest=read_json(manifest_path); rows=[]; curves=[]; hashes=[]
    expected=set()
    for env in manifest['environments']:
        for condition in env['conditions']:
            config=env['config'] | {k:v for k,v in condition.items() if k!='key'}
            for seed in manifest['seeds']:
                expected.add((env['key'],condition['key'],seed))
                folder=root/env['key']/condition['key']/f'seed_{seed}'
                if not (folder/'COMPLETE').exists():raise ValueError(f'Неполный запуск: {folder}')
                c=read_json(folder/'config.json');meta=read_json(folder/'metadata.json')
                if c!=config|{'seed':seed}:raise AssertionError(f'Конфигурация: {folder}')
                hashes.append(meta['training_source_sha256'])
                details=read_json(folder/'final_episode_details.json'); scores=read_json(folder/'final_returns.json')
                modes={'behaviour','mean_greedy','mean_sampled'} if c['algorithm'].startswith('noisy-') else {'behaviour','mean_greedy'}
                assert set(details)==set(scores)==modes
                for mode, episodes in details.items():
                    assert len(episodes)==c['final_episodes']==30
                    assert [r['seed'] for r in episodes]==list(range(c['final_seed'],c['final_seed']+30))
                    assert [r['return'] for r in episodes]==scores[mode]
                    for record in episodes:
                        assert 1<=record['length']<=500 and (record['terminated'] or record['truncated'])
                        expected_return=(record['length'] if c['env']=='CartPole-v1' else
                                         -record['length']+int(record['terminated']))
                        assert record['return']==expected_return
                        if record['truncated']:assert record['length']==500
                curve=pd.read_csv(folder/'evaluations.csv')
                assert list(curve.step)==list(range(0,c['steps']+1,c['eval_interval']))
                assert np.isfinite(curve.to_numpy()).all()
                payload=torch.load(folder/'model.pt',map_location='cpu',weights_only=True)
                assert payload['config']==c and payload['step']==c['steps']
                assert all(torch.isfinite(t).all() for t in payload['model'].values())
                threshold=475 if c['env']=='CartPole-v1' else -100
                reached=curve.loc[curve.return_mean>=threshold,'step']
                primary=np.array(scores['behaviour'])
                success=[r['truncated'] if c['env']=='CartPole-v1' else r['terminated'] for r in details['behaviour']]
                family=c['algorithm'].removeprefix('noisy-')
                if family=='a3c':
                    assert meta['actor_workers']==4 and meta['stale_gradient_updates']>0
                else:
                    assert meta['max_initial_logratio_error']<2e-5
                rows.append({'environment':env['key'],'condition':condition['key'],'family':family,
                    'seed':seed,'steps':c['steps'],'final_mean':float(primary.mean()),
                    'mean_greedy':float(np.mean(scores['mean_greedy'])),
                    'mean_sampled':float(np.mean(scores.get('mean_sampled',primary))),
                    'auc_mean':float(np.trapezoid(curve.return_mean,curve.step)/c['steps']),
                    'success_rate':float(np.mean(success)),
                    'first_threshold_step':float(reached.iloc[0]) if len(reached) else np.nan,
                    'tail_validation_sd':float(curve.return_mean.tail(5).std(ddof=1)),
                    'elapsed_seconds':meta['elapsed_seconds'],'gradient_updates':meta['gradient_updates'],
                    'max_staleness':meta.get('max_staleness',0),
                    'trainable_parameters':meta['trainable_parameters']})
                curves.append(curve.assign(environment=env['key'],condition=condition['key'],seed=seed))
    observed={(p.parts[-4],p.parts[-3],int(p.parent.name.split('_')[-1])) for p in root.glob('*/*/seed_*/COMPLETE')}
    assert observed==expected and len(rows)==120
    assert all(h==hashes[0] for h in hashes)
    return pd.DataFrame(rows),pd.concat(curves,ignore_index=True),manifest


def summarize_policy(runs):
    conditions=[]; comparisons=[]; generations=[]; modes=[]
    for (env,condition), group in runs.groupby(['environment','condition']):
        conditions.append({'environment':env,'condition':condition,'label':LABELS[condition],
            'n_seeds':len(group),'final_mean':float(group.final_mean.mean()),
            'final_seed_sd':float(group.final_mean.std(ddof=1)),
            'median':float(group.final_mean.median()),'auc_mean':float(group.auc_mean.mean()),
            'auc_seed_sd':float(group.auc_mean.std(ddof=1)),
            'mean_greedy':float(group.mean_greedy.mean()),'mean_sampled':float(group.mean_sampled.mean()),
            'success_rate':float(group.success_rate.mean()),
            'threshold_reached':int(group.first_threshold_step.notna().sum()),
            'threshold_step_median_among_reached':(float(group.first_threshold_step.median())
                                                  if group.first_threshold_step.notna().any() else None),
            'tail_validation_sd':float(group.tail_validation_sd.mean())})
        if condition.startswith('noisy_'):
            reference=group.family.iloc[0]
            base=runs.loc[(runs.environment==env)&(runs.condition==reference)]
            for metric in ['final_mean','auc_mean']:
                comparisons.append({'environment':env,'condition':condition,'reference':reference,
                                    'metric':metric,**paired_interval(group.set_index('seed')[metric],
                                                                    base.set_index('seed')[metric])})
            modes.append({'environment':env,'condition':condition,'metric':'behaviour_minus_mean_sampled',
                          **paired_interval(group.set_index('seed').final_mean,
                                            group.set_index('seed').mean_sampled)})
    for env in runs.environment.unique():
        for family in ['a3c','ppo']:
            a=runs.loc[(runs.environment==env)&(runs.condition==f'noisy_{family}_independent')].set_index('seed')
            b=runs.loc[(runs.environment==env)&(runs.condition==f'noisy_{family}_factorized')].set_index('seed')
            for metric in ['final_mean','auc_mean']:
                generations.append({'environment':env,'family':family,'metric':metric,
                                    **paired_interval(a[metric],b[metric])})
    return {'conditions':conditions,'comparisons':comparisons,'generation_comparisons':generations,
            'mode_comparisons':modes,'n_training_runs':len(runs),'training_steps':int(runs.steps.sum()),
            'primary_evaluation':'categorical behaviour; noisy parameter cadence 20 (A3C), 128 (PPO)',
            'uncertainty':'Все 95%-интервалы описательные: paired bootstrap 20000, seed7241, без коррекции множественности',
            'convergence':'Первое достижение validation475/-100; не доказательство сходимости; неуспехи цензурируются',
            'asynchrony':'Настоящие конкурентные actor-learner; численная повторяемость A3C ограничена порядком потоков'}


def make_figures(runs,curves,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'savefig.dpi':200})
    colors={'a3c':'#333333','ppo':'#333333','independent':'#2565a0','factorized':'#b27327'}
    for env in ['cartpole','acrobot']:
        fig,axes=plt.subplots(2,1,figsize=(7.2,6.4),sharex=True)
        for ax,family in zip(axes,['a3c','ppo']):
            subset=curves.loc[curves.environment==env]
            lows=[];highs=[]
            for condition in LABELS:
                if condition!=family and not condition.startswith('noisy_'+family+'_'):continue
                data=subset.loc[subset.condition==condition].groupby('step').return_mean.agg(['mean','std'])
                color=colors[condition.split('_')[-1]]
                ax.plot(data.index,data['mean'],color=color,label=LABELS[condition])
                ax.fill_between(data.index,data['mean']-data['std'],data['mean']+data['std'],color=color,alpha=.12)
                lows.extend(data['mean']-data['std']);highs.extend(data['mean']+data['std'])
            if env=='acrobot':ax.axhline(-500,color='#999999',ls=':',lw=1)
            ax.set(title=family.upper(),ylabel='Возврат поведения')
            ax.grid(alpha=.2);ax.legend(frameon=False,fontsize=8)
        axes[-1].set_xlabel('Суммарные шаги среды всех actor-learner')
        fig.suptitle('CartPole-v1' if env=='cartpole' else 'Acrobot-v1')
        fig.tight_layout();fig.savefig(out/f'{env}_learning.png');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(7.2,6.2),sharex=False)
    for i,env in enumerate(['cartpole','acrobot']):
        for j,(family,gen) in enumerate([('a3c','independent'),('ppo','factorized')]):
            ax=axes[i,j];data=curves.loc[(curves.environment==env)&(curves.condition==f'noisy_{family}_{gen}')]
            for col,label in [('sigma_weight_hidden','Общий слой'),('sigma_weight_actor','Политика'),('sigma_weight_critic','Ценность')]:
                stat=data.groupby('step')[col].agg(['mean','std'])
                ax.plot(stat.index,stat['mean'],label=label)
                ax.fill_between(stat.index,stat['mean']-stat['std'],stat['mean']+stat['std'],alpha=.12)
            ax.set(title=f'{env}, Noisy-{family.upper()}',ylabel='Средний |σ| весов',xlabel='Шаги среды')
            ax.ticklabel_format(axis='x',style='sci',scilimits=(0,0));ax.grid(alpha=.2)
    axes[0,0].legend(frameon=False,fontsize=8)
    fig.tight_layout();fig.savefig(out/'sigma_weights.png');plt.close(fig)


def verify_policy(root=Path('results/policy')):
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    verified=[]
    for marker in sorted(Path(root).glob('*/*/seed_*/COMPLETE')):
        folder=marker.parent;model,c=load_policy_checkpoint(folder/'model.pt')
        stored=read_json(folder/'final_episode_details.json')
        for mode,records in stored.items():
            actual=evaluate_policy(model,c,c.final_episodes,c.final_seed,mode)
            assert actual==records,(folder,mode)
            verified.append({'environment':folder.parts[-3],'condition':folder.parts[-2],
                             'seed':c.seed,'mode':mode,'n_episodes':len(actual),'matches':True})
    assert len(verified)==320 and sum(r['n_episodes'] for r in verified)==9600
    return pd.DataFrame(verified)


def main():
    torch.set_num_threads(1)
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,default=Path('results/policy'))
    p.add_argument('--out',type=Path,default=Path('results/policy_analysis'))
    p.add_argument('--verify',action='store_true')
    args=p.parse_args();runs,curves,_=load_policy_study(args.results)
    summary=summarize_policy(runs);args.out.mkdir(parents=True,exist_ok=True)
    runs.to_csv(args.out/'per_run.csv',index=False);curves.to_csv(args.out/'curves.csv',index=False)
    for key in ['conditions','comparisons','generation_comparisons','mode_comparisons']:
        pd.DataFrame(summary[key]).to_csv(args.out/(key+'.csv'),index=False)
    (args.out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    make_figures(runs,curves,args.out/'figures')
    if args.verify:verify_policy(args.results).to_csv(args.out/'checkpoint_verification.csv',index=False)
    print(json.dumps(summary['conditions'],ensure_ascii=False,indent=2))


if __name__=='__main__':main()
