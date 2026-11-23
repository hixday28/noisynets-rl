"""Сводка изолированных запусков A3C на CartPole по фактическим файлам."""

import argparse
import json
from pathlib import Path

import numpy as np


def summarize(root):
    rows=[]
    for marker in sorted(root.glob('**/seed_*/COMPLETE')):
        folder=marker.parent
        c=json.loads((folder/'config.json').read_text())
        scores=json.loads((folder/'final_returns.json').read_text())
        meta=json.loads((folder/'metadata.json').read_text())
        treatment=json.loads((folder/'treatment.json').read_text())
        row={'condition':folder.parent.name,'seed':c['seed'],'steps':c['steps'],
             'learning_rate':c['learning_rate'],'anneal_lr':treatment['anneal_lr'],
             'seconds':meta['elapsed_seconds']}
        for mode,values in scores.items():
            row[mode]=float(np.mean(values))
            row[mode+'_sd']=float(np.std(values,ddof=1))
            row[mode+'_min']=float(np.min(values))
        row['meets_475_both']=row['behaviour']>=475 and row['mean_greedy']>=475
        rows.append(row)
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    p.add_argument('--out',type=Path)
    args=p.parse_args()
    rows=summarize(args.root)
    text=json.dumps(rows,ensure_ascii=False,indent=2)
    if args.out:args.out.write_text(text+'\n')
    print(text)


if __name__=='__main__':main()
