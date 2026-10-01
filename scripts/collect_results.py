#!/usr/bin/env python3
"""Consolidate the four fixed-validation curves and local training metrics."""
from pathlib import Path
import argparse, csv, json
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,default=ROOT/'work');o=p.parse_args()
    work=o.work.resolve();out=work/'results';out.mkdir(parents=True,exist_ok=True)
    rows=[];metrics=[]
    for size in ('3b','7b'):
        for group in ('baseline','sft'):
            run=work/'runs'/(size+'_'+group)
            for path in sorted((run/'fixed_validation/results').glob('iteration_*.json')):
                data=json.loads(path.read_text())
                rows.append({'backbone':size,'group':group,'iteration':int(path.stem.split('_')[-1]),
                             'success_rate':data.get('success_rate',data.get('metrics',{}).get('val/success_rate')),
                             'successes':data.get('successes'),'test_score':data.get('metrics',{}).get('val/test_score'),
                             'manifest_sha256':data.get('manifest_sha256')})
            source=run/'logs/metrics.jsonl'
            if source.exists():
                for line in source.read_text().splitlines():
                    if line.strip():metrics.append(dict(backbone=size,group=group,**json.loads(line)))
    if rows:
        with (out/'fixed_validation.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (out/'metrics_all.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in metrics))
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
    for axis,size in zip(axes,('3b','7b')):
        for group in ('baseline','sft'):
            selected=[r for r in rows if r['backbone']==size and r['group']==group and r['success_rate'] is not None]
            axis.plot([r['iteration'] for r in selected],[100*r['success_rate'] for r in selected],marker='o',label=group)
        axis.set_title('Qwen2.5-'+size.upper());axis.set_xlabel('Outer iteration');axis.grid(alpha=.25);axis.legend()
    axes[0].set_ylabel('Fixed128 success rate (%)');fig.tight_layout();fig.savefig(out/'validation_comparison.png',dpi=180);plt.close(fig)
    print(out)
if __name__=='__main__':main()
