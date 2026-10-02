#!/usr/bin/env python3
"""Export the selected experiment's validation curve and training metrics."""
from pathlib import Path
import argparse, csv, json

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, default=ROOT/'work')
    parser.add_argument('--size', choices=['3b', '7b'], default='3b')
    parser.add_argument('--method', choices=['sft', 'baseline'], default='sft')
    options = parser.parse_args()
    work = options.work.resolve()
    run = work/'runs'/f'{options.size}_{options.method}'
    out = work/'results'/run.name
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted((run/'fixed_validation/results').glob('iteration_*.json')):
        data = json.loads(path.read_text(encoding='utf-8'))
        iteration = int(path.stem.split('_')[-1])
        if (data.get('validation_id') != 'alfworld-valid-seen-128-v1' or
                data.get('tasks') != 128 or data.get('iteration') != iteration or
                not isinstance(data.get('successes'), int) or
                not 0 <= data['successes'] <= 128 or
                data.get('success_rate') != data['successes']/128):
            raise RuntimeError('Evaluation does not match this project\'s fixed 128-task protocol: '+path.name)
        rows.append({'backbone': options.size, 'group': options.method,
                     'iteration': iteration,
                     'success_rate': data.get('success_rate', data.get('metrics', {}).get('val/success_rate')),
                     'successes': data.get('successes'),
                     'test_score': data.get('native_metrics', {}).get('val/test_score'),
                     'validation_id': data.get('validation_id')})
    if not rows:
        raise RuntimeError(f'No evaluations found for {run.name}; no empty success chart is produced.')
    with (out/'fixed_validation.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    source = run/'logs/metrics.jsonl'
    metrics = []
    if source.is_file():
        for line in source.read_text(encoding='utf-8').splitlines():
            if line.strip(): metrics.append(dict(backbone=options.size, group=options.method, **json.loads(line)))
    (out/'metrics_all.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in metrics), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axis = plt.subplots(figsize=(7, 4))
    selected = [row for row in rows if row['success_rate'] is not None]
    axis.plot([row['iteration'] for row in selected], [100*row['success_rate'] for row in selected], marker='o', label=run.name)
    axis.set(title='Qwen2.5-'+options.size.upper(), xlabel='Outer iteration', ylabel='Fixed128 success rate (%)')
    axis.grid(alpha=.25); axis.legend(); fig.tight_layout()
    fig.savefig(out/'validation.png', dpi=180); plt.close(fig)
    print(out)


if __name__ == '__main__': main()
