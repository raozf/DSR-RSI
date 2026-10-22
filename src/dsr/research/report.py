"""Generate numerical-control research artifacts from verified receipts only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from .control_evolution import HarnessState, file_digest, read_json, SCOPE


def generate(workspace, *, summary_path=None):
    root = Path(workspace).resolve()
    result = read_json(root / 'report.json')
    current = HarnessState(root / 'state').resolve()
    if current['revision'] != result['current_revision']:
        raise ValueError('Report revision is not current')
    for path, expected in result['report_evidence'].items():
        if file_digest(Path(path)) != expected:
            raise ValueError('Changed report evidence')
    rows = []
    for round_record in result['rounds']:
        directory = root / 'campaign/rounds' / f"round-{round_record['index'] + 1:04d}"
        record = read_json(directory / 'selection.json')['output']
        parent = record['metrics'][record['parent_id']]
        survivor = record['metrics'][record['survivor_id']]
        costs = [read_json(path) for path in directory.rglob('cost.json') if 'worker' not in path.parts]
        rows.append({
            'round': round_record['index'] + 1,
            'parent_id': record['parent_id'], 'selected_id': record['survivor_id'],
            'parent_score': parent['normalized_score'], 'selected_score': survivor['normalized_score'],
            'paired_gain': survivor['normalized_score'] - parent['normalized_score'],
            'decision': record['decision'],
            'episodes': sum(c['episodes'] for c in costs),
            'steps': sum(c['simulation_steps'] for c in costs),
            'execution_wall_seconds': sum(c['wall_seconds'] for c in costs),
        })
    analysis = root / 'analysis'
    analysis.mkdir(exist_ok=True)
    with (analysis / 'evolution.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # Matplotlib is a reporting-only optional dependency; no handwritten chart
    # or generated values are substituted if it is absent.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    rounds = [r['round'] for r in rows]
    ax.plot(rounds, [r['parent_score'] for r in rows], 'o-', label='Incumbent, fresh paired conditions')
    ax.plot(rounds, [r['selected_score'] for r in rows], 'o-', label='Selected candidate, same conditions')
    ax.set(xlabel='Evolution round (different fresh cohort each round)', ylabel='Numerical control score',
           title='DSR source evolution: damped point mass only', xticks=rounds, ylim=(0, 1))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(analysis / 'evolution.png', dpi=150)
    plt.close(fig)
    costs = [read_json(path) for path in root.rglob('cost.json') if 'worker' not in path.parts]
    summary = {k: result[k] for k in ('status', 'scope', 'qualification', 'current_revision', 'current_harness',
               'initial_mean_score', 'selected_mean_score', 'initial_successes', 'selected_successes',
               'report_cases', 'paired_mean_gain', 'paired_gain_standard_error', 'api_calls', 'gpu_hours', 'limitations')}
    summary.update(schema='dsr.research-summary/v1', experiment=str(root), rounds=rows,
        total_executed_episodes=sum(c['episodes'] for c in costs),
        total_simulation_steps=sum(c['simulation_steps'] for c in costs),
        execution_wall_seconds=sum(c['wall_seconds'] for c in costs),
        plot=str(analysis / 'evolution.png'), csv=str(analysis / 'evolution.csv'),
        statistical_scope='Descriptive paired sample only; deterministic single campaign, no superiority or novelty claim')
    (analysis / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    manifest = {}
    for path in sorted(root.rglob('*')):
        if (not path.is_file() or path.is_symlink() or 'isolation-runtime' in path.parts
                or ('worker' in path.parts and 'root' in path.relative_to(root).parts)
                or path.name in {'artifact_manifest.json'} or path.suffix == '.log'):
            continue
        manifest[str(path.relative_to(root))] = {'sha256': file_digest(path), 'bytes': path.stat().st_size}
    (analysis / 'artifact_manifest.json').write_text(json.dumps({'schema':'dsr.artifact-manifest/v1', 'scope':SCOPE, 'files':manifest}, indent=2) + '\n')
    if summary_path:
        path = Path(summary_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', required=True)
    parser.add_argument('--summary')
    args = parser.parse_args()
    print(json.dumps(generate(args.workspace, summary_path=args.summary), indent=2))


if __name__ == '__main__':
    main()
