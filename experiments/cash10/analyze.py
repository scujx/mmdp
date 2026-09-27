"""Analyze forty Raw/Search evaluations using whole training-seed pairing."""
from pathlib import Path
import argparse
import csv
import importlib.util
import json
import numpy as np

spec = importlib.util.spec_from_file_location('cash10_statistics', Path(__file__).with_name('statistics.py'))
stats = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stats)
POLICIES = {'flat': 'flat_ppo', 'mmdp': 'mmdp_ppo', 'flat-aa': 'flat_aa_ppo', 'mmdp-aa': 'mmdp_aa_ppo'}


def calculate(directories):
    values = np.full((10, 4, 4, 200), np.nan)
    for directory in directories:
        metadata = json.loads((directory/'evaluation.json').read_text())
        if metadata['actual_training_transitions'] != 2000384: raise ValueError('Final 2M policies required')
        seed = int(metadata['training_seed'])
        if seed not in range(42, 52): raise ValueError('Unexpected training seed')
        family = stats.FAMILIES.index(POLICIES[metadata['policy']])
        with (directory/'episodes.csv').open(newline='') as handle:
            for row in csv.DictReader(handle):
                environment = int(row['environment_seed'])-994000
                if not 0 <= environment < 200: raise ValueError('Unexpected test instance')
                cap = stats.MODES.index(int(row['planning_cap']))
                key = seed-42, family, cap, environment
                if not np.isnan(values[key]): raise ValueError('Duplicate episode')
                value = float(row['episode_return'])
                if not np.isfinite(value): raise ValueError('Nonfinite return')
                values[key] = value
    if not np.isfinite(values).all(): raise ValueError('Need 40 final policies x four budgets x 200 common instances')
    result, _, _ = stats.calculate(values.mean(axis=3))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', nargs='+', type=Path, required=True, help='Evaluation directories, each with evaluation.json and episodes.csv')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    result = calculate(args.input)
    with args.output.open('x') as handle: json.dump(result, handle, indent=2, allow_nan=False)
    print('Completed paired analysis; no saved reference results were used.')
