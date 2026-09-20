#!/usr/bin/env python3
"""Display a planned Stage. This utility never executes experiments."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--stage', required=True, help='For example 1A, 2B or 4B')
    parser.add_argument('--json', action='store_true', dest='as_json')
    args = parser.parse_args()
    try:
        registry = json.loads((args.root / 'configs/stage_registry.json').read_text(encoding='utf-8'))
        stage_id = args.stage.upper().replace('STAGE ', '').strip()
        stage = next((item for item in registry if item['id'] == stage_id), None)
        if stage is None:
            raise ValueError('Unknown Stage; choose ' + ', '.join(item['id'] for item in registry))
        if args.as_json:
            print(json.dumps(stage, ensure_ascii=False, indent=2))
        else:
            print(f"Stage {stage['id']} — {stage['name']}\n")
            print('PLAN ONLY — no experiment has been launched.\n')
            for key in ['purpose', 'prerequisites', 'inputs', 'training_budget', 'evaluation', 'criteria']:
                print(f"{key}: {stage[key]}\n")
            print('Run matrix:\n' + json.dumps(stage['run_matrix'], ensure_ascii=False, indent=2))
            print('\nExecution card: ' + str(args.root / 'stage_cards' / f'stage_{stage_id.lower()}.md'))
            print('Scope: stop after this Stage. Next Stage requires separate authorization.')
        return 0
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f'Planning error: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
