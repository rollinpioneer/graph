#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import yaml
def main():
 p=argparse.ArgumentParser();p.add_argument('--input-index',type=Path,required=True);p.add_argument('--inference-protocol',type=Path,required=True);p.add_argument('--estimands',type=Path,required=True);p.add_argument('--checkpoint-manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();index=yaml.safe_load(a.input_index.read_text());protocol=yaml.safe_load(a.inference_protocol.read_text());est=yaml.safe_load(a.estimands.read_text());n=len(a.checkpoint_manifest.read_text().splitlines())-1
 text=f'''# Experimental Setup\n\nThe locked benchmark uses the transport dual-order and recovery tasks with frozen train/validation/test partitions. The manual graph supplies nodes and typed legal, failure, and recovery edges. We independently ran {n} frozen reward-model checkpoints with ensemble seeds {protocol['model']['ensemble_seeds']} in evaluation/inference mode, with history length {protocol['model']['history_steps']}.\n\n`content_group_id` is the statistics unit. Core reward comparisons include linear and sequential baselines and the predeclared structural ablations. We use {est['bootstrap_resamples']} stratified, paired content-group bootstrap resamples. Controlled symbolic stress tests demonstrate fixed graph semantics and are not described as real-robot generalization. Policy evidence remains secondary/mixed.\n''';a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(text,encoding='utf-8')
if __name__=='__main__':main()
