import argparse,json,csv
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument('--supervision-dir',required=True);p.add_argument('--output-md',required=True);a=p.parse_args(); rows=list(csv.DictReader(open(Path(a.supervision_dir)/'tables/episode_manifest.csv'))); Path(a.output_md).write_text('# Supervision Summary\n\n- episodes: %d\n- content groups: %d\n- transport_recovery representatives: %d\n- transport_dual_order mechanism probes: %d\n'%(len(rows),len({r['content_group_id'] for r in rows}),sum(r['task_id']=='transport_recovery' for r in rows),sum(r['task_id']=='transport_dual_order' for r in rows)))
if __name__=='__main__':main()
