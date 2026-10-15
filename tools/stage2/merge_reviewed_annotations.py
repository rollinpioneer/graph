#!/usr/bin/env python3
import argparse,shutil,json
from pathlib import Path
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--primary-dir',required=True); ap.add_argument('--review-dir'); ap.add_argument('--output-dir',required=True); ap.add_argument('--report'); a=ap.parse_args(); shutil.copytree(a.primary_dir,a.output_dir,dirs_exist_ok=True); out={'merged':len(list(Path(a.output_dir).glob('*.json'))),'review_overrides':0};
    if a.report: Path(a.report).write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__': main()
