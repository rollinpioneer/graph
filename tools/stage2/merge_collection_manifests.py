#!/usr/bin/env python3
import argparse,shutil
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--stage1'); ap.add_argument('--new'); ap.add_argument('--output-dir',required=True); a=ap.parse_args(); shutil.copytree(a.new,a.output_dir,dirs_exist_ok=True); print('collection manifests copied and preserved')
if __name__=='__main__': main()
