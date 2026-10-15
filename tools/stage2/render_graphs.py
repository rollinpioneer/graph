#!/usr/bin/env python3
import argparse,glob,shutil
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--graph-dir',required=True); ap.add_argument('--output-dir',required=True); a=ap.parse_args();
    import pathlib; pathlib.Path(a.output_dir).mkdir(parents=True,exist_ok=True)
    for p in glob.glob(a.graph_dir+'/*.yaml'): shutil.copy(p,a.output_dir+'/'+pathlib.Path(p).stem+'.yaml')
    print('graph render fallback: YAML copies retained; Graphviz unavailable')
if __name__=='__main__': main()
