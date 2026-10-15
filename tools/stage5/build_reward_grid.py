#!/usr/bin/env python3
import argparse,json,yaml
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);a=p.parse_args(); c=yaml.safe_load(open(a.config)); s=c['search']; i=0
 with open(a.output,'w') as f:
  for l in s['lambda_values']:
   for e in s['eta_values']:
    for b in s['beta_values']:
     for conf in s['confidence_values']:
      f.write(json.dumps({'config_id':f'cfg_{i:03d}','lambda':l,'eta':e,'beta':b,'confidence':conf})+'\n'); i+=1
if __name__=='__main__': main()
