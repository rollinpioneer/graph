#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math

def norm(v): return math.sqrt(sum(float(x)*float(x) for x in v))
def xy_error(eef,obj): return math.hypot(float(eef[0])-float(obj[0]),float(eef[1])-float(obj[1]))
def z_error(eef,obj,grasp_height=0.13): return abs((float(eef[2])-float(obj[2]))-float(grasp_height))
def above_target(obj,approach_height=0.16): return [float(obj[0]),float(obj[1]),float(obj[2])+float(approach_height)]
def grasp_target(obj,grasp_height=0.13): return [float(obj[0]),float(obj[1]),float(obj[2])+float(grasp_height)]
def preclose_ok(eef,obj,obj_vel,consecutive_ticks,xy_tol=0.008,z_tol=0.008,speed_max=0.08,required_ticks=3):
    ok=xy_error(eef,obj)<=xy_tol and z_error(eef,obj)<=z_tol and norm(obj_vel)<=speed_max
    return ok and int(consecutive_ticks)>=int(required_ticks)
def next_attempt(current,max_attempts):
    current=int(current); max_attempts=int(max_attempts)
    if current<1 or max_attempts<1: raise ValueError('attempts must be positive')
    return current+1 if current<max_attempts else None
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--json',required=True); a=ap.parse_args(); d=json.loads(a.json)
    print(json.dumps({'above':above_target(d['object_position']),'grasp':grasp_target(d['object_position']),'xy_error':xy_error(d['eef_position'],d['object_position']),'z_error':z_error(d['eef_position'],d['object_position']),'preclose_ok':preclose_ok(d['eef_position'],d['object_position'],d['object_velocity'],d.get('consecutive_ticks',0))},indent=2))
if __name__=='__main__': main()

def in_workspace(p):
    x,y,z = float(p[0]), float(p[1]), float(p[2])
    return -1.0 <= x <= 1.0 and -0.8 <= y <= 0.8 and 0.30 <= z <= 1.20
