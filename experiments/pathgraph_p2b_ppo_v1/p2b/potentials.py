from __future__ import annotations
import math
from .io_utils import finite

METHODS=('TASK_ONLY','COUNT_PBRS','COUNT_EVENTS_PBRS','GEOM_COUNT_EVENTS_PBRS','GRAPH_COST_PBRS','GRAPH_FULL_PBRS')

class PotentialBank:
    """One bank per episode/env. Uses historical V6 for the two graph arms."""
    def __init__(self,episode_id,cap_class):
        self.episode_id=episode_id;self.cap=cap_class(episode_id)
        self.open=set();self.seen=set();self.anchors={};self.last_clock=None
    def observe(self,state):
        if state['episode_id']!=self.episode_id:raise ValueError('episode mixing')
        key=(state['available_at_ns'],state['capture_order'])
        if self.last_clock is not None and key<=self.last_clock:raise ValueError('noncausal order')
        self.last_clock=key
        if set(state['objects'])!={'A','B'}:raise ValueError('this experiment is dual-object only')
        for e in state.get('events',[]):
            if e['known_at_ns']>state['available_at_ns']:raise ValueError('future event')
            k=(e.get('object_id'),e.get('loss_id'))
            if k[0] not in state['objects'] or not k[1]:raise ValueError('invalid event key')
            if e['kind']=='LOSS':
                if k not in self.seen:self.seen.add(k);self.open.add(k)
            elif e['kind']=='HOLD_REESTABLISHED':self.open.discard(k)
            elif e['kind']!='RECOVERY_STARTED':raise ValueError('unrecognized event')
        if len({x[0] for x in self.open})!=len(self.open):raise ValueError('two open losses on same object')
        valid=sum(o['valid'] is True for o in state['objects'].values())/2.
        count=valid-1.
        events=count-.5*len(self.open)/2.
        geom=0.
        for oid,o in state['objects'].items():
            if type(o['held']) is not bool or type(o['valid']) is not bool:raise ValueError('unknown truth value')
            x,y=map(finite,o['pos'][:2]);tx,ty=map(finite,o['target_xy'])
            d=math.hypot(x-tx,y-ty);goal=(tx,ty)
            if oid not in self.anchors:self.anchors[oid]=(d,goal)
            d0,g0=self.anchors[oid]
            if goal!=g0:raise ValueError('changed goal')
            progress=0. if d0<=1e-9 else max(0.,min(1.,1.-d/d0))
            if o['held'] and not o['valid']:geom+=progress/2.
        v6=self.cap.observe(state)
        values={'TASK_ONLY':0.,'COUNT_PBRS':count,'COUNT_EVENTS_PBRS':events,
                'GEOM_COUNT_EVENTS_PBRS':events+.25*geom,
                'GRAPH_COST_PBRS':-v6['cost_cap'],'GRAPH_FULL_PBRS':v6['psi']}
        for x in values.values():finite(x)
        # Compare phase loss facts to the independent matching event bank.
        phase_lost={oid for oid,o in state['objects'].items() if o['phase'] in ('LOST','RECOVERING')}
        if phase_lost!={oid for oid,_ in self.open}:raise ValueError('event/phase loss mismatch')
        return values,{'matched_open_losses':len(self.open),'v6':v6,'count':valid,'geometry':geom}
