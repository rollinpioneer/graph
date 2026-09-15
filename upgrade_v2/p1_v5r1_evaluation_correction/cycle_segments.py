from __future__ import annotations
from typing import Any

def candidate_cycles(facts: list[dict], ledger: list[dict], object_id: str) -> list[dict]:
    losses=[e for e in ledger if e["object_id"]==object_id and e["event"]=="OBSERVED_SIM_HELD_LOSS"]
    recs={e["loss_id"]: e for e in ledger if e["object_id"]==object_id and e["event"]=="OBSERVED_SIM_HOLD_REESTABLISHED"}
    out=[]
    for j,loss in enumerate(losses):
        onset=int(loss["onset_tick"])
        # last stable held strictly before onset
        s=None
        for fr in facts:
            if fr["tick"]>=onset: break
            o=fr["objects"].get(object_id)
            if o and o["held"] is True:
                s=fr["tick"]
        rec=recs.get(loss["loss_id"])
        e=None; status="NO_OBSERVED_RETURN"
        if rec is None:
            status="NO_OBSERVED_RETURN"
        else:
            # first frame after rec with held True and similar pos to s
            sframe=next((fr for fr in facts if fr["tick"]==s), None) if s is not None else None
            next_onset=int(losses[j+1]["onset_tick"]) if j+1<len(losses) else 10**12
            for fr in facts:
                if fr["tick"]<=int(rec["onset_tick"]): continue
                if fr["tick"]>=next_onset: break
                o=fr["objects"].get(object_id)
                if not o or o["held"] is not True: continue
                if sframe is None:
                    e=fr["tick"]; status="CLOSURE_UNRESOLVED"; break
                sp=sframe["objects"][object_id]["pos"]; op=o["pos"]
                if None in sp or None in op:
                    e=fr["tick"]; status="CLOSURE_UNRESOLVED"; break
                dist=((sp[0]-op[0])**2+(sp[1]-op[1])**2+(sp[2]-op[2])**2)**0.5
                if dist==0.0:
                    e=fr["tick"]; status="EXACT_OBSERVED_TASK_RETURN"; break
                if dist<=1e-3:
                    e=fr["tick"]; status="APPROXIMATE_TASK_RETURN"; break
            else:
                status="NO_OBSERVED_RETURN"
        out.append(dict(loss_id=loss["loss_id"], object_id=object_id, onset_tick=onset,
                        start_tick=s, completion_tick=(int(rec["onset_tick"]) if rec else None),
                        end_tick=e, closure=status, index=j))
    return out