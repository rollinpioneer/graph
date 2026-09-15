from __future__ import annotations
from .raw_reader import parse_flag, fnum

def rebuild_events(episode_id: str, rows: list[dict], objects: list[str]) -> tuple[list[dict], list[dict]]:
    """Return (per_frame_facts, event_ledger). No case_id, no EOF terminal."""
    ever={o:False for o in objects}
    prev_held={o:None for o in objects}
    open_loss={o:None for o in objects}
    local_counter=0
    facts=[]; ledger=[]
    prev_close=None
    for i,row in enumerate(rows):
        tick=int(float(row["tick"]))
        t=float(row["t"])
        ns=tick*50_000_000
        close=parse_flag(row.get("gripper_closed"))
        rec={"episode_id":episode_id,"tick":tick,"t":t,"available_at_ns":ns,"capture_order":i,
             "gripper_closed":close,"eef":[fnum(row,"eef_x"),fnum(row,"eef_y"),fnum(row,"eef_z")],
             "objects":{}}
        held_now=[]
        for o in objects:
            held=parse_flag(row.get(f"{o}_held"))
            valid=parse_flag(row.get(f"{o}_valid"))
            if held is True:
                ever[o]=True
            pos=[fnum(row,f"{o}_x"), fnum(row,f"{o}_y"), fnum(row,f"{o}_z")]
            vel=[fnum(row,f"{o}_vx"), fnum(row,f"{o}_vy"), fnum(row,f"{o}_vz")]
            tgt=None
            if o=="obj":
                tgt=[fnum(row,"target_obj_x"), fnum(row,"target_obj_y"), fnum(row,"target_obj_z")]
            elif o in ("A","B"):
                tgt=[fnum(row,f"target_{o}_x"), fnum(row,f"target_{o}_y"), None]
            dist=None
            if pos[0] is not None and tgt and tgt[0] is not None and tgt[1] is not None:
                dist=((pos[0]-tgt[0])**2+(pos[1]-tgt[1])**2)**0.5
            loss_kind=None; rec_kind=None; loss_id=open_loss[o]
            if prev_held[o] is True and held is False:
                if close is True:
                    local_counter+=1
                    loss_id=f"{episode_id}:{o}:{tick}:{local_counter}"
                    open_loss[o]=loss_id
                    loss_kind="OBSERVED_SIM_HELD_LOSS"
                    ledger.append(dict(episode_id=episode_id, object_id=o, event="OBSERVED_SIM_HELD_LOSS",
                                       loss_id=loss_id, onset_tick=tick, known_at_ns=ns, kind=loss_kind))
                else:
                    loss_kind="RELEASE_OR_LOSS_AMBIGUOUS"
                    ledger.append(dict(episode_id=episode_id, object_id=o, event="RELEASE_OR_LOSS_AMBIGUOUS",
                                       loss_id=None, onset_tick=tick, known_at_ns=ns, kind=loss_kind))
            if open_loss[o] and prev_held[o] is False and held is True:
                rec_kind="OBSERVED_SIM_HOLD_REESTABLISHED"
                ledger.append(dict(episode_id=episode_id, object_id=o, event="OBSERVED_SIM_HOLD_REESTABLISHED",
                                   loss_id=open_loss[o], onset_tick=tick, known_at_ns=ns, kind=rec_kind))
                open_loss[o]=None
            rec["objects"][o]=dict(held=held, ever_held=ever[o], valid=valid, pos=pos, vel=vel,
                                   target=tgt, distance_actual=dist, dist_eef=fnum(row,f"{o}_dist_eef"),
                                   open_loss_id=loss_id)
            if held is True:
                held_now.append(o)
            prev_held[o]=held
        if len(held_now)==1:
            rec["active_object"]=held_now[0]
        elif len(held_now)==0:
            outstanding=[o for o,v in open_loss.items() if v]
            rec["active_object"]=outstanding[0] if len(outstanding)==1 else None
        else:
            rec["active_object"]=None  # unknown multi
        a=rec["objects"].get("A",{}).get("valid")
        b=rec["objects"].get("B",{}).get("valid")
        if "A" in rec["objects"] and "B" in rec["objects"]:
            rec["success"]= (a is True and b is True)
            rec["terminal_failure"]=None  # no EOF fill
        else:
            rec["success"]= rec["objects"]["obj"]["held"] is False and rec["objects"]["obj"]["distance_actual"] is not None and rec["objects"]["obj"]["distance_actual"]<=0.08 and rec["objects"]["obj"]["ever_held"]
            rec["terminal_failure"]=None
        rec["open_losses"]=dict(open_loss)
        facts.append(rec)
        prev_close=close
    return facts, ledger