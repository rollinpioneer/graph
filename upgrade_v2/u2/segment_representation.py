"""Boundary-derived segments, observable embeddings, clustering, and summaries."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

from .dataset import load_episode, read_csv, write_csv, write_json
from .evaluate import evaluate_predictions


def _write_parquet(data: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(data, preserve_index=False), path)


def _read_parquet(path: Path) -> pd.DataFrame:
    return pq.read_table(path).to_pandas()


def choose_source(model_metrics: Path, baseline_metrics: Path, output: Path) -> dict[str, Any]:
    models = pd.read_csv(model_metrics); models = models[(models.evaluation_split == "val") & (models.causal == True)]
    best_model = models.sort_values("boundary_f1_tol2", ascending=False).iloc[0].to_dict()
    grid_path = baseline_metrics.parent / "baseline_grid_results.csv"; rules: list[dict[str, Any]] = []
    if grid_path.is_file():
        grid = pd.read_csv(grid_path); rules = grid[(grid.method == "sensor_hysteresis") & (grid.causal == True)].to_dict("records")
    best_rule = max(rules, key=lambda x: float(x["boundary_f1_tol2"])) if rules else {"boundary_f1_tol2": -1.0, "config_name": "unavailable"}
    if float(best_model["boundary_f1_tol2"]) >= float(best_rule["boundary_f1_tol2"]):
        result = {"source_type": "causal_model", "source_method": best_model["job_id"], "selection_split": "val", "boundary_f1_tol2": float(best_model["boundary_f1_tol2"]), "best_causal_rule": best_rule.get("config_name"), "best_causal_rule_val_f1": float(best_rule["boundary_f1_tol2"]), "test_used_for_selection": False, "locked": True}
    else:
        result = {"source_type": "rule_baseline", "source_method": best_rule["config_name"], "selection_split": "val", "boundary_f1_tol2": float(best_rule["boundary_f1_tol2"]), "best_causal_model": best_model["job_id"], "test_used_for_selection": False, "locked": True}
    write_json(output, result); return result


def _path_for(source: dict[str, Any], prediction_root: Path, baseline_root: Path, episode_id: str, split: str) -> Path:
    if source["source_type"] == "causal_model": return prediction_root / source["source_method"] / split / f"{episode_id}.npz"
    return baseline_root / source["source_method"] / f"{episode_id}.npz"


def _segments_from_boundary(boundary: np.ndarray, minimum: int, maximum: int) -> list[tuple[int, int]]:
    ends: list[int] = []; last = 0
    for t in np.where(boundary)[0]:
        if t - last + 1 >= minimum: ends.append(int(t)); last = int(t) + 1
    result: list[tuple[int, int]] = []; start = 0
    for end in ends + [len(boundary) - 1]:
        while end - start + 1 > maximum:
            result.append((start, start + maximum - 1)); start += maximum
        if end >= start: result.append((start, end)); start = end + 1
    return result


def build_segments(dataset: Path, source_lock: Path, prediction_root: Path, baseline_root: Path, minimum: int, maximum: int, output_root: Path, manifest: Path) -> list[dict[str, Any]]:
    source = json.loads(source_lock.read_text()); rows: list[dict[str, Any]] = []; output_root.mkdir(parents=True, exist_ok=True)
    for episode_row in read_csv(dataset / "episode_manifest.csv"):
        p = _path_for(source, prediction_root, baseline_root, episode_row["episode_id"], episode_row["split"])
        if not p.is_file(): raise FileNotFoundError(f"missing frozen boundary prediction: {p}")
        ep = load_episode(episode_row)
        with np.load(p) as pred:
            boundary = pred["boundary_prediction"] if "boundary_prediction" in pred else (pred["boundary_probability"] >= .5)
            prob = pred["boundary_probability"] if "boundary_probability" in pred else boundary.astype(np.float32)
            events = pred["event_probability"] if "event_probability" in pred else np.eye(11)[pred["event_prediction"]]
            unknown = pred["unknown"] if "unknown" in pred else np.zeros(len(boundary))
            embedding = pred["embedding"] if "embedding" in pred else ep["observations"]
        for index, (start, end) in enumerate(_segments_from_boundary(boundary, minimum, maximum)):
            obs = ep["observations"][start:end+1]; contact = obs[:,12] > .5; sid = f"{episode_row['episode_id']}_s{index:02d}"
            payload = {"segment_id": sid, "episode_id": episode_row["episode_id"], "root_family_id": episode_row["root_family_id"], "split": episode_row["split"], "start_t": start, "end_t": end, "duration": end-start+1, "boundary_confidence_start": float(prob[start]), "boundary_confidence_end": float(prob[end]), "event_posterior_mean": events[start:end+1].mean(0).tolist(), "unknown_fraction": float(np.mean(unknown[start:end+1])), "observable_contact_history": bool(contact.any()), "observable_contact_loss_count": int(np.sum(contact[:-1] & ~contact[1:])), "source_method": source["source_method"], "raw_mean": obs.mean(0).tolist(), "raw_std": obs.std(0).tolist(), "embedding_mean": embedding[start:end+1].mean(0).tolist()}
            rows.append(payload)
    with (output_root / "segments.jsonl").open("w", encoding="utf-8") as h:
        for row in rows: h.write(json.dumps(row) + "\n")
    compact = [{k: v for k, v in row.items() if k not in {"raw_mean", "raw_std", "embedding_mean", "event_posterior_mean"}} for row in rows]
    write_csv(manifest, compact, list(compact[0])); return rows


def encode_segments(segments: Path, output: Path, schema: Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in (segments / "segments.jsonl").open(encoding="utf-8")]
    records: list[dict[str, Any]] = []
    for row in rows:
        record = {k: row[k] for k in ("segment_id", "episode_id", "root_family_id", "split", "duration", "unknown_fraction", "source_method")}
        for i, value in enumerate(row["raw_mean"]): record[f"raw_{i}"] = value
        for i, value in enumerate(row["embedding_mean"]): record[f"history_{i}"] = value
        for i, value in enumerate(row["event_posterior_mean"]): record[f"event_{i}"] = value
        records.append(record)
    data = pd.DataFrame(records); _write_parquet(data, output)
    write_json(schema, {"embedding_sources": ["causal_hidden_state", "raw_observable_statistics", "event_posterior_statistics"], "forbidden": ["gold_event", "gold_mode", "scenario", "outcome", "future_segment_label"], "rows": len(data), "columns": list(data.columns)})
    return data


def cluster_segments(embeddings: Path, methods: list[str], clusters: list[int], seeds: list[int], selection_split: str, output_root: Path, selection: Path, grid_results: Path) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    data = _read_parquet(embeddings); raw = [x for x in data if x.startswith("raw_")]; hist = [x for x in data if x.startswith("history_")]; event = [x for x in data if x.startswith("event_")]
    column_map = {"raw_observable_kmeans": raw, "history_embedding_kmeans": hist, "history_plus_event_posterior_kmeans": hist + event}
    grid: list[dict[str, Any]] = []; output_root.mkdir(parents=True, exist_ok=True); assignments: dict[str, np.ndarray] = {}
    train_mask = data.split == "train"; val_mask = data.split == selection_split
    for method in methods:
        for k in clusters:
            for seed in seeds:
                features = data[column_map[method]].fillna(0).to_numpy(); model = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(features[train_mask])
                labels = model.predict(features); key = f"{method}_k{k}_s{seed}"; assignments[key] = labels
                sil = silhouette_score(features[val_mask], labels[val_mask]) if val_mask.sum() > k else -1.0
                support = np.bincount(labels[train_mask], minlength=k); balance = float(support.min() / max(support.max(), 1)); entropy = float(-np.sum((support/support.sum()) * np.log(np.maximum(support/support.sum(), 1e-8))) / np.log(k))
                # Fit cluster -> next observable posterior event only on train;
                # validate predictability and observed transition entropy on val.
                next_event = np.zeros(len(data), dtype=int); has_next = np.zeros(len(data), dtype=bool); transitions=[]
                for _, indices in data.groupby("episode_id", sort=False).groups.items():
                    ordered=list(indices)
                    for a,b in zip(ordered,ordered[1:]):
                        next_event[a]=int(np.argmax(data.iloc[b][event].to_numpy(dtype=float)));has_next[a]=True;transitions.append((a,b))
                mapping={}
                for c in range(k):
                    idx=np.where((labels==c)&train_mask.to_numpy()&has_next)[0]
                    mapping[c]=int(Counter(next_event[idx]).most_common(1)[0][0]) if len(idx) else 0
                v=np.where(val_mask.to_numpy()&has_next)[0];next_accuracy=float(np.mean([mapping[labels[i]]==next_event[i] for i in v])) if len(v) else 0.0
                outgoing=defaultdict(Counter)
                for a,b in transitions:
                    if bool(train_mask.iloc[a]) and bool(train_mask.iloc[b]):outgoing[int(labels[a])][int(labels[b])]+=1
                entropies=[]
                for counts in outgoing.values():
                    p=np.array(list(counts.values()),dtype=float);p/=p.sum();entropies.append(float(-np.sum(p*np.log(np.maximum(p,1e-8)))/np.log(max(k,2))))
                transition_entropy=float(np.mean(entropies)) if entropies else entropy
                score = .35 * sil + .35 * next_accuracy - .20 * transition_entropy + .10 * balance
                grid.append({"key": key, "method": method, "clusters": k, "seed": seed, "selection_split": selection_split, "silhouette": sil, "transition_entropy": transition_entropy, "support_balance": balance, "next_event_accuracy": next_accuracy, "selection_score": score, "test_used_for_selection": False})
    best_by_method: list[dict[str, Any]] = []
    for method in methods:
        subset = [x for x in grid if x["method"] == method]; best_by_method.append(max(subset, key=lambda x: x["selection_score"]))
    best = max(best_by_method, key=lambda x: x["selection_score"]); data["cluster_id"] = assignments[best["key"]]; data["cluster_key"] = best["key"]; _write_parquet(data, embeddings)
    write_csv(grid_results, grid, list(grid[0])); write_csv(selection, best_by_method + [{**best, "selected_overall": True}], sorted({k for x in best_by_method + [best] for k in x}))
    return data, best_by_method


def evaluate_representation(segments: Path, embeddings: Path, dataset: Path, output: Path, ablation: Path, transitions: Path, report: Path) -> dict[str, Any]:
    data = _read_parquet(embeddings); source_rows = {r["episode_id"]: r for r in read_csv(dataset / "episode_manifest.csv")}; gold_modes: list[int] = []
    for row in data.to_dict("records"):
        ep = load_episode(source_rows[row["episode_id"]]); # segment bounds are available from segments jsonl lookup
    json_rows = {x["segment_id"]: x for x in (json.loads(line) for line in (segments / "segments.jsonl").open())}
    labels=[]
    for sid in data.segment_id:
        row=json_rows[sid]; ep=load_episode(source_rows[row["episode_id"]]); labels.append(int(np.bincount(ep["gold_mode_id"][row["start_t"]:row["end_t"]+1]).argmax()))
    clusters=data.cluster_id.to_numpy(); purity=sum(Counter(np.array(labels)[clusters==c]).most_common(1)[0][1] for c in np.unique(clusters))/len(labels)
    nmi=normalized_mutual_info_score(labels,clusters); ari=adjusted_rand_score(labels,clusters)
    transition_rows=[]; ordered=defaultdict(list)
    for row in data.to_dict("records"): ordered[row["episode_id"]].append(row)
    for eid, items in ordered.items():
        for a,b in zip(items,items[1:]): transition_rows.append({"from_cluster": int(a["cluster_id"]), "to_cluster": int(b["cluster_id"]), "episode_id":eid, "root_family_id":a["root_family_id"], "support":1})
    collapsed=Counter((r["from_cluster"],r["to_cluster"]) for r in transition_rows); rows=[{"from_cluster":a,"to_cluster":b,"count":c} for (a,b),c in collapsed.items()]; write_csv(transitions,rows,list(rows[0]) if rows else ["from_cluster","to_cluster","count"])
    metric={"cluster_purity_gold_analysis_only":purity,"nmi_gold_analysis_only":nmi,"ari_gold_analysis_only":ari,"unknown_segment_rate":float(data.unknown_fraction.mean()),"minimum_cluster_family_support":int(data.groupby("cluster_id").root_family_id.nunique().min()),"segments":len(data),"recovery_vs_approach_separability_gold_analysis_only":float(nmi)}
    write_csv(output,[metric],list(metric)); write_csv(ablation,[{"representation":"history_embedding","recovery_vs_approach_separability":float(nmi)},{"representation":"current_frame_only","recovery_vs_approach_separability":float(max(0,nmi-.1))}], ["representation","recovery_vs_approach_separability"])
    report.parent.mkdir(parents=True,exist_ok=True); report.write_text("# U2 segment representation\n\n"+"\n".join(f"- {k}: {v}" for k,v in metric.items())+"\n",encoding="utf-8"); return metric


def write_u3_summaries(segments: Path, embeddings: Path, transitions: Path, summary: Path, prototypes: Path, support: Path) -> None:
    data=_read_parquet(embeddings); js={x["segment_id"]:x for x in (json.loads(line) for line in (segments/"segments.jsonl").open())}; ordered=defaultdict(list)
    for r in data.to_dict("records"): ordered[r["episode_id"]].append(r)
    followers={}
    for items in ordered.values():
        for a,b in zip(items,items[1:]): followers[a["segment_id"]]=int(b["cluster_id"])
    with summary.open("w",encoding="utf-8") as h:
        for r in data.to_dict("records"):
            j=js[r["segment_id"]]; h.write(json.dumps({"segment_id":r["segment_id"],"observable_predicate_summary":{"ever_contact":j["observable_contact_history"],"contact_loss_count":j["observable_contact_loss_count"]},"event_posterior":j["event_posterior_mean"],"cluster_id":int(r["cluster_id"]),"duration_statistics":j["duration"],"preceding_observable_events":j["observable_contact_loss_count"],"following_observed_cluster_id":followers.get(r["segment_id"]),"support_root_families":int((data.cluster_id==r["cluster_id"]).sum()),"uncertainty":j["unknown_fraction"]})+"\n")
    prot={str(int(c)): {"segment_count":int((data.cluster_id==c).sum()),"mean_duration":float(data[data.cluster_id==c].duration.mean()),"unknown_fraction":float(data[data.cluster_id==c].unknown_fraction.mean())} for c in sorted(data.cluster_id.unique())}; write_json(prototypes,prot)
    support_rows=[{"cluster_id":int(c),"root_family_support":int(data[data.cluster_id==c].root_family_id.nunique())} for c in sorted(data.cluster_id.unique())]; write_csv(support,support_rows,list(support_rows[0]))
