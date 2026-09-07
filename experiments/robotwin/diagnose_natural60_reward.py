#!/usr/bin/env python3
"""Fixed-population reward-content diagnostic. Never runs a policy or optimizer."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
SOURCE = "robotwin_wan_head_multitask4_smoke2_20260901"
PREFIX = {"open_microwave": 16, "hanging_mug": 13, "place_can_basket": 5}
DEFAULT_OUT = ROOT / "evaluate_results/robotwin_imagination_restart/robotwin_natural60_diagnostic_20260907"
PROTOCOL = ROOT / "experiments/robotwin/ROBOTWIN_NATURAL60_DIAGNOSTIC_PROTOCOL_20260907.md"
VAE = ROOT.parent / "checkpoints/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temp.replace(path)


def auc(labels, scores):
    labels, scores = np.asarray(labels, bool), np.asarray(scores, float)
    if not labels.any() or labels.all() or not np.isfinite(scores).all():
        raise ValueError("AUC requires finite scores and both classes")
    d = scores[labels, None] - scores[~labels][None, :]
    return float(np.mean((d > 0) + 0.5 * (d == 0)))


def donor_scores(matrix, indices=None):
    """Resample both targets and donors; exclude repeated copies of self identity."""
    indices = np.arange(len(matrix)) if indices is None else np.asarray(indices)
    values = matrix[np.ix_(indices, indices)]
    mask = indices[:, None] != indices[None, :]
    if np.any(mask.sum(1) == 0):
        raise ValueError("no non-self donor")
    return np.where(mask, values, 0).sum(1) / mask.sum(1)


def prepare(out):
    inventory = out / "inventory.json"
    if inventory.exists():
        raise FileExistsError("Inventory already frozen; reuse it, never silently refreeze")
    base = ROOT / "evaluate_results/robotwin/robotwin_uncond_3cam_384" / SOURCE
    source = ROOT / "evaluate_results/robotwin_imagination_restart" / SOURCE
    rows, images, episodes = [], {}, []
    for task, count in PREFIX.items():
        ep_roots = sorted((base / task / "imagination_transitions" / task / "policy").glob("episode_*"))
        assert len(ep_roots) == 20
        for ep in ep_roots:
            metas = sorted(ep.glob("replan_*/metadata.json"))
            first = json.loads(metas[0].read_text())
            lengths = []
            valid = 0
            for k, mp in enumerate(metas):
                m = json.loads(mp.read_text())
                assert m["replan_idx"] == k and m["schema_version"] == "robotwin_imagination_trajectory_v2"
                assert m["environment_seed"] == first["environment_seed"]
                assert m["episode_success"] == first["episode_success"] and m["action_mode"] == "policy"
                assert m["action_noise_std"] == 0 and m["target_step"] == 24
                lengths.append(m["effective_k"])
                aligned = m["trajectory_alignment_valid"]
                valid += int(aligned)
                r = {"task": task, "episode": first["trial_idx"], "seed": first["environment_seed"],
                     "success": first["episode_success"], "replan": k, "effective_k": m["effective_k"],
                     "aligned": aligned, "metadata": str(mp.relative_to(ROOT)), "metadata_sha256": sha(mp),
                     "arrays": str((mp.parent / m["rollout_arrays_file"]).relative_to(ROOT)),
                     "arrays_sha256": sha(mp.parent / m["rollout_arrays_file"])}
                if aligned:
                    assert m["effective_k"] == 24
                    for kind in ("predicted", "actual"):
                        assert m[f"{kind}_trajectory_action_offsets"] == [0, 4, 8, 12, 16, 20, 24]
                        keys = []
                        for rel in m[f"{kind}_trajectory_files"]:
                            p = mp.parent / rel
                            key = sha(p)
                            images.setdefault(key, str(p.relative_to(ROOT)))
                            keys.append(key)
                        r[kind] = keys
                rows.append(r)
            assert valid >= count and sum(lengths[:count]) == count * 24 and sum(lengths) > count * 24
            episodes.append({"task": task, "episode": first["trial_idx"], "seed": first["environment_seed"],
                             "success": first["episode_success"], "action_steps": sum(lengths), "valid_chunks": valid})
    assert len(episodes) == 60 and sum(e["success"] for e in episodes) == 18
    write_json(inventory, {"schema": "natural60_inventory_v1", "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                          "protocol_sha256": sha(PROTOCOL), "vae_path": str(VAE), "vae_sha256": sha(VAE),
                          "source_manifest_sha256": sha(source / "local_expert_manifest.json"),
                          "prefix_chunks": PREFIX, "episodes": episodes, "records": rows, "images": images})
    print(json.dumps({"episodes": len(episodes), "records": len(rows), "unique_images": len(images), "inventory_sha256": sha(inventory)}), flush=True)


def encode(out, device, limit):
    import torch
    from experiments.robotwin.validate_frozen_plan_vae_reward import WanVaeFrameEncoder
    d = json.loads((out / "inventory.json").read_text())
    assert sha(PROTOCOL) == d["protocol_sha256"] and sha(VAE) == d["vae_sha256"]
    cache = out / "head_latents"
    cache.mkdir(exist_ok=True)
    provenance = {"vae_sha256": d["vae_sha256"], "dtype": "bf16", "device": device,
                  "encoder_sha256": sha(ROOT / "experiments/robotwin/validate_frozen_plan_vae_reward.py"),
                  "feature": "full-composite single-frame Wan VAE, float32 head region"}
    pp = cache / "provenance.json"
    if pp.exists():
        assert json.loads(pp.read_text()) == provenance
    else:
        write_json(pp, provenance)
    torch.set_num_threads(4)
    encoder = WanVaeFrameEncoder(vae_path=VAE, device=device, dtype=torch.bfloat16)
    started, done = time.monotonic(), 0
    for key, rel in d["images"].items():
        dest = cache / f"{key}.npy"
        if dest.exists():
            continue
        assert sha(ROOT / rel) == key
        feature = encoder.encode(ROOT / rel)["head"]
        assert np.isfinite(feature).all()
        temp = dest.with_suffix(".tmp")
        with temp.open("wb") as f:
            np.save(f, feature, allow_pickle=False)
        temp.replace(dest)
        encoder.cache.clear()
        done += 1
        if done % 100 == 0 or done == 1:
            print(json.dumps({"encoded_this_run": done, "seconds": round(time.monotonic()-started, 1), "total_required": len(d["images"])}), flush=True)
        if limit and done >= limit:
            break
    print(json.dumps({"encoded_this_run": done, "cached": len(list(cache.glob('*.npy'))), "required": len(d["images"])}), flush=True)


def unit_deltas(frames):
    delta = frames[1:].astype(np.float32) - frames[0].astype(np.float32)
    delta = delta.reshape(6, -1)
    norms = np.linalg.norm(delta, axis=1)
    if not np.isfinite(norms).all() or np.any(norms <= 1e-8):
        raise ValueError("Degenerate delta: stop instead of silently changing the sample")
    return delta / norms[:, None], float(np.sqrt(np.mean(delta ** 2)))


def bootstrap_metrics(task_data, repeats=10000):
    rng = np.random.default_rng(20260907)
    names = ["matched", "donor", "time_reversed", "latent_motion", "action_variation", "action_magnitude", "negative_duration", "content_delta"]
    point, boot = {}, {}
    for task, d in task_data.items():
        labels = np.array(d["labels"], bool)
        matrix = np.array(d["matrix"])
        scores = {k: np.array(d[k]) for k in names if k not in ("donor", "content_delta")}
        scores["donor"] = donor_scores(matrix)
        point[task] = {k: auc(labels, v) for k, v in scores.items()}
        point[task]["content_delta"] = point[task]["matched"] - point[task]["donor"]
        arrays = {k: [] for k in names}
        pos, neg = np.flatnonzero(labels), np.flatnonzero(~labels)
        for _ in range(repeats):
            idx = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
            values = {k: auc(labels[idx], v[idx]) for k, v in scores.items() if k != "donor"}
            values["donor"] = auc(labels[idx], donor_scores(matrix, idx))
            values["content_delta"] = values["matched"] - values["donor"]
            for k in names:
                arrays[k].append(values[k])
        boot[task] = {k: np.array(v) for k, v in arrays.items()}
    point["macro"] = {k: float(np.mean([v[k] for v in point.values()])) for k in names}
    boot["macro"] = {k: np.mean([v[k] for v in boot.values()], axis=0) for k in names}
    return {task: {k: {"estimate": value, "ci95": np.quantile(boot[task][k], [.025, .975]).tolist()} for k, value in vals.items()} for task, vals in point.items()}


def score(out, repeats):
    d = json.loads((out / "inventory.json").read_text())
    assert sha(PROTOCOL) == d["protocol_sha256"]
    cache = out / "head_latents"
    missing = [k for k in d["images"] if not (cache / f"{k}.npy").exists()]
    if missing:
        raise ValueError(f"{len(missing)} missing frozen image features")
    buckets = defaultdict(list)
    for r in d["records"]:
        assert sha(ROOT / r["metadata"]) == r["metadata_sha256"]
        assert sha(ROOT / r["arrays"]) == r["arrays_sha256"]
        with np.load(ROOT/r["arrays"],allow_pickle=False) as arrays:
            assert np.array_equal(arrays["baseline_actions"],arrays["executed_actions"])
        if r["aligned"]:
            buckets[r["task"], r["replan"]].append(r)
    chunks = []
    matrices = {t: np.zeros((20, 20), np.float64) for t in PREFIX}
    for (task, k), records in sorted(buckets.items()):
        records.sort(key=lambda r: r["episode"])
        pp, aa, motion = [], [], []
        for r in records:
            p = np.stack([np.load(cache / f"{key}.npy", allow_pickle=False) for key in r["predicted"]])
            a = np.stack([np.load(cache / f"{key}.npy", allow_pickle=False) for key in r["actual"]])
            pu, _ = unit_deltas(p)
            au, am = unit_deltas(a)
            pp.append(pu); aa.append(au); motion.append(am)
        pp, aa = np.stack(pp), np.stack(aa)
        cross = np.einsum("itd,jtd->ij", aa, pp, optimize=True) / 6
        reversed_scores = np.einsum("itd,itd->i", aa, pp[:, ::-1], optimize=True) / 6
        if k < PREFIX[task]:
            assert len(records) == 20 and [r["episode"] for r in records] == list(range(20))
            matrices[task] += cross / PREFIX[task]
        for i, r in enumerate(records):
            arrays = np.load(ROOT / r["arrays"], allow_pickle=False)
            actions = arrays["executed_actions"][:24]
            active = np.delete(actions, [6, 13], axis=1)
            row = {key: r[key] for key in ("task", "episode", "seed", "success", "replan")}
            row.update(matched=float(cross[i, i]), donor=float((cross[i].sum()-cross[i, i])/(len(records)-1)) if len(records)>1 else None,
                       donor_count=len(records)-1, time_reversed=float(reversed_scores[i]), latent_motion=motion[i],
                       action_variation=float(np.sqrt(np.mean(np.diff(active, axis=0)**2))),
                       action_magnitude=float(np.sqrt(np.mean(active**2))))
            if "task_progress" in arrays:
                progress = arrays["task_progress"][:, 3]
                row["progress_start"] = float(progress[0])
                row["progress_delta"] = float(progress[-1]-progress[0])
                row["setback"] = bool(np.min(progress - progress[0]) <= -.02)
            arrays.close()
            chunks.append(row)
    write_json(out / "chunk_scores.json", chunks)
    grouped = defaultdict(list)
    for r in chunks:
        grouped[r["task"], r["episode"]].append(r)
    episode_rows, task_data = [], {}
    for e in d["episodes"]:
        rows = grouped[e["task"], e["episode"]]
        prefix = [r for r in rows if r["replan"] < PREFIX[e["task"]]]
        result = dict(e)
        for scope, selected in (("prefix", prefix), ("full", rows)):
            result[scope] = {name: float(np.mean([r[name] for r in selected if r[name] is not None])) for name in ("matched", "donor", "time_reversed", "latent_motion", "action_variation", "action_magnitude")}
            result[scope]["negative_duration"] = -e["action_steps"]
        episode_rows.append(result)
    for task in PREFIX:
        eps = sorted([r for r in episode_rows if r["task"] == task], key=lambda r:r["episode"])
        td = {"labels": [r["success"] for r in eps], "matrix": matrices[task].tolist()}
        td.update({name: [r["prefix"][name] for r in eps] for name in eps[0]["prefix"] if name != "donor"})
        task_data[task] = td
    stats = bootstrap_metrics(task_data, repeats)
    full = {task: {name: auc([r["success"] for r in episode_rows if r["task"]==task], [r["full"][name] for r in episode_rows if r["task"]==task]) for name in episode_rows[0]["full"]} for task in PREFIX}
    progress_rows = []
    for (task, ep), rows in grouped.items():
        if task != "open_microwave":
            continue
        row = {"episode": ep, "seed": rows[0]["seed"], "success": rows[0]["success"], "chunks":len(rows)}
        for name in ("matched", "donor", "latent_motion"):
            x, y = [r[name] for r in rows], [r["progress_delta"] for r in rows]
            rho = spearmanr(x, y).statistic if np.ptp(x)>0 and np.ptp(y)>0 else np.nan
            row[name+"_progress_rho"] = float(rho) if np.isfinite(rho) else None
        row["setback_replans"] = [r["replan"] for r in rows if r["setback"]]
        ordered = sorted(rows,key=lambda r:r["replan"])
        for name in ("matched","donor"):
            for event in (True,False):
                differences = [current[name]-previous[name] for previous,current in zip(ordered,ordered[1:]) if current["setback"]==event and current["replan"]==previous["replan"]+1]
                label = "setback" if event else "non_setback"
                row[f"{name}_{label}_mean_score_change"] = float(np.mean(differences)) if differences else None
                row[f"{name}_{label}_score_drop_fraction"] = float(np.mean(np.array(differences)<0)) if differences else None
        progress_rows.append(row)
    m = stats["macro"]
    gates = {"auc_at_least_065":m["matched"]["estimate"]>=.65,
             "auc_lower_above_chance":m["matched"]["ci95"][0]>.5,
             "content_delta_at_least_005":m["content_delta"]["estimate"]>=.05,
             "content_delta_lower_positive":m["content_delta"]["ci95"][0]>0,
             "two_positive_tasks":sum(stats[t]["matched"]["estimate"]>.5 for t in PREFIX)>=2,
             "no_supported_reverse_task":all(stats[t]["matched"]["ci95"][1]>=.5 for t in PREFIX)}
    decision = "promising_development_signal" if all(gates.values()) else "inconclusive_do_not_train"
    if m["content_delta"]["ci95"][1]<=0:
        decision = "reject_incremental_content_on_fixed_prefix"
    elif m["matched"]["ci95"][1]<.65:
        decision = "exclude_registered_useful_prefix_auc"
    write_json(out / "episode_scores.json", episode_rows)
    write_json(out / "prefix_cross_scores.json", task_data)
    # Reproduce all previously scored policy chunks from these same sources.
    legacy_path = ROOT / "evaluate_results/robotwin_imagination_restart/robotwin_video_expert_multitask3_balanced_pairs_seed44_20260904/merged_wan_vae_head_rewards.json"
    legacy = json.loads(legacy_path.read_text())
    lookup = {(r["task"],r["seed"],r["replan"]):r["matched"] for r in chunks}
    errors = []
    for pair in legacy["pairs"]:
        for r in pair["fastwam_failure"]["per_replan"]:
            key = (pair["task"],pair["environment_seed"],r["replan_idx"])
            if key in lookup:
                errors.append(abs(lookup[key]-r["camera_scores"]["head"]))
    assert errors and max(errors)<1e-4, (len(errors),max(errors,default=None))
    write_json(out / "reward_diagnostic.json", {"schema":"natural60_reward_diagnostic_v1", "inventory_sha256":sha(out/"inventory.json"),
               "bootstrap_replicates":repeats, "bootstrap_unit":"episode; within-task outcome-stratified; donors resampled", "prefix":stats,
               "legacy_score_reproduction":{"chunks_compared":len(errors),"maximum_absolute_error":max(errors),"source_sha256":sha(legacy_path)},
               "input_validation":{"episodes":len(d["episodes"]),"records":len(d["records"]),"scored_complete_chunks":len(chunks),"incomplete_chunks_excluded":len(d["records"])-len(chunks),"all_source_metadata_and_array_hashes_match":True,"all_executed_actions_equal_baseline":True,"nondegenerate_scored_deltas":True},
               "full_episode_auc_descriptive":full, "microwave_progress":progress_rows, "gates":gates, "decision":decision,
               "limitations":["seen development data; failures overlap training", "full scores confounded by duration", "three fixed tasks only", "setbacks are not irreversible failure labels", "no state-based failure localization for mug/basket", "no online efficacy claim"]})
    print(json.dumps({"macro":stats["macro"], "gates":gates, "decision":decision}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["prepare", "encode", "score"])
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=0, help="encoding smoke limit; never changes population")
    p.add_argument("--bootstrap", type=int, default=10000)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    if a.mode == "prepare": prepare(a.output)
    elif a.mode == "encode": encode(a.output, a.device, a.limit)
    else: score(a.output, a.bootstrap)
