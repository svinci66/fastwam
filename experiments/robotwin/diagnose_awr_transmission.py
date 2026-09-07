#!/usr/bin/env python3
"""Read-only final-checkpoint reward/return/advantage/action-weight audit."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from collections import defaultdict

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from experiments.robotwin.diagnose_natural60_reward import DEFAULT_OUT, sha, write_json
from fastwam.rl.models import ResidualActor, ResidualActorConfig, ValueCritic, ValueCriticConfig
from fastwam.rl.replay_buffer import ReplayBuffer
from fastwam.rl.rewards import CompositeRewardConfig
from fastwam.rl.awr_trainer import TaskBalancedBatchSampler, advantage_weights, masked_action_mse

BASE = ROOT / "evaluate_results/robotwin_imagination_restart"
ORIGINAL = BASE / "robotwin_video_expert_multitask3_balanced_pairs_seed44_20260904"
RANK = BASE / "robotwin_video_expert_multitask3_paired_rank_seed44_20260905"
VARIANTS = {
    "control": (ORIGINAL, "no_imagination"),
    "raw025": (ORIGINAL, "with_imagination"),
    "rank025": (RANK, "with_imagination"),
    "rank010": (RANK, "with_imagination010"),
}


def numeric_summary(values):
    a = np.asarray(values, float)
    return {"mean": float(a.mean()), "min": float(a.min()), "max": float(a.max()),
            "quantiles_0_10_50_90_100": np.quantile(a, [0, .1, .5, .9, 1]).tolist()}


def frozen_values(checkpoint, arrays):
    actor = ResidualActor(ResidualActorConfig(**checkpoint["actor_config"])).eval()
    critic = ValueCritic(ValueCriticConfig(**checkpoint["critic_config"])).eval()
    actor.load_state_dict(checkpoint["actor"])
    critic.load_state_dict(checkpoint["critic"])
    values, predictions = [], []
    with torch.inference_mode():
        for lo in range(0, len(arrays["proprio"]), 128):
            hi = lo + 128
            context = torch.from_numpy(np.concatenate([arrays["observation_feature"][lo:hi], arrays["proprio"][lo:hi]], axis=1))
            base = torch.from_numpy(arrays["baseline_actions"][lo:hi])
            language = torch.from_numpy(arrays["language_feature"][lo:hi])
            values.append(critic(context, baseline_actions=base, language_feature=language).numpy())
            predictions.append(actor(context, base, language_feature=language).numpy())
    return np.concatenate(values), np.concatenate(predictions)


def diagnostic_weights(returns, values, schedules, config):
    weight_sum = np.zeros(len(returns), float)
    exposures = np.zeros(len(returns), int)
    preclip_all, weights_all, occurrences = [], [], []
    for indices in schedules:
        idx = np.asarray(indices)
        g, v = torch.tensor(returns[idx]), torch.tensor(values[idx])
        unnorm = torch.exp(torch.clamp((g-v)/config["beta"], max=20))
        preclip = unnorm / unnorm.mean() if config["normalize_advantage_weights"] else unnorm
        weights = advantage_weights(g, v, beta=config["beta"], maximum=config["max_advantage_weight"], normalize=config["normalize_advantage_weights"]).numpy()
        np.add.at(weight_sum, idx, weights)
        np.add.at(exposures, idx, 1)
        preclip_all.extend(preclip.numpy()); weights_all.extend(weights); occurrences.extend(idx.tolist())
    # Task-uniform sampling resets queues each epoch; a large task can leave
    # some transitions unvisited even after three epochs. Preserve that fact.
    means = np.divide(weight_sum, exposures, out=np.zeros_like(weight_sum), where=exposures>0)
    return means, weight_sum, exposures, np.array(weights_all), np.array(preclip_all), np.array(occurrences)


def main():
    torch.set_num_threads(4)
    out = DEFAULT_OUT
    out.mkdir(parents=True, exist_ok=True)
    replay = ReplayBuffer.load(ORIGINAL / "replay")
    arrays = replay.arrays()
    manifest = json.loads((ORIGINAL / "replay/manifest.json").read_text())
    task_map = {v:k for k,v in manifest["provenance"]["task_id_map"].items()}
    n = len(replay)
    assert n == 2934
    tasks = np.array([task_map[t.task_id] for t in replay.transitions])
    behaviors = np.array([t.behavior_mode for t in replay.transitions])
    labels = torch.tensor([t.task_id for t in replay.transitions])
    episode_indices = defaultdict(list)
    for i,t in enumerate(replay.transitions): episode_indices[t.episode_id].append(i)
    phases = np.empty(n, dtype="U8")
    early = np.zeros(n, bool)
    for indices in episode_indices.values():
        for pos,i in enumerate(indices):
            phases[i] = "early" if pos/len(indices)<1/3 else ("middle" if pos/len(indices)<2/3 else "late")
            early[i] = pos<6
    results, per_variant, schedules = {}, {}, None
    source_hashes = {}
    for relative in ("src/fastwam/rl/models.py", "src/fastwam/rl/awr_trainer.py", "src/fastwam/rl/replay_buffer.py", "src/fastwam/rl/rewards.py"):
        source_hashes[relative] = sha(ROOT/relative)
    for name, (root, sub) in VARIANTS.items():
        path = root / "training/seed44" / sub / "checkpoint.pt"
        source_hashes[str(path.relative_to(ROOT))] = sha(path)
        source_hashes[str((root/"replay/manifest.json").relative_to(ROOT))] = sha(root/"replay/manifest.json")
        cp = torch.load(path, map_location="cpu", weights_only=False)
        assert cp["awr_config"]["seed"] == 44 and cp["awr_config"]["epochs"] == 3
        assert cp["replay_manifest_sha256"] == sha(root/"replay/manifest.json")
        assert not cp["awr_config"]["use_goal_conditioning"]
        rr = replay if root==ORIGINAL else ReplayBuffer.load(root/"replay")
        aa = arrays if root==ORIGINAL else rr.arrays()
        for k in ("observation_feature","proprio","baseline_actions","executed_actions","effective_k","language_feature"):
            assert np.array_equal(arrays[k], aa[k]), (name,k)
        for a,b in zip(replay.transitions, rr.transitions):
            assert (a.episode_id,a.transition_index,a.env_seed)==(b.episode_id,b.transition_index,b.env_seed)
        reward, breakdown = rr.relabel_rewards(CompositeRewardConfig(**cp["reward_config"]), imitation_dimension_scales=np.array(json.loads((root/"training/seed44"/sub/"run_config.json").read_text())["imitation_dimension_scales"]))
        returns = rr.monte_carlo_returns(cp["awr_config"]["gamma"], timeout_bootstrap_values={t.episode_id:0.0 for t in rr.transitions if t.truncated}, transition_rewards=reward)
        value, pred = frozen_values(cp, arrays)
        if schedules is None:
            sampler = TaskBalancedBatchSampler(labels, cp["awr_config"]["batch_size"], generator=torch.Generator().manual_seed(cp["awr_config"]["seed"]))
            schedules = [batch for _ in range(3) for batch in sampler]
        mean_w, sum_w, exposure, w_occ, preclip, idx_occ = diagnostic_weights(returns, value, schedules, cp["awr_config"])
        mse = masked_action_mse(torch.from_numpy(pred), torch.from_numpy(arrays["executed_actions"]), torch.from_numpy(arrays["effective_k"])).numpy()
        mask = np.arange(pred.shape[1])[None,:] < arrays["effective_k"][:,None]
        error = (pred - arrays["executed_actions"]) * mask[:,:,None]
        scale = np.asarray(cp["actor_config"]["residual_scale"])
        active = scale>0
        active_mse = np.sum(error[...,active]**2,axis=(1,2))/(mask.sum(1)*pred.shape[2])
        frozen_mse = np.sum(error[...,~active]**2,axis=(1,2))/(mask.sum(1)*pred.shape[2])
        assert np.allclose(active_mse+frozen_mse,mse,atol=1e-7)
        # Gradient of each weighted per-example action MSE wrt output action;
        # excludes minibatch 1/B and parameter Jacobian, explicitly not parameter gradient.
        output_grad_norm = 2*np.linalg.norm(error[...,active].reshape(n,-1),axis=1)/(arrays["effective_k"]*pred.shape[2])
        target = arrays["executed_actions"]-arrays["baseline_actions"]
        low = np.clip(arrays["baseline_actions"]-scale, np.asarray(cp["actor_config"]["action_low"]), np.asarray(cp["actor_config"]["action_high"]))
        high = np.clip(arrays["baseline_actions"]+scale, np.asarray(cp["actor_config"]["action_low"]), np.asarray(cp["actor_config"]["action_high"]))
        low[...,~active] = arrays["baseline_actions"][...,~active]
        high[...,~active] = arrays["baseline_actions"][...,~active]
        assert np.all(low<=high)
        closest = np.clip(arrays["executed_actions"],low,high)
        lower_bound_mse = np.sum((closest-arrays["executed_actions"])**2*mask[:,:,None],axis=(1,2))/(mask.sum(1)*pred.shape[2])
        exceed = np.sum((np.abs(target[...,active]) > scale[active]+1e-6)*mask[:,:,None],axis=(1,2))/(mask.sum(1)*active.sum())
        frozen_error = np.max(np.abs(target[...,~active])*mask[:,:,None],axis=(1,2))
        saturation = np.sum((np.abs((pred-arrays["baseline_actions"])[...,active])/scale[active]>=.95)*mask[:,:,None],axis=(1,2))/(mask.sum(1)*active.sum())
        components = {k:np.array([getattr(b,k) for b in breakdown]) for k in ("success_component","imitation_component","imagination_raw","imagination_applied","total")}
        g, v = returns, value
        summary = {"gamma":cp["awr_config"]["gamma"], "return":numeric_summary(g), "value":numeric_summary(v),
                   "unvisited_transitions":int(np.sum(exposure==0)),
                   "advantage":numeric_summary(g-v), "weight":numeric_summary(w_occ),
                   "clip_fraction":float(np.mean(preclip>cp["awr_config"]["max_advantage_weight"])),
                   "occurrence_ess":float(w_occ.sum()**2/np.square(w_occ).sum()), "occurrences":len(w_occ),
                   "unique_transition_mass_ess":float(sum_w.sum()**2/np.square(sum_w).sum()),
                   "target_exceed_fraction":float(exceed.mean()), "actor_saturation_fraction":float(saturation.mean()), "groups":{}}
        summary["frozen_dimensions_loss_fraction"] = float(np.sum(sum_w*frozen_mse)/np.sum(sum_w*mse))
        summary["action_box_irreducible_loss_fraction"] = float(np.sum(sum_w*lower_bound_mse)/np.sum(sum_w*mse))
        groups = {"all":np.ones(n,bool)}
        for task in sorted(set(tasks)):
            for behavior in ("expert","policy"):
                b = (tasks==task)&(behaviors==behavior)
                groups[f"{task}/{behavior}"] = b
                groups[f"{task}/{behavior}/first6"] = b&early
                for phase in ("early","middle","late"):
                    groups[f"{task}/{behavior}/{phase}"] = b&(phases==phase)
        for group,sel in groups.items():
            summary["groups"][group] = {"transitions":int(sel.sum()), "mean_return":float(g[sel].mean()),
                "unvisited_transitions":int(np.sum(exposure[sel]==0)),
                "mean_value":float(v[sel].mean()), "mean_advantage":float((g-v)[sel].mean()), "mean_weight":float(mean_w[sel].mean()),
                "weight_mass_fraction":float(sum_w[sel].sum()/sum_w.sum()),
                "action_loss_mass_fraction":float((sum_w*mse)[sel].sum()/(sum_w*mse).sum()),
                "active_action_loss_mass_fraction":float((sum_w*active_mse)[sel].sum()/(sum_w*active_mse).sum()),
                "within_group_frozen_loss_fraction":float((sum_w*frozen_mse)[sel].sum()/max((sum_w*mse)[sel].sum(),1e-30)),
                "output_gradient_norm_mass_fraction":float((sum_w*output_grad_norm)[sel].sum()/(sum_w*output_grad_norm).sum()),
                "target_exceed_fraction":float(exceed[sel].mean()), "frozen_target_nonzero_fraction":float(np.mean(frozen_error[sel]>1e-6)),
                "saturation_fraction":float(saturation[sel].mean())}
        results[name] = summary
        per_variant[name] = dict(returns=g,values=v,advantage=g-v,weights=mean_w,weight_mass=sum_w,exposures=exposure,action_mse=mse,active_action_mse=active_mse,frozen_action_mse=frozen_mse,action_box_lower_bound_mse=lower_bound_mse,output_gradient_norm=output_grad_norm,**components)
        del cp, rr, aa
    comparisons = {}
    control = per_variant["control"]
    config = json.loads((ORIGINAL/"training/seed44/no_imagination/run_config.json").read_text())["awr"]
    for name,d in per_variant.items():
        if name=="control":continue
        held_critic = diagnostic_weights(d["returns"], control["values"], schedules, config)[0]
        pairs = []
        pair_groups = defaultdict(dict)
        for episode,indices in episode_indices.items():
            t = replay.transitions[indices[0]]
            pair_groups[t.task_id,t.env_seed][t.behavior_mode] = indices
        for (task_id, seed), groups in pair_groups.items():
            ex, po = groups["expert"],groups["policy"]
            i,j = ex[0],po[0]
            pairs.append({"task":task_map[task_id],"seed":seed,
                "both_initial_transitions_sampled":bool(d["exposures"][i]>0 and d["exposures"][j]>0),
                "control_initial_return_gap":float(control["returns"][i]-control["returns"][j]),
                "treatment_initial_return_gap":float(d["returns"][i]-d["returns"][j]),
                "control_initial_advantage_gap":float(control["advantage"][i]-control["advantage"][j]),
                "treatment_initial_advantage_gap":float(d["advantage"][i]-d["advantage"][j]),
                "control_initial_weight_gap":float(control["weights"][i]-control["weights"][j]),
                "treatment_initial_weight_gap":float(d["weights"][i]-d["weights"][j]),
                "expert_mean_weight_change":float(np.mean(d["weights"][ex]-control["weights"][ex])),
                "policy_mean_weight_change":float(np.mean(d["weights"][po]-control["weights"][po]))})
        def count_gap(kind):
            return sum(p[f"treatment_initial_{kind}_gap"] >= p[f"control_initial_{kind}_gap"]-1e-8 for p in pairs if kind!="weight" or p["both_initial_transitions_sampled"])
        comparisons[name] = {"return_shift":numeric_summary(d["returns"]-control["returns"]),
            "value_shift":numeric_summary(d["values"]-control["values"]),
            "advantage_shift":numeric_summary(d["advantage"]-control["advantage"]),
            "weight_shift_with_control_critic":numeric_summary(held_critic-control["weights"]),
            "weight_shift_with_own_final_critic":numeric_summary(d["weights"]-control["weights"]),
            "nonshrinking_initial_return_gaps":count_gap("return"),"nonshrinking_initial_advantage_gaps":count_gap("advantage"),
            "initial_weight_comparable_pairs":sum(p["both_initial_transitions_sampled"] for p in pairs),
            "nonshrinking_initial_weight_gaps":count_gap("weight"),"pairs":pairs}
    serial_rows = [{"index":i,"episode_id":t.episode_id,"task":tasks[i],"seed":t.env_seed,"behavior":t.behavior_mode,
                   "replan":t.transition_index,"phase":phases[i],"effective_k":t.effective_k} for i,t in enumerate(replay.transitions)]
    write_json(out/"awr_transition_index.json",serial_rows)
    np.savez_compressed(out/"awr_transmission_arrays.npz", **{f"{name}__{k}":v for name,d in per_variant.items() for k,v in d.items()})
    write_json(out/"awr_transmission.json", {"schema":"frozen_final_checkpoint_awr_transmission_v1", "source_hashes":source_hashes,
        "runtime":{"torch":torch.__version__,"numpy":np.__version__,"device":"cpu","threads":torch.get_num_threads()},
        "interpretation":"Final critics/actors evaluated on recreated seed44 three-epoch sampler indices. NOT historical training weights. No updates.",
        "unvisited_weight_convention":"Zero mean weight for transitions not sampled in the three schedules; exposures explicitly recorded. Exclude unsampled starts from paired weight counts.",
        "gradient_scope":"Active-dimension per-example action-output MSE gradient norm excluding 1/B and parameter Jacobian; frozen gripper excluded; not parameter gradients",
        "next_feature_is_current_copy":bool(np.array_equal(arrays["observation_feature"],arrays["next_observation_feature"])),
        "episodes":len(episode_indices),"transitions":n,"variants":results,"comparisons":comparisons})
    print(json.dumps({name:{k:v for k,v in s.items() if k in ("clip_fraction","occurrence_ess","unique_transition_mass_ess","actor_saturation_fraction")} for name,s in results.items()}))
    print(json.dumps({name:{k:v for k,v in s.items() if k.startswith("nonshrinking")} for name,s in comparisons.items()}))


if __name__ == "__main__": main()
