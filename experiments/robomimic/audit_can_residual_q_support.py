#!/usr/bin/env python3
"""Audit residual proposals with pairwise Q and a train-only KNN support gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.robomimic.train_can_pairwise_q import PairwiseQ, _auc, _prepare_features
from experiments.robomimic.train_can_residual_actor import ResidualActor, _features, _predict


def _normalized_support_features(
    state: np.ndarray,
    action: np.ndarray,
    normalization: dict[str, np.ndarray],
) -> np.ndarray:
    return _features(state, action, normalization).astype(np.float32)


def kth_neighbor_distance(
    queries: np.ndarray,
    references: np.ndarray,
    *,
    k: int,
    exclude_identical_index: bool = False,
    chunk_size: int = 256,
) -> np.ndarray:
    if k <= 0 or k > len(references) - int(exclude_identical_index):
        raise ValueError("k is incompatible with the reference count")
    # Expanding (query - reference) to a three-dimensional tensor costs several
    # gigabytes for frozen vision features.  The quadratic identity below is
    # equivalent, but its largest temporary is only query_count x reference_count.
    references64 = references.astype(np.float64, copy=False)
    reference_squared_norm = np.sum(references64 * references64, axis=1)
    feature_dim = references.shape[1]
    distances = []
    for start in range(0, len(queries), chunk_size):
        query = queries[start : start + chunk_size].astype(np.float64, copy=False)
        squared = (
            np.sum(query * query, axis=1)[:, None]
            + reference_squared_norm[None, :]
            - 2.0 * query @ references64.T
        ) / feature_dim
        np.maximum(squared, 0.0, out=squared)
        if exclude_identical_index:
            rows = np.arange(len(query))
            columns = np.arange(start, start + len(query))
            squared[rows, columns] = np.inf
        kth_squared = np.partition(squared, k - 1, axis=1)[:, k - 1]
        distances.append(np.sqrt(kth_squared))
    return np.concatenate(distances)


@torch.no_grad()
def _q_advantage(
    checkpoint: dict[str, Any],
    state: np.ndarray,
    base_action: np.ndarray,
    candidate_action: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model = PairwiseQ(checkpoint["input_dim"], tuple(checkpoint["hidden_dims"])).to(device)
    model.load_state_dict(checkpoint["model"])
    normalization = checkpoint["normalization"]
    base = _prepare_features(
        state,
        base_action,
        state_mode=checkpoint["state_mode"],
        **normalization,
    )
    candidate = _prepare_features(
        state,
        candidate_action,
        state_mode=checkpoint["state_mode"],
        **normalization,
    )
    model.eval()
    advantages = []
    for start in range(0, len(state), batch_size):
        base_batch = torch.from_numpy(base[start : start + batch_size]).to(device)
        candidate_batch = torch.from_numpy(candidate[start : start + batch_size]).to(device)
        advantages.append((model(candidate_batch) - model(base_batch)).cpu().numpy())
    return np.concatenate(advantages)


def _rate(mask: np.ndarray) -> float:
    return float(np.mean(mask)) if len(mask) else float("nan")


def audit(args: argparse.Namespace) -> dict[str, Any]:
    with np.load(args.dataset, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    device = torch.device(
        args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    actor_checkpoint = torch.load(args.actor_checkpoint, map_location=device, weights_only=False)
    actor = ResidualActor(
        actor_checkpoint["input_dim"],
        actor_checkpoint["output_dim"],
        tuple(actor_checkpoint["hidden_dims"]),
        actor_checkpoint["residual_scale"],
        preserve_last_action_dim=actor_checkpoint.get("preserve_last_action_dim", False),
        action_dim=actor_checkpoint.get("action_dim"),
    ).to(device)
    actor.load_state_dict(actor_checkpoint["model"])
    actor_features = _features(
        arrays["state"], arrays["base_action_chunk"], actor_checkpoint["normalization"]
    )
    residual = _predict(
        actor,
        actor_features,
        device=device,
        batch_size=args.batch_size,
        output_shape=tuple(actor_checkpoint["target_shape"]),
    )
    proposal = np.clip(arrays["base_action_chunk"] + residual, -1.0, 1.0)

    train = arrays["source_split"] == "train"
    valid = arrays["source_split"] == "valid"
    improved = arrays["has_improvement"].astype(bool)
    support_normalization = actor_checkpoint["normalization"]
    train_base_support = _normalized_support_features(
        arrays["state"][train], arrays["base_action_chunk"][train], support_normalization
    )
    train_target_action = np.clip(
        arrays["base_action_chunk"][train] + arrays["target_residual_chunk"][train],
        -1.0,
        1.0,
    )
    positive_train = improved[train]
    train_target_support = _normalized_support_features(
        arrays["state"][train][positive_train],
        train_target_action[positive_train],
        support_normalization,
    )
    references = np.concatenate([train_base_support, train_target_support], axis=0)
    leave_one_out = kth_neighbor_distance(
        references,
        references,
        k=args.k,
        exclude_identical_index=True,
        chunk_size=args.knn_chunk_size,
    )
    threshold = float(np.quantile(leave_one_out, args.support_quantile))

    valid_base_support = _normalized_support_features(
        arrays["state"][valid], arrays["base_action_chunk"][valid], support_normalization
    )
    valid_proposal_support = _normalized_support_features(
        arrays["state"][valid], proposal[valid], support_normalization
    )
    base_distance = kth_neighbor_distance(
        valid_base_support, references, k=args.k, chunk_size=args.knn_chunk_size
    )
    proposal_distance = kth_neighbor_distance(
        valid_proposal_support, references, k=args.k, chunk_size=args.knn_chunk_size
    )
    in_support = (base_distance <= threshold) & (proposal_distance <= threshold)

    rng = np.random.default_rng(args.seed)
    random_action = rng.uniform(-1.0, 1.0, size=arrays["base_action_chunk"][valid].shape).astype(
        np.float32
    )
    random_support = _normalized_support_features(
        arrays["state"][valid], random_action, support_normalization
    )
    random_distance = kth_neighbor_distance(
        random_support, references, k=args.k, chunk_size=args.knn_chunk_size
    )

    q_checkpoint = torch.load(args.q_checkpoint, map_location=device, weights_only=False)
    train_advantage = _q_advantage(
        q_checkpoint,
        arrays["state"][train],
        arrays["base_action_chunk"][train],
        proposal[train],
        device=device,
        batch_size=args.batch_size,
    )
    negative_train = ~improved[train]
    if args.q_advantage_threshold is None:
        q_advantage_threshold = float(
            np.quantile(train_advantage[negative_train], args.q_threshold_quantile)
        )
        q_threshold_source = "train_zero_target_quantile"
    else:
        q_advantage_threshold = float(args.q_advantage_threshold)
        q_threshold_source = "explicit"

    advantage = _q_advantage(
        q_checkpoint,
        arrays["state"][valid],
        arrays["base_action_chunk"][valid],
        proposal[valid],
        device=device,
        batch_size=args.batch_size,
    )
    q_accept = advantage > q_advantage_threshold
    intervene = in_support & q_accept
    valid_improved = improved[valid]
    residual_norm = np.linalg.norm(residual[valid].reshape(np.count_nonzero(valid), -1), axis=1)
    threshold_sweep = []
    for quantile in (0.50, 0.75, 0.90, 0.95):
        sweep_threshold = float(np.quantile(train_advantage[negative_train], quantile))
        sweep_intervene = in_support & (advantage > sweep_threshold)
        threshold_sweep.append(
            {
                "train_zero_target_quantile": quantile,
                "threshold": sweep_threshold,
                "overall_intervention_rate": _rate(sweep_intervene),
                "improvement_target_intervention_rate": _rate(
                    sweep_intervene[valid_improved]
                ),
                "zero_target_intervention_rate": _rate(sweep_intervene[~valid_improved]),
            }
        )

    output_support = args.output_support.expanduser().resolve()
    output_support.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_support,
        references=references,
        threshold=np.asarray(threshold, dtype=np.float32),
        k=np.asarray(args.k, dtype=np.int32),
        quantile=np.asarray(args.support_quantile, dtype=np.float32),
        state_mean=support_normalization["state_mean"],
        state_std=support_normalization["state_std"],
        action_mean=support_normalization["action_mean"],
        action_std=support_normalization["action_std"],
    )

    report = {
        "dataset": str(args.dataset.resolve()),
        "actor_checkpoint": str(args.actor_checkpoint.resolve()),
        "q_checkpoint": str(args.q_checkpoint.resolve()),
        "support_index": str(output_support),
        "device": str(device),
        "k": args.k,
        "support_quantile": args.support_quantile,
        "support_threshold": threshold,
        "train_support_reference_count": len(references),
        "q_advantage_threshold": q_advantage_threshold,
        "q_threshold_source": q_threshold_source,
        "q_threshold_quantile": args.q_threshold_quantile,
        "train_zero_target_q_advantage_count": int(np.count_nonzero(negative_train)),
        "train_zero_target_q_accept_rate": _rate(
            train_advantage[negative_train] > q_advantage_threshold
        ),
        "validation": {
            "states": int(np.count_nonzero(valid)),
            "improvement_targets": int(np.count_nonzero(valid_improved)),
            "zero_targets": int(np.count_nonzero(~valid_improved)),
            "base_in_support_rate": _rate(base_distance <= threshold),
            "proposal_in_support_rate": _rate(proposal_distance <= threshold),
            "joint_in_support_rate": _rate(in_support),
            "random_action_rejection_rate": _rate(random_distance > threshold),
            "q_positive_advantage_rate": _rate(q_accept),
            "q_advantage_mean": float(np.mean(advantage)),
            "q_advantage_median": float(np.median(advantage)),
            "q_advantage_target_auc": _auc(valid_improved.astype(np.int8), advantage),
            "intervention_rate": _rate(intervene),
            "intervention_rate_on_improvement_targets": _rate(intervene[valid_improved]),
            "intervention_rate_on_zero_targets": _rate(intervene[~valid_improved]),
            "residual_norm_mean": float(np.mean(residual_norm)),
            "residual_norm_p95": float(np.quantile(residual_norm, 0.95)),
            "residual_norm_max": float(np.max(residual_norm)),
            "q_threshold_sweep": threshold_sweep,
        },
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--actor-checkpoint", type=Path, required=True)
    parser.add_argument("--q-checkpoint", type=Path, required=True)
    parser.add_argument("--output-support", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--support-quantile", type=float, default=0.95)
    parser.add_argument("--q-advantage-threshold", type=float)
    parser.add_argument("--q-threshold-quantile", type=float, default=0.95)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--knn-chunk-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260820)
    args = parser.parse_args()
    report = audit(args)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
