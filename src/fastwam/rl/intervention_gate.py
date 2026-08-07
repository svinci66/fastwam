"""Direct residual-intervention gate trained from causal single-action pairs.

Unlike the replay-level paired gate, every row here represents the exact
replan at which one residual chunk was forced while its matched FastWAM branch
executed the original chunk.  Labels therefore belong to the candidate action
being classified instead of being broadcast over an entire episode.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .models import ActionValueCritic


POSITIVE_LABEL = "rescue"
NEGATIVE_LABEL = "regression"
NON_IMPROVING_LABELS = {
    "local_improve",
    "local_worse",
    "neutral",
    "terminal_tie_unscored",
}


@dataclass(frozen=True)
class InterventionGateExamples:
    context: np.ndarray
    baseline_actions: np.ndarray
    candidate_actions: np.ndarray
    language_feature: np.ndarray | None
    labels: np.ndarray
    weights: np.ndarray
    pair_ids: tuple[str, ...]
    outcome_labels: tuple[str, ...]
    task_names: tuple[str, ...]
    environment_seeds: np.ndarray
    intervention_replans: np.ndarray
    rows: tuple[dict, ...]

    def __len__(self) -> int:
        return int(self.labels.size)


class InterventionGateDataset(Dataset):
    def __init__(self, examples: InterventionGateExamples):
        self.tensors = {
            "context": torch.from_numpy(examples.context),
            "baseline_actions": torch.from_numpy(examples.baseline_actions),
            "candidate_actions": torch.from_numpy(examples.candidate_actions),
            "label": torch.from_numpy(examples.labels),
            "weight": torch.from_numpy(examples.weights),
        }
        if examples.language_feature is not None:
            self.tensors["language_feature"] = torch.from_numpy(
                examples.language_feature
            )

    def __len__(self) -> int:
        return int(self.tensors["label"].shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {key: value[index] for key, value in self.tensors.items()}


def discover_pair_jsonl(roots: Iterable[str | Path]) -> tuple[Path, ...]:
    paths: set[Path] = set()
    for raw_root in roots:
        root = Path(raw_root).expanduser().resolve()
        if root.is_file():
            paths.add(root)
        elif root.is_dir():
            paths.update(root.rglob("accepted_pairs.jsonl"))
        else:
            raise FileNotFoundError(root)
    if not paths:
        raise ValueError("No accepted_pairs.jsonl files were discovered")
    return tuple(sorted(paths))


def _read_pair_rows(paths: Iterable[Path]) -> list[dict]:
    rows: list[dict] = []
    seen: set[str] = set()
    for path in paths:
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") != "accepted":
                continue
            record_dir = str(
                Path(row["residual_record_dir"]).expanduser().resolve()
            )
            if record_dir in seen:
                continue
            seen.add(record_dir)
            row = dict(row)
            row["source_pair_jsonl"] = str(path)
            row["source_line_number"] = line_number
            rows.append(row)
    if not rows:
        raise ValueError("No unique accepted intervention pairs were found")
    return rows


def _load_record(
    record_dir: Path, *, require_applied_residual: bool
) -> tuple[dict[str, np.ndarray], dict]:
    metadata_path = record_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != "robotwin_imagination_transition_v1":
        raise ValueError(f"Unsupported transition metadata: {metadata_path}")
    if require_applied_residual and not bool(
        metadata.get("residual_gate_applied", False)
    ):
        raise ValueError(f"Pair does not point to an applied residual: {record_dir}")
    arrays_path = record_dir / str(metadata["rollout_arrays_file"])
    with np.load(arrays_path, allow_pickle=False) as payload:
        arrays = {key: payload[key] for key in payload.files}
    return arrays, metadata


def _restore_terminal_full_chunk(
    row: dict,
    arrays: dict[str, np.ndarray],
    metadata: dict,
) -> dict[str, np.ndarray]:
    """Recover an audited full candidate chunk from its shadow branch.

    A successful forced branch may terminate after a short prefix.  The paired
    shadow branch was evaluated at the exact same state and retains the full
    actor candidate.  Reuse it only when the exporter-recorded hashes for both
    full chunks agree and the stored terminal prefixes match exactly.
    """

    baseline = np.asarray(arrays["baseline_actions"])
    residual = np.asarray(arrays["candidate_residual_actions"])
    if not bool(metadata.get("terminated", False)):
        return arrays
    target_step = metadata.get("target_step")
    if target_step is not None and baseline.shape[0] >= int(target_step):
        return arrays
    shadow_dir = Path(row["baseline_record_dir"]).expanduser().resolve()
    shadow_arrays, shadow_metadata = _load_record(
        shadow_dir, require_applied_residual=False
    )
    hash_keys = ("baseline_actions_sha256", "candidate_residual_actions_sha256")
    if any(
        not metadata.get(key) or metadata.get(key) != shadow_metadata.get(key)
        for key in hash_keys
    ):
        raise ValueError(
            f"Terminal pair lacks matching full candidate hashes: {shadow_dir}"
        )
    shadow_baseline = np.asarray(shadow_arrays.get("baseline_actions"))
    shadow_residual = np.asarray(shadow_arrays.get("candidate_residual_actions"))
    if (
        baseline.ndim != 2
        or residual.shape != baseline.shape
        or shadow_baseline.ndim != 2
        or shadow_residual.shape != shadow_baseline.shape
        or shadow_baseline.shape[1:] != baseline.shape[1:]
        or shadow_baseline.shape[0] <= baseline.shape[0]
        or not np.array_equal(shadow_baseline[: baseline.shape[0]], baseline)
        or not np.array_equal(shadow_residual[: residual.shape[0]], residual)
    ):
        raise ValueError(
            f"Terminal candidate prefix does not match its full shadow chunk: {shadow_dir}"
        )
    restored = dict(arrays)
    restored["baseline_actions"] = shadow_baseline
    restored["candidate_residual_actions"] = shadow_residual
    return restored


def load_intervention_gate_examples(
    pair_paths: Iterable[str | Path],
    *,
    include_non_improving: bool = True,
    non_improving_weight: float = 0.25,
) -> InterventionGateExamples:
    """Load only the forced intervention transition from each accepted pair."""

    if not np.isfinite(non_improving_weight) or non_improving_weight <= 0.0:
        raise ValueError("non_improving_weight must be finite and positive")
    paths = discover_pair_jsonl(pair_paths)
    rows = _read_pair_rows(paths)
    selected: list[tuple[dict, float, float]] = []
    for row in rows:
        outcome = str(row.get("label", ""))
        if outcome == POSITIVE_LABEL:
            selected.append((row, 1.0, 1.0))
        elif outcome == NEGATIVE_LABEL:
            selected.append((row, 0.0, 1.0))
        elif include_non_improving and outcome in NON_IMPROVING_LABELS:
            selected.append((row, 0.0, float(non_improving_weight)))
    if not selected:
        raise ValueError("No supported intervention outcome labels were found")
    if {label for _, label, _ in selected} != {0.0, 1.0}:
        raise ValueError("Intervention gate data must contain rescue and negative examples")

    contexts: list[np.ndarray] = []
    baselines: list[np.ndarray] = []
    candidates: list[np.ndarray] = []
    languages: list[np.ndarray | None] = []
    labels: list[float] = []
    raw_weights: list[float] = []
    pair_ids: list[str] = []
    outcome_labels: list[str] = []
    task_names: list[str] = []
    environment_seeds: list[int] = []
    intervention_replans: list[int] = []
    selected_rows: list[dict] = []
    for row, label, weight in selected:
        record_dir = Path(row["residual_record_dir"]).expanduser().resolve()
        arrays, metadata = _load_record(
            record_dir, require_applied_residual=True
        )
        required = (
            "residual_observation_feature",
            "proprio",
            "baseline_actions",
            "candidate_residual_actions",
        )
        missing = [key for key in required if key not in arrays]
        if missing:
            raise ValueError(f"Missing arrays {missing} in {record_dir}")
        arrays = _restore_terminal_full_chunk(row, arrays, metadata)
        feature = np.asarray(
            arrays["residual_observation_feature"], dtype=np.float32
        ).reshape(-1)
        proprio = np.asarray(arrays["proprio"], dtype=np.float32).reshape(-1)
        baseline = np.asarray(arrays["baseline_actions"], dtype=np.float32)
        residual = np.asarray(
            arrays["candidate_residual_actions"], dtype=np.float32
        )
        if baseline.shape != residual.shape or baseline.ndim != 2:
            raise ValueError(
                f"Action shape mismatch in {record_dir}: {baseline.shape}, {residual.shape}"
            )
        values = (feature, proprio, baseline, residual)
        if any(not np.all(np.isfinite(value)) for value in values):
            raise ValueError(f"Non-finite intervention input in {record_dir}")
        language = arrays.get("language_feature")
        if language is not None:
            language = np.asarray(language, dtype=np.float32).reshape(-1)
            if not np.all(np.isfinite(language)):
                raise ValueError(f"Non-finite language feature in {record_dir}")
        contexts.append(np.concatenate((feature, proprio)).astype(np.float32))
        baselines.append(baseline)
        candidates.append(baseline + residual)
        languages.append(language)
        labels.append(label)
        raw_weights.append(weight)
        task = str(row["task_name"])
        seed = int(row["environment_seed"])
        replan = int(row["intervention_replan_idx"])
        pair_ids.append(f"{task}:{seed}:{replan}")
        outcome_labels.append(str(row["label"]))
        task_names.append(task)
        environment_seeds.append(seed)
        intervention_replans.append(replan)
        selected_rows.append(row)

    language_present = [value is not None for value in languages]
    if any(language_present) and not all(language_present):
        raise ValueError("Language features must be present for every pair or none")
    label_array = np.asarray(labels, dtype=np.float32)
    weight_array = np.asarray(raw_weights, dtype=np.float32)
    for label in (0.0, 1.0):
        selection = label_array == label
        weight_array[selection] /= float(weight_array[selection].sum())
    weight_array *= len(weight_array) / float(weight_array.sum())
    return InterventionGateExamples(
        context=np.stack(contexts).astype(np.float32),
        baseline_actions=np.stack(baselines).astype(np.float32),
        candidate_actions=np.stack(candidates).astype(np.float32),
        language_feature=(
            None
            if not all(language_present)
            else np.stack([value for value in languages if value is not None]).astype(
                np.float32
            )
        ),
        labels=label_array,
        weights=weight_array,
        pair_ids=tuple(pair_ids),
        outcome_labels=tuple(outcome_labels),
        task_names=tuple(task_names),
        environment_seeds=np.asarray(environment_seeds, dtype=np.int64),
        intervention_replans=np.asarray(intervention_replans, dtype=np.int64),
        rows=tuple(selected_rows),
    )


def audit_intervention_gate_coverage(
    examples: InterventionGateExamples,
) -> dict[str, object]:
    positive = examples.labels == 1.0
    negative = ~positive
    # Build the groups explicitly; the unusual class imbalance matters more
    # than transition count for any claimed held-out validation.
    positive_groups = sorted(
        {
            (examples.task_names[index], int(examples.environment_seeds[index]))
            for index in np.flatnonzero(positive)
        }
    )
    negative_groups = sorted(
        {
            (examples.task_names[index], int(examples.environment_seeds[index]))
            for index in np.flatnonzero(negative)
        }
    )
    independent_validation_possible = (
        len(positive_groups) >= 2 and len(negative_groups) >= 2
    )
    return {
        "num_examples": len(examples),
        "positive_examples": int(positive.sum()),
        "negative_examples": int(negative.sum()),
        "positive_task_seed_groups": [list(value) for value in positive_groups],
        "negative_task_seed_groups": [list(value) for value in negative_groups],
        "independent_seed_validation_possible": independent_validation_possible,
        "deployment_ready": False,
        "deployment_blocker": (
            None
            if independent_validation_possible
            else "rescue examples do not span at least two task/seed groups"
        ),
    }


def _gate_logits(
    model: ActionValueCritic, batch: dict[str, torch.Tensor]
) -> torch.Tensor:
    return model(
        batch["context"],
        batch["baseline_actions"],
        batch["candidate_actions"],
        batch.get("language_feature"),
    )


def train_intervention_gate_ensemble(
    models: tuple[ActionValueCritic, ActionValueCritic],
    examples: InterventionGateExamples,
    *,
    device: torch.device | str,
    epochs: int = 30,
    batch_size: int = 32,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
    max_grad_norm: float = 1.0,
    seed: int = 42,
) -> list[list[dict[str, float]]]:
    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0.0:
        raise ValueError("epochs, batch_size, and learning_rate must be positive")
    dataset = InterventionGateDataset(examples)
    device = torch.device(device)
    histories: list[list[dict[str, float]]] = []
    for model_index, model in enumerate(models):
        model_seed = int(seed) + 1009 * model_index
        torch.manual_seed(model_seed)
        generator = torch.Generator().manual_seed(model_seed)
        loader = DataLoader(
            dataset,
            batch_size=min(int(batch_size), len(dataset)),
            shuffle=True,
            generator=generator,
        )
        model.to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        history: list[dict[str, float]] = []
        for epoch in range(int(epochs)):
            model.train()
            loss_sum = 0.0
            weight_sum = 0.0
            for cpu_batch in loader:
                batch = {key: value.to(device) for key, value in cpu_batch.items()}
                optimizer.zero_grad(set_to_none=True)
                logits = _gate_logits(model, batch)
                losses = nn.functional.binary_cross_entropy_with_logits(
                    logits, batch["label"], reduction="none"
                )
                loss = torch.sum(losses * batch["weight"]) / torch.sum(
                    batch["weight"]
                )
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                batch_weight = float(batch["weight"].sum().detach().cpu())
                loss_sum += float(loss.detach().cpu()) * batch_weight
                weight_sum += batch_weight
            history.append({"epoch": float(epoch), "loss": loss_sum / weight_sum})
        model.eval().requires_grad_(False)
        histories.append(history)
    return histories


@torch.no_grad()
def predict_intervention_gate(
    models: tuple[ActionValueCritic, ActionValueCritic],
    examples: InterventionGateExamples,
    *,
    device: torch.device | str,
    batch_size: int = 256,
) -> np.ndarray:
    dataset = InterventionGateDataset(examples)
    loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)))
    device = torch.device(device)
    columns: list[list[np.ndarray]] = [[], []]
    for cpu_batch in loader:
        batch = {key: value.to(device) for key, value in cpu_batch.items()}
        for index, model in enumerate(models):
            model.to(device).eval()
            columns[index].append(torch.sigmoid(_gate_logits(model, batch)).cpu().numpy())
    return np.stack([np.concatenate(values) for values in columns], axis=1)


def summarize_intervention_fit(
    examples: InterventionGateExamples, probabilities: np.ndarray
) -> dict[str, object]:
    if probabilities.shape != (len(examples), 2):
        raise ValueError("probabilities must have shape [N,2]")
    conservative = probabilities.min(axis=1)
    positive = examples.labels == 1.0
    negative = ~positive
    max_negative = max(0.5, float(np.max(conservative[negative])))
    threshold = float(
        np.nextafter(np.float32(max_negative), np.float32(1.0), dtype=np.float32)
    )
    approved = conservative >= threshold
    return {
        "evaluation_scope": "diagnostic_resubstitution_only",
        "recommended_threshold_for_smoke_only": threshold,
        "true_positive_rate": float(np.mean(approved[positive])),
        "false_positive_rate": float(np.mean(approved[negative])),
        "weighted_brier": float(
            np.average(
                np.square(conservative - examples.labels), weights=examples.weights
            )
        ),
        "positive_probability_mean": float(np.mean(conservative[positive])),
        "negative_probability_mean": float(np.mean(conservative[negative])),
        "ensemble_disagreement_max": float(
            np.max(np.abs(probabilities[:, 0] - probabilities[:, 1]))
        ),
        "pair_rows": [
            {
                "pair_id": pair_id,
                "outcome_label": outcome,
                "label": int(label),
                "probabilities": [float(value) for value in probability],
                "conservative_probability": float(min(probability)),
            }
            for pair_id, outcome, label, probability in zip(
                examples.pair_ids,
                examples.outcome_labels,
                examples.labels,
                probabilities,
            )
        ],
    }


def intervention_decision_metrics(
    outcome_labels: Iterable[str], approvals: Iterable[bool]
) -> dict[str, float | int]:
    labels = np.asarray(list(outcome_labels), dtype=object)
    decisions = np.asarray(list(approvals), dtype=bool)
    if labels.shape != decisions.shape:
        raise ValueError("outcome_labels and approvals must have the same length")
    rescue = labels == POSITIVE_LABEL
    regression = labels == NEGATIVE_LABEL
    if not np.any(rescue) or not np.any(regression):
        raise ValueError("decision metrics require rescue and regression examples")
    return {
        "rescue_total": int(rescue.sum()),
        "rescue_approved": int(np.sum(decisions & rescue)),
        "rescue_recall": float(np.mean(decisions[rescue])),
        "regression_total": int(regression.sum()),
        "regression_approved": int(np.sum(decisions & regression)),
        "regression_false_approval_rate": float(np.mean(decisions[regression])),
    }
