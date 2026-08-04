"""Run RoboTwin's evaluator with small, validated FastWAM compatibility hooks.

RoboTwin's upstream evaluator hard-codes expert trajectory filtering.  That is
useful for its official benchmark, but it can silently advance the environment
seed when motion planning is nondeterministic.  Controlled corruption studies
need a stricter invariant: every policy variant must start from the same scene.

This entrypoint leaves the upstream checkout untouched.  It validates and
patches narrowly scoped source fragments before executing the evaluator:

* make expert filtering configurable through ``expert_check``;
* allow a fixed instruction when expert filtering is disabled;
* expose the environment seed in logs and process state for provenance.
* optionally replay a strict, task-specific environment-seed manifest;
* optionally select an official instruction deterministically from its seed.

The exact-fragment checks intentionally fail when upstream changes, rather than
silently applying an incompatible patch.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _replace_once(source: str, old: str, new: str, *, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(
            f"Expected exactly one RoboTwin {label} fragment, found {count}. "
            "The upstream evaluator may have changed."
        )
    return source.replace(old, new, 1)


def patch_eval_policy_source(source: str) -> str:
    source = _replace_once(
        source,
        "    expert_check = True\n",
        "    expert_check = bool(usr_args.get(\"expert_check\", True))\n",
        label="expert-check",
    )
    source = _replace_once(
        source,
        "    suc_test_seed_list = []\n",
        "    suc_test_seed_list = []\n"
        "    strict_environment_seeds = None\n"
        "    seed_manifest_path = usr_args.get(\"environment_seed_manifest_path\")\n"
        "    if seed_manifest_path is not None and str(seed_manifest_path).strip().lower() not in {\"\", \"none\", \"null\"}:\n"
        "        with open(str(seed_manifest_path), \"r\", encoding=\"utf-8\") as manifest_file:\n"
        "            manifest_payload = __import__(\"json\").load(manifest_file)\n"
        "        manifest_entry = manifest_payload.get(args[\"task_name\"])\n"
        "        if manifest_entry is None:\n"
        "            raise ValueError(f\"Seed manifest has no task {args['task_name']!r}\")\n"
        "        if isinstance(manifest_entry, dict):\n"
        "            manifest_entry = manifest_entry.get(\"seeds\")\n"
        "        strict_environment_seeds = [int(value) for value in manifest_entry]\n"
        "        if len(strict_environment_seeds) != test_num:\n"
        "            raise ValueError(\n"
        "                f\"Seed manifest for {args['task_name']} has {len(strict_environment_seeds)} seeds, expected {test_num}\"\n"
        "            )\n"
        "    deterministic_instruction_by_seed = bool(\n"
        "        usr_args.get(\"deterministic_instruction_by_seed\", False)\n"
        "    )\n",
        label="seed-manifest",
    )
    source = _replace_once(
        source,
        "        args[\"render_freq\"] = 0\n",
        "        args[\"render_freq\"] = 0\n"
        "        if strict_environment_seeds is not None:\n"
        "            now_seed = strict_environment_seeds[succ_seed]\n",
        label="strict-seed-selection",
    )
    source = _replace_once(
        source,
        "            except UnStableError as e:\n",
        "            except UnStableError as e:\n"
        "                if strict_environment_seeds is not None:\n"
        "                    raise RuntimeError(\n"
        "                        f\"Strict environment seed {now_seed} became unstable\"\n"
        "                    ) from e\n",
        label="strict-unstable-seed",
    )
    source = _replace_once(
        source,
        "            except Exception as e:\n",
        "            except Exception as e:\n"
        "                if strict_environment_seeds is not None:\n"
        "                    raise RuntimeError(\n"
        "                        f\"Strict environment seed {now_seed} failed expert validation\"\n"
        "                    ) from e\n",
        label="strict-expert-exception",
    )
    source = _replace_once(
        source,
        "        else:\n"
        "            now_seed += 1\n"
        "            args[\"render_freq\"] = render_freq\n"
        "            continue\n\n"
        "        args[\"render_freq\"] = render_freq\n",
        "        else:\n"
        "            if strict_environment_seeds is not None:\n"
        "                raise RuntimeError(\n"
        "                    f\"Strict environment seed {now_seed} is not expert-feasible\"\n"
        "                )\n"
        "            now_seed += 1\n"
        "            args[\"render_freq\"] = render_freq\n"
        "            continue\n\n"
        "        args[\"render_freq\"] = render_freq\n",
        label="strict-infeasible-seed",
    )
    source = _replace_once(
        source,
        "            suc_test_seed_list.append(now_seed)\n",
        "            suc_test_seed_list.append(now_seed)\n"
        "            os.environ[\"FASTWAM_ENVIRONMENT_SEED\"] = str(now_seed)\n"
        "            print(f\"FASTWAM_ACCEPTED_ENV_SEED episode_id={now_id} seed={now_seed}\", flush=True)\n",
        label="accepted-seed",
    )
    source = _replace_once(
        source,
        "        episode_info_list = [episode_info[\"info\"]]\n"
        "        results = generate_episode_descriptions(args[\"task_name\"], episode_info_list, test_num)\n"
        "        instruction = np.random.choice(results[0][instruction_type])\n",
        "        fixed_instruction = usr_args.get(\"fixed_instruction\")\n"
        "        if fixed_instruction is not None and str(fixed_instruction).strip().lower() not in {\"\", \"none\", \"null\"}:\n"
        "            instruction = str(fixed_instruction)\n"
        "        else:\n"
        "            if not expert_check:\n"
        "                raise ValueError(\"fixed_instruction is required when expert_check is disabled\")\n"
        "            episode_info_list = [episode_info[\"info\"]]\n"
        "            results = generate_episode_descriptions(args[\"task_name\"], episode_info_list, test_num)\n"
        "            instruction_candidates = results[0][instruction_type]\n"
        "            if deterministic_instruction_by_seed:\n"
        "                instruction_rng = np.random.default_rng(int(now_seed))\n"
        "                instruction = str(instruction_rng.choice(instruction_candidates))\n"
        "            else:\n"
        "                instruction = np.random.choice(instruction_candidates)\n"
        "        print(\n"
        "            f\"FASTWAM_EVAL_INSTRUCTION episode_id={now_id} seed={now_seed} instruction={instruction!r}\",\n"
        "            flush=True,\n"
        "        )\n",
        label="instruction-selection",
    )
    return source


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--upstream-script", required=True)
    known, forwarded = parser.parse_known_args()

    upstream_script = Path(known.upstream_script).resolve()
    if not upstream_script.is_file():
        raise FileNotFoundError(f"RoboTwin evaluator not found: {upstream_script}")

    source = patch_eval_policy_source(upstream_script.read_text(encoding="utf-8"))
    # Direct execution of ``script/eval_policy.py`` normally places its script
    # directory first on sys.path (needed for ``import test_render``).
    sys.path.insert(0, str(upstream_script.parent))
    sys.argv = [str(upstream_script), *forwarded]
    globals_dict = {
        "__name__": "__main__",
        "__file__": str(upstream_script),
        "__package__": None,
        "__cached__": None,
    }
    exec(compile(source, str(upstream_script), "exec"), globals_dict)


if __name__ == "__main__":
    main()
