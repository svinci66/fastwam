"""Run RoboTwin's evaluator with small, validated FastWAM compatibility hooks.

RoboTwin's upstream evaluator hard-codes expert trajectory filtering.  That is
useful for its official benchmark, but it can silently advance the environment
seed when motion planning is nondeterministic.  Controlled corruption studies
need a stricter invariant: every policy variant must start from the same scene.

This entrypoint leaves the upstream checkout untouched.  It validates and
patches three narrowly scoped source fragments before executing the evaluator:

* make expert filtering configurable through ``expert_check``;
* allow a fixed instruction when expert filtering is disabled;
* expose the environment seed in logs and process state for provenance.

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
        "            instruction = np.random.choice(results[0][instruction_type])\n",
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
