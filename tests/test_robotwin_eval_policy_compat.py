from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.robotwin.eval_policy_compat import patch_eval_policy_source


def test_patch_eval_policy_source_adds_pairing_controls():
    source = '''\
def eval_policy():
    expert_check = True
    suc_test_seed_list = []
    while active:
        args["render_freq"] = 0
        if expert_check:
            try:
                run_expert()
            except UnStableError as e:
                continue
            except Exception as e:
                continue
        if accepted:
            suc_test_seed_list.append(now_seed)
        else:
            now_seed += 1
            args["render_freq"] = render_freq
            continue

        args["render_freq"] = render_freq
        episode_info_list = [episode_info["info"]]
        results = generate_episode_descriptions(args["task_name"], episode_info_list, test_num)
        instruction = np.random.choice(results[0][instruction_type])
'''

    patched = patch_eval_policy_source(source)

    assert 'usr_args.get("expert_check", True)' in patched
    assert "FASTWAM_ACCEPTED_ENV_SEED" in patched
    assert "fixed_instruction is required" in patched
    assert "environment_seed_manifest_path" in patched
    assert "strict_environment_seeds[succ_seed]" in patched
    assert "np.random.default_rng(int(now_seed))" in patched
    assert "FASTWAM_EVAL_INSTRUCTION" in patched


def test_patch_eval_policy_source_fails_closed_on_upstream_change():
    try:
        patch_eval_policy_source("expert_check = True\n")
    except RuntimeError as exc:
        assert "expert-check" in str(exc)
    else:
        raise AssertionError("expected incompatible upstream source to be rejected")
