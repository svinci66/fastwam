import json

from experiments.robotwin.analyze_place_can_physics_audit import analyze


def _write_audit(path, *, seed, success, basket=False, ground=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    partners = []
    if basket:
        partners.append("110_basket")
    if ground:
        partners.append("ground")
    records = [
        {
            "event": "start",
            "episode_id": 0,
            "seed": seed,
            "initial": {
                "position": [0.1, 0.2, 0.7],
                "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            },
        },
        {
            "event": "action",
            "action_index": 0,
            "post": {
                "position": [0.1, 0.2, 0.85],
                "contacts": {
                    "partners": partners,
                    "gripper_contact": True,
                },
            },
        },
        {
            "event": "finish",
            "success": success,
            "num_actions": 1,
            "final": {"position": [0.1, 0.2, 0.85]},
            "max_displacement_from_initial_m": 0.15,
            "max_linear_speed_mps": 0.2,
            "max_angular_speed_rps": 0.3,
            "first_anomaly_action": None,
            "anomaly_counts": {},
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in records))


def test_analyze_proves_pairing_and_classifies_imagination_win(tmp_path):
    seed = 4800108
    instruction = "Place the can in the basket."
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "place_can_basket": {
                    "seeds": [seed],
                    "instructions": [instruction],
                }
            }
        )
    )

    rows = []
    outcomes = {"baseline": False, "no_imagination": False, "imagination": True}
    audit_root = tmp_path / "physics_audit"
    for variant, success in outcomes.items():
        rows.append(
            {
                "task": "place_can_basket",
                "variant": variant,
                "episode_records": [
                    {
                        "episode_id": 0,
                        "seed": seed,
                        "instruction": instruction,
                        "success": success,
                    }
                ],
            }
        )
        _write_audit(
            audit_root
            / variant
            / "place_can_basket"
            / f"episode0_seed{seed}.jsonl",
            seed=seed,
            success=success,
            basket=success,
            ground=not success,
        )

    online_summary_path = tmp_path / "summary.json"
    online_summary_path.write_text(
        json.dumps(
            {
                "rows": rows,
                "initial_state_audit": {
                    "place_can_basket": {"exact_match": True}
                },
                "protocol_pairing_audit": {
                    "place_can_basket": {
                        "exact_seed_and_instruction_match": True
                    }
                },
            }
        )
    )

    result = analyze(
        audit_root=audit_root,
        online_summary_path=online_summary_path,
        manifest_path=manifest_path,
    )

    assert result["pairing_audit"] == {
        "online_initial_observation_exact_match": True,
        "online_seed_and_instruction_exact_match": True,
        "actor_initial_pose_exact_match": True,
    }
    assert result["flip_counts"]["imagination_win"] == 1
    assert result["aggregate"]["imagination"]["successes"] == 1
    assert result["aggregate"]["no_imagination"]["ground_contact_episodes"] == 1
