import cv2
import numpy as np

from experiments.robotwin.export_expert_imagination_transitions import (
    completed_episode_transition_count,
    decode_jpeg,
    expert_action_chunk,
    resolve_task_data_dir,
)


def test_decode_jpeg_returns_rgb() -> None:
    rgb = np.zeros((8, 9, 3), dtype=np.uint8)
    rgb[..., 0] = 255
    ok, encoded = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    assert ok
    decoded = decode_jpeg(encoded.tobytes())
    assert decoded.shape == rgb.shape
    assert decoded.dtype == np.uint8
    assert decoded[0, 0].tolist() == [255, 0, 0]


def test_expert_action_chunk_uses_future_qpos_and_keeps_partial_terminal() -> None:
    states = np.arange(6 * 14, dtype=np.float32).reshape(6, 14)
    full, full_k, full_end = expert_action_chunk(states, 0, 3)
    assert full_k == 3
    assert full_end == 3
    assert np.array_equal(full, states[1:4])

    partial, partial_k, partial_end = expert_action_chunk(states, 3, 3)
    assert partial_k == 2
    assert partial_end == 5
    assert np.array_equal(partial, states[4:6])


def test_resolve_nested_task_data_dir(tmp_path) -> None:
    data_dir = tmp_path / "task" / "aloha-agilex_clean_50" / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "episode0.hdf5").touch()
    assert resolve_task_data_dir(tmp_path, "task") == data_dir


def test_completed_episode_requires_contiguous_terminal_boundary(tmp_path) -> None:
    episode_dir = tmp_path / "task" / "expert" / "episode_0000"
    for index, terminal in enumerate((False, True)):
        record_dir = episode_dir / f"replan_{index:04d}"
        record_dir.mkdir(parents=True)
        (record_dir / "metadata.json").write_text(
            '{"replan_idx": %d, "terminated": %s, "truncated": false}'
            % (index, str(terminal).lower()),
            encoding="utf-8",
        )
    assert completed_episode_transition_count(tmp_path, "task", 0) == 2
