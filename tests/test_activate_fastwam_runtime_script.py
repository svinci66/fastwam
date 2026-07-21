import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "activate_fastwam_runtime.sh"


def test_activate_fastwam_runtime_script_has_valid_bash_syntax():
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


def test_activate_fastwam_runtime_script_requires_source():
    result = subprocess.run(
        ["bash", str(SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "must be sourced" in result.stderr


def test_activate_fastwam_runtime_script_can_source_with_overrides(tmp_path):
    environment = tmp_path / "envs" / "fastwam"
    (environment / "bin").mkdir(parents=True)
    python = environment / "bin" / "python"
    python.touch(mode=0o755)
    (environment / "bin" / "activate").touch()

    result = subprocess.run(
        [
            "bash",
            "--noprofile",
            "--norc",
            "-c",
            (
                f"export FASTWAM_ROOT={tmp_path}; "
                f"export FASTWAM_ENV={environment}; "
                "export FASTWAM_RENDER_REQUEST=osmesa; "
                f"source {SCRIPT}; "
                'test "$MUJOCO_GL" = osmesa; '
                'test "$PYOPENGL_PLATFORM" = osmesa; '
                'test "$TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD" = 1; '
                'test "$(command -v python)" = "$FASTWAM_ENV/bin/python"'
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "FastWAM persistent environment activated" in result.stdout
