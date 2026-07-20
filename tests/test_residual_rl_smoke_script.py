import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "run_libero_residual_rl_smoke.sh"


def test_residual_rl_smoke_script_has_valid_bash_syntax():
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


def test_residual_rl_smoke_script_help_does_not_require_runtime_assets():
    result = subprocess.run(
        ["bash", str(SCRIPT), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "single-GPU LIBERO residual-RL smoke" in result.stdout
    assert "--checkpoint" in result.stdout
    assert "--resume" in result.stdout
