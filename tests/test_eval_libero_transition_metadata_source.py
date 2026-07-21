import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_SCRIPT = PROJECT_ROOT / "experiments" / "libero" / "eval_libero_single.py"


def _dict_values_for_key(tree: ast.AST, key_name: str) -> list[ast.expr]:
    values = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values):
            if isinstance(key, ast.Constant) and key.value == key_name:
                values.append(value)
    return values


def test_noise_std_is_carried_by_each_clip_and_read_from_that_clip():
    tree = ast.parse(EVAL_SCRIPT.read_text(encoding="utf-8"))
    expressions = {
        ast.unparse(value) for value in _dict_values_for_key(tree, "action_noise_std")
    }

    assert "action_noise_std if action_mode == 'noise' else 0.0" in expressions
    assert "float(clip.get('action_noise_std', 0.0))" in expressions
