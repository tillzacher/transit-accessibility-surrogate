"""The behavioral instrument is independent of training and geographic infrastructure."""

import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_generation_and_scoring_need_only_array_libraries():
    code = """
import sys
from modules.generated_metamorphic_suite import build_evaluation_set
from modules.generated_suite_scoring import score_evaluation_set
assert not any(x in sys.modules for x in ('torch', 'geopandas', 'sqlalchemy', 'modules.training_pipeline'))
"""
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


def test_inference_does_not_import_training():
    code = """
import sys
from modules.surrogate_inference import SurrogateInferenceAdapter
assert 'modules.training_pipeline' not in sys.modules
assert 'geopandas' not in sys.modules
assert 'sqlalchemy' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


def test_author_code_has_no_logging_dependency():
    for path in (ROOT / "modules").glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(x.name != "logging" for x in node.names), path
            if isinstance(node, ast.ImportFrom):
                assert node.module != "logging", path
