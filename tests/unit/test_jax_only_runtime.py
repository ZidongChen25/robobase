import ast
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import tomllib


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "robobase"
FORBIDDEN_RUNTIME_PACKAGES = {"torch", "torchvision", "timm", "jax_resnet"}


def _import_root(node: ast.Import | ast.ImportFrom) -> str:
    if isinstance(node, ast.Import):
        return node.names[0].name.split(".", maxsplit=1)[0]
    return (node.module or "").split(".", maxsplit=1)[0]


def test_first_party_package_has_no_torch_ecosystem_imports():
    offenders = []
    for source_path in PACKAGE_ROOT.rglob("*.py"):
        tree = ast.parse(source_path.read_text(encoding="utf-8"), source_path)
        for node in ast.walk(tree):
            if (
                isinstance(node, (ast.Import, ast.ImportFrom))
                and _import_root(node) in FORBIDDEN_RUNTIME_PACKAGES
            ):
                offenders.append(f"{source_path.relative_to(REPO_ROOT)}:{node.lineno}")
    assert offenders == []


def test_package_dependencies_do_not_install_torch_ecosystem():
    metadata = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependency_sets = [metadata["project"].get("dependencies", [])]
    dependency_sets.extend(
        metadata["project"].get("optional-dependencies", {}).values()
    )
    dependency_sets.extend(metadata.get("dependency-groups", {}).values())
    offenders = [
        dependency
        for dependencies in dependency_sets
        for dependency in dependencies
        if dependency.lower()
        .split("[", maxsplit=1)[0]
        .replace("-", "_")
        .split(" ", maxsplit=1)[0]
        in FORBIDDEN_RUNTIME_PACKAGES
    ]
    assert offenders == []


def test_jax_runtime_imports_when_torch_is_blocked():
    source = f"""
import importlib.abc
import pathlib
import sys

class BlockTorchEcosystem(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        blocked = ('torch', 'torchvision', 'timm', 'jax_resnet')
        if fullname in blocked or fullname.startswith(tuple(name + '.' for name in blocked)):
            raise ImportError(f'blocked by JAX-only runtime test: {{fullname}}')
        return None

sys.meta_path.insert(0, BlockTorchEcosystem())
sys.path.insert(0, {str(REPO_ROOT)!r})
import robobase.language
import robobase.method
import robobase.models.backbone
import robobase.replay_buffer.iterator
import robobase.utils
import train
import robobase.factory
import robobase.workspace

import jax.numpy as jnp
from robobase.models.encoder import JaxResNetEncoder

encoder = JaxResNetEncoder(
    input_shape=(1, 3, 32, 32),
    model='resnet18',
    jit=False,
    pretrained=False,
    resize_to_224=False,
)
features = encoder.encode(jnp.zeros((1, 1, 3, 32, 32), dtype=jnp.float32))
assert features.shape == (1, 1, 512), features.shape

try:
    JaxResNetEncoder(
        input_shape=(1, 3, 32, 32),
        model='resnet18',
        jit=False,
        pretrained=True,
        resize_to_224=False,
    )
except FileNotFoundError:
    pass
else:
    raise AssertionError('missing pretrained NPZ must raise FileNotFoundError')

assert not any(
    name in {{'torch', 'torchvision', 'timm', 'jax_resnet'}}
    or name.startswith(('torch.', 'torchvision.', 'timm.', 'jax_resnet.'))
    for name in sys.modules
)
"""
    with tempfile.TemporaryDirectory() as home:
        env = dict(os.environ)
        env.update(
            {
                "HOME": home,
                "JAX_PLATFORMS": "cpu",
                "ROBOBASE_DISABLE_PRETRAINED_DOWNLOAD": "1",
            }
        )
        env.pop("ROBOBASE_RESNET18_JAX_NPZ", None)
        result = subprocess.run(
            [sys.executable, "-c", source],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
    assert result.returncode == 0, result.stderr
