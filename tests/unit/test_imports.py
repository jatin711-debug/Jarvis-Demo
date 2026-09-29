"""Every module imports on a slim install; heavy dependencies load lazily.

CI installs only the packages listed in ALLOWED_TOP_LEVEL. Anything heavier
(torch, kokoro, playwright, pyaudio, mem0, ...) must be imported inside the
function that needs it, so a voice-less or hardware-less environment can still
import and test the package.
"""

import ast
import importlib
import pkgutil
import sys
from pathlib import Path

import pytest

import jarvis

PACKAGE_ROOT = Path(jarvis.__file__).parent
ALLOWED_TOP_LEVEL = {"httpx", "pydantic", "yaml", "psutil", "numpy", "soundfile"}

MODULES = sorted(m.name for m in pkgutil.walk_packages(jarvis.__path__, "jarvis."))


def _module_level_imports(tree: ast.Module) -> set[str]:
    """Root names imported at module scope, ignoring `if TYPE_CHECKING:` blocks."""
    roots: set[str] = set()

    def visit(body):
        for node in body:
            if isinstance(node, ast.Import):
                roots.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level == 0 and node.module:
                    roots.add(node.module.split(".")[0])
            elif isinstance(node, ast.If):
                test = ast.unparse(node.test)
                if "TYPE_CHECKING" not in test:
                    visit(node.body)
                    visit(node.orelse)
            elif isinstance(node, ast.Try):
                visit(node.body)
                for handler in node.handlers:
                    visit(handler.body)
                visit(node.orelse)
                visit(node.finalbody)

    visit(tree.body)
    return roots


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    if name.endswith(".__main__"):
        pytest.skip("entry points run on import")
    importlib.import_module(name)


@pytest.mark.parametrize("name", MODULES)
def test_only_light_dependencies_at_module_level(name):
    path = PACKAGE_ROOT.parent / (name.replace(".", "/") + ".py")
    if not path.exists():  # package __init__
        path = PACKAGE_ROOT.parent / name.replace(".", "/") / "__init__.py"
    roots = _module_level_imports(ast.parse(path.read_text(encoding="utf-8")))
    third_party = {r for r in roots if r not in sys.stdlib_module_names and r != "jarvis"}
    heavy = third_party - ALLOWED_TOP_LEVEL
    assert not heavy, f"{name} imports {sorted(heavy)} at module level; import lazily inside functions"
