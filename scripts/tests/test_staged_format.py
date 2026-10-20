"""Verify the adopted two-stage hook with real partially staged commits."""

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from install_hooks import install  # noqa: E402


class StagedFormatTests(unittest.TestCase):
    def test_partial_staging_and_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def git(*args, check=True):
                return subprocess.run(
                    ["git", "-C", str(root), *args],
                    capture_output=True,
                    text=True,
                    check=check,
                )

            git("init", "-q")
            git("config", "user.name", "Fixture")
            git("config", "user.email", "fixture@example.invalid")
            ruff = shutil.which("ruff")
            self.assertIsNotNone(ruff)
            formatter = ROOT / "node_modules/.bin/git-format-staged"
            config = {
                "repos": [
                    {
                        "repo": "local",
                        "hooks": [
                            {
                                "id": "format",
                                "name": "format",
                                "language": "system",
                                "entry": f'{formatter} --formatter "{ruff} format --stdin-filename {{}} -"',
                                "types": ["python"],
                                "stages": ["manual"],
                            },
                            {
                                "id": "lint",
                                "name": "lint",
                                "language": "system",
                                "entry": f"{ruff} check --select F821",
                                "types": ["python"],
                                "stages": ["pre-commit"],
                            },
                        ],
                    }
                ]
            }
            (root / ".pre-commit-config.yaml").write_text(json.dumps(config))
            code = root / "sample.py"
            code.write_text("value = 0\n")
            git("add", ".")
            git("commit", "-qm", "baseline")
            install(root)
            code.write_text("value=1\n")
            git("add", "sample.py")
            code.write_text("value=1\n\n# unstaged addition\n")
            result = git("commit", "-qm", "format", check=False)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(git("show", "HEAD:sample.py").stdout, "value = 1\n")
            self.assertIn("# unstaged addition", code.read_text())
            code.write_text("value = undefined_name\n")
            git("add", "sample.py")
            code.write_text("value = 2\n")
            result = git("commit", "-qm", "must fail", check=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(code.read_text(), "value = 2\n")
            self.assertIn("undefined_name", git("show", ":sample.py").stdout)
            code.write_text("value = (\n")
            git("add", "sample.py")
            result = git("commit", "-qm", "syntax failure", check=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(code.read_text(), "value = (\n")
