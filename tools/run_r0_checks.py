"""Run and retain R0 checks without private data, weights or absolute paths."""
from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    evidence = root / 'reports' / 'evidence'
    evidence.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', PYTHONIOENCODING='utf-8')
    checks = [
        ('pytest.txt', ['-m', 'pytest', '-q', 'tests']),
        ('environment.txt', ['experiments/exp0/env_check.py']),
        ('compileall.txt', ['-m', 'compileall', '-q', 'experiments', 'tests', 'tools']),
    ]
    failed = False
    for filename, args in checks:
        result = subprocess.run([sys.executable, *args], cwd=root, env=env,
                                text=True, encoding='utf-8', capture_output=True)
        output = result.stdout + result.stderr
        # Tracebacks may include machine paths; preserve evidence without publishing them.
        for private in (str(root), str(Path.home())):
            output = output.replace(private, '<local>')
        record = (f'UTC: {datetime.now(timezone.utc).isoformat()}\n'
                  f'cwd: repository root\nPYTEST_DISABLE_PLUGIN_AUTOLOAD=1\n'
                  f'$ python {" ".join(args)}\n{output}'
                  f'\nExit code: {result.returncode}\n')
        (evidence / filename).write_text(record, encoding='utf-8')
        print(record, end='')
        failed |= result.returncode != 0
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
