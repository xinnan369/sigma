"""Record one actual command, output, time and exit code with private paths removed."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import subprocess
import time


def sanitize(text: str) -> str:
    for path, replacement in [(Path.cwd(), '<BRIDGE>'), (Path.cwd().parent, '<WORKSPACE>')]:
        if path.drive:
            wsl_path = '/mnt/' + path.drive[0].lower() + path.as_posix()[2:]
            text = text.replace(wsl_path, replacement)
    for path, replacement in [(str(Path.cwd()), '<BRIDGE>'),
                              (str(Path.home()), '<HOME>')]:
        text = text.replace(path, replacement).replace(path.replace('\\', '/'), replacement)
    text = re.sub(r'/mnt/[a-z]/Users/[^/\s]+', '<WINDOWS_HOME>', text)
    text = re.sub(r'/home/[^/\s]+', '<LINUX_HOME>', text)
    text = re.sub(r'("editable_project_location":\s*)"[^"]*"',
                  r'\1"<LOCAL_EDITABLE_PROJECT>"', text)
    text = re.sub(r'\x1b\[[0-9;]*m', '', text.replace('\x00', ''))
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    return '\n'.join(line.rstrip() for line in text.split('\n'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('A command is required')
    start = time.perf_counter()
    result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', errors='replace',
                            env=dict(os.environ, PYTHONIOENCODING='utf-8', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'))
    output = (f'UTC: {datetime.now(timezone.utc).isoformat()}\n'
              f'Command argv: {command!r}\n'
              f'PYTEST_DISABLE_PLUGIN_AUTOLOAD=1\n'
              f'{result.stdout}{result.stderr}\n'
              f'Elapsed seconds: {time.perf_counter() - start:.3f}\n'
              f'Exit code: {result.returncode}\n')
    output = sanitize(output)
    args.output.parent.mkdir(exist_ok=True, parents=True)
    args.output.write_text(output, encoding='utf-8', newline='\n')
    print(output)
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
