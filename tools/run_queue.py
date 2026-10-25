"""Run commands one after another in a process that keeps going if the Claude session ends.
Output (minus warning lines) is appended to outputs/live.log.
Usage (detached, from PowerShell):  Start-Process -WindowStyle Hidden uv -ArgumentList 'run','python','tools/run_queue.py','queue.txt'
queue.txt: one command per line (blank lines and lines starting with # are skipped).
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKIP = ("warn", "symlink", "developer mode", "deprecat", "triton", "redirects", "loading weights", "float32")


def main():
    cmds = [l.strip() for l in Path(sys.argv[1]).read_text().splitlines() if l.strip() and not l.startswith("#")]
    with open(ROOT / "outputs" / "live.log", "a", encoding="utf-8") as log:
        log.write(f"{time.strftime('%H:%M:%S')}  [queue] starting {len(cmds)} commands from {sys.argv[1]}\n")
        log.flush()
        for i, c in enumerate(cmds, 1):
            log.write(f"{time.strftime('%H:%M:%S')}  [queue] {i}/{len(cmds)}: {c}\n")
            log.flush()
            p = subprocess.Popen(c, shell=True, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                 encoding="utf-8", errors="replace")
            for line in p.stdout:
                if not any(k in line.lower() for k in SKIP):
                    log.write(line)
                    log.flush()
            p.wait()
            log.write(f"{time.strftime('%H:%M:%S')}  [queue] {i}/{len(cmds)} finished with exit code {p.returncode}\n")
            log.flush()
        log.write(f"{time.strftime('%H:%M:%S')}  [queue] all done\n")


if __name__ == "__main__":
    main()
