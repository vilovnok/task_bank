import os
import subprocess
import sys
import time


def start_script(cmd: list[str], label: str) -> subprocess.Popen:
    proc = subprocess.Popen(cmd)
    print(sys.executable)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Запущен {label} (PID {proc.pid})")
    print("  CMD:", " ".join(cmd))
    return proc


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    manager_path = os.path.join(base_dir, "manager.py")

    scripts = [
        ([sys.executable, manager_path, "--stand", "sberdev"], "manager stand=sberdev"),
        ([sys.executable, manager_path, "--stand", "sigma"],   "manager stand=sigma"),
        ([sys.executable, manager_path, "--stand", "cloud"],   "manager stand=cloud"),
    ]

    for cmd, label in scripts:
        start_script(cmd, label)


if __name__ == "__main__":
    main()