# This file is part of pelagos_py.
#
# Copyright 2025-2026 National Oceanography Centre and The Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The dashboard's one pipeline subprocess: start, stop, stream its output and talk to it."""

from __future__ import annotations

import atexit
import codecs
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from pelagos_py.utils.demo_data import WORKSPACE_DIR

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def _clean_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


RUN_BOOTSTRAP = Path(__file__).resolve().parent / "run_bootstrap.py"
# one folder per dashboard process, so two dashboards can't delete each other's figures
FIG_DIR = Path(tempfile.mkdtemp(prefix="pelagos-py-figures-"))
atexit.register(shutil.rmtree, FIG_DIR, ignore_errors=True)
_FIGDATA_CACHE: dict[str, dict] = {}


class _Run:
    # The one pipeline subprocess. Lines are append-only so a reconnecting client can replay them.

    def __init__(self):
        self.proc: subprocess.Popen | None = None
        self.lines: list[str] = []
        self.live: str | None = None  # latest progress-bar redraw
        self.live_after = 0
        self.finished = False
        self.returncode: int | None = None
        self.reports: set[str] = set()  # only announced PDFs are served
        self.paused = False
        # every report/log file this dashboard's runs wrote; the only ones Outputs lists
        self.outputs: set[Path] = set()
        self._lock = threading.Lock()
        # processing clock from __PELAGOS_TIME__, so RAM samples share the runtime readout's x-axis
        self._active = 0.0
        self._since: float | None = None
        self._columns = threading.Condition()
        self._columns_done: set[int] = set()
        self._columns_id = 0

    def _sample_mem(self, proc):
        # polled from here so it can't interleave with the run's console output
        try:
            import psutil

            child = psutil.Process(proc.pid)
        except Exception:
            return
        while proc.poll() is None:
            try:
                rss = child.memory_info().rss / 1e6
            except Exception:
                return
            with self._lock:
                if self._since is not None and proc is self.proc:
                    x = self._active + time.time() - self._since
                    self.lines.append(f"__PELAGOS_SAMPLE__ {x:.1f}\t{rss:.1f}")
            time.sleep(0.5)

    def is_running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, config_path: Path):
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        # a pipe isn't a terminal, so force tqdm bars on (their colour codes are stripped below)
        env["FORCE_COLOR"] = "1"
        env.pop("NO_COLOR", None)
        env["COLUMNS"] = "110"
        # locking can fail (Errno -101) right after another process (the variable check) closed the file
        env["HDF5_USE_FILE_LOCKING"] = "FALSE"
        # check and start under one lock: the app serves requests from a thread pool
        with self._lock:
            if self.is_running():
                raise RuntimeError("A pipeline is already running.")
            for pattern in ("*.png", "*.json", "*.f32", "*_full.npz", "cols_*.bin"):
                for old in FIG_DIR.glob(pattern):
                    old.unlink(missing_ok=True)
            _FIGDATA_CACHE.clear()
            self.lines = []
            self.live = None
            self.live_after = 0
            self.finished = False
            self.returncode = None
            self.reports = set()
            self.paused = False
            self._active, self._since = 0.0, None
            self.proc = subprocess.Popen(
                [sys.executable, str(RUN_BOOTSTRAP), str(config_path), str(FIG_DIR)],
                cwd=str(WORKSPACE_DIR),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                bufsize=-1,  # binary BufferedReader: keeps \r and has read1, so bar ticks stream promptly
            )
        threading.Thread(target=self._pump, args=(self.proc,), daemon=True).start()
        threading.Thread(
            target=self._sample_mem, args=(self.proc,), daemon=True
        ).start()

    def _commit(self, proc, text: str):
        if text.startswith("__PELAGOS_DATA__ "):  # a reply to columns(), not a log line
            with self._columns:
                self._columns_done.add(int(text.split(" ", 1)[1]))
                self._columns.notify_all()
            return
        with self._lock:
            # a stopped run's pump can still be draining after the next run started
            if proc is not self.proc:
                return
            # for Outputs only, not a log line
            if text.startswith("__PELAGOS_OUTPUT__ "):
                self.outputs.add(Path(text.split(" ", 1)[1].strip()).resolve())
                return
            if text.startswith("__PELAGOS_REPORT__ "):
                report = text.split(" ", 1)[1].split("\t")[0].strip()
                self.reports.add(report)
                self.outputs.add(Path(report).resolve())
            if text.startswith("__PELAGOS_PAUSE__ "):
                self.paused = True
            self.lines.append(text)
            self.live = None
            self.live_after = len(self.lines)
            if text.startswith("__PELAGOS_TIME__ "):
                try:
                    active, paused, _ = text.split(" ", 1)[1].split("\t")
                    self._active = float(active)
                    self._since = None if paused.strip() == "1" else time.time()
                except ValueError:
                    pass

    def _progress(self, proc, text: str):
        with self._lock:
            if proc is not self.proc:
                return
            self.live = text
            self.live_after = len(self.lines)

    def _pump(self, proc):
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        stream = proc.stdout
        seg = ""  # chars since the last \r or \n
        pending_cr = False
        while True:
            chunk = stream.read1(4096)  # type: ignore[union-attr]
            if not chunk:
                break
            for ch in decoder.decode(chunk):
                if ch == "\n":
                    # \r\n is a line ending, not a redraw
                    pending_cr = False
                    self._commit(proc, _clean_ansi(seg))
                    seg = ""
                    continue
                if pending_cr:
                    # a lone \r is a progress-bar redraw
                    cleaned = _clean_ansi(seg)
                    if cleaned:
                        self._progress(proc, cleaned)
                    seg = ""
                    pending_cr = False
                if ch == "\r":
                    pending_cr = True
                else:
                    seg += ch
        if seg:
            if pending_cr:
                cleaned = _clean_ansi(seg)
                if cleaned:
                    self._progress(proc, cleaned)
            else:
                self._commit(proc, _clean_ansi(seg))
        returncode = proc.wait()
        proc.stdin.close()
        with self._lock:
            if proc is self.proc:
                self.returncode = returncode
                self.finished = True
        with self._columns:
            self._columns.notify_all()  # a columns() request can't be answered any more

    def stop(self):
        if self.is_running() and sys.platform == "win32":
            self.proc.terminate()  # Windows Popen can't send SIGINT (raises ValueError)
            self.proc.wait()
        elif self.is_running():
            # SIGINT first so the child runs its cleanup and doesn't leak pool semaphores
            self.proc.send_signal(signal.SIGINT)
            try:
                self.proc.wait(timeout=1.5)
            except subprocess.TimeoutExpired:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait()

    def send(self, command: str):
        # a control line for run_bootstrap.py's pause loop; ignored once the run has exited
        if self.proc is not None and self.proc.stdin is not None and self.is_running():
            try:
                self.proc.stdin.write((command + "\n").encode())
                self.proc.stdin.flush()
            except (BrokenPipeError, ValueError, OSError):
                pass

    def resume(self, command: str) -> bool:
        # continue, skip or rerun; refused unless paused, so a double click can't answer the next pause
        with self._lock:
            if not self.paused or not self.is_running():
                return False
            self.paused = False
        self.send(command)
        return True

    def columns(self, names: list[str], timeout: float = 60.0) -> bytes | None:
        # packed columns from the paused runner, or None if it isn't paused or didn't answer
        if not self.paused:
            return None
        with self._columns:
            self._columns_id += 1
            request_id = self._columns_id
        self.send("data " + json.dumps({"id": request_id, "names": names}))
        with self._columns:
            answered = self._columns.wait_for(
                lambda: request_id in self._columns_done or not self.is_running(),
                timeout,
            )
            self._columns_done.discard(request_id)
        path = FIG_DIR / f"cols_{request_id}.bin"
        if not answered or not path.is_file():
            return None
        blob = path.read_bytes()
        path.unlink(missing_ok=True)
        return blob


_run = _Run()
atexit.register(_run.stop)
