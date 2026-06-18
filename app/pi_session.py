"""
PiSession — long-running pi --mode rpc process manager for chat-brain.

One persistent `pi --mode rpc` subprocess per chat_id. The process stays
alive across messages, accumulating context, until the chat explicitly
wipes it (/new, /start, /wipe, or a configured wipe command). Sessions
are named ``chat_<chat_id>`` and persisted by pi under
``~/.pi/agent/sessions/`` so they survive process restarts.

This is the canonical "thread doesn't close until I say so" behavior
from the Hetzner build. No message ever lands in a fresh process by
accident.
"""

from __future__ import annotations

import json
import logging
import os
import shlex
import shutil
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

log = logging.getLogger("chat_brain.pi_session")

PI_BIN = os.environ.get("PI_BIN", "pi")
SESSIONS_DIR = Path(os.environ.get("PI_SESSIONS_DIR",
                                   str(Path.home() / ".pi" / "agent" / "sessions")))
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

# Commands that wipe a session back to a fresh pi. The user's Telegram
# message must equal (after stripping @bot mentions) one of these.
WIPE_COMMANDS = {"/new", "/start", "/wipe", "/reset", "/clear"}


@dataclass
class _Proc:
    chat_id: str
    p: subprocess.Popen
    lock: threading.Lock = field(default_factory=threading.Lock)
    last_used: float = field(default_factory=time.time)
    pending: dict[str, dict] = field(default_factory=dict)  # id -> event buffer


class PiSessionManager:
    """Thread-safe per-chat pi RPC process manager.

    A single global instance lives in chat-brain's process. Methods are
    safe to call from FastAPI's threadpool.
    """

    def __init__(self, model: str = "minimax/MiniMax-M3",
                 idle_timeout_s: int = 60 * 60) -> None:
        self._procs: dict[str, _Proc] = {}
        self._mu = threading.Lock()
        self._model = model
        self._idle_timeout_s = idle_timeout_s
        if not shutil.which(PI_BIN):
            log.warning("pi binary %r not on PATH; manager will fail on first use", PI_BIN)

    # --- lifecycle --------------------------------------------------------

    def _session_name(self, chat_id: str) -> str:
        # pi session names are filename-friendly; sanitize chat_id
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(chat_id))
        return f"chat_{safe}"

    def _spawn(self, chat_id: str) -> _Proc:
        name = self._session_name(chat_id)
        cmd = [
            PI_BIN,
            "--mode", "rpc",
            "--provider", "minimax",
            "--model", self._model,
            "--name", name,
            "--session-dir", str(SESSIONS_DIR),
        ]
        log.info("spawning pi rpc for chat_id=%s cmd=%s", chat_id, shlex.join(cmd))
        env = os.environ.copy()
        # Make sure the user-level pi config is honored
        env.setdefault("HOME", str(Path.home()))
        p = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )
        proc = _Proc(chat_id=chat_id, p=p)
        with self._mu:
            self._procs[chat_id] = proc
        return proc

    def _kill(self, chat_id: str) -> None:
        with self._mu:
            proc = self._procs.pop(chat_id, None)
        if not proc:
            return
        try:
            proc.p.send_signal(signal.SIGTERM)
            try:
                proc.p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.p.kill()
                proc.p.wait(timeout=5)
        except Exception as exc:  # noqa: BLE001
            log.warning("error killing pi for chat_id=%s: %s", chat_id, exc)

    def _get_or_spawn(self, chat_id: str) -> _Proc:
        with self._mu:
            proc = self._procs.get(chat_id)
        if proc and proc.p.poll() is None:
            proc.last_used = time.time()
            return proc
        if proc:
            # process died; drop it and respawn
            self._kill(chat_id)
        return self._spawn(chat_id)

    # --- prompt API -------------------------------------------------------

    def is_wipe_command(self, text: str) -> bool:
        # Strip @bot_username suffix if present
        t = text.strip().split()[0].lower() if text.strip() else ""
        return t in WIPE_COMMANDS

    def wipe(self, chat_id: str) -> str:
        """Kill the live process and delete the persisted session file.

        Returns a one-line confirmation the caller can show in Telegram.
        """
        self._kill(chat_id)
        # Pi stores sessions as JSONL under ~/.pi/agent/sessions/, often
        # in a `--user--/` subdir keyed on who owned the cwd. The
        # session name we set via --name lives in the file's first
        # header line, NOT in the filename. Scan all JSONL files for
        # a header whose `name` matches and delete those.
        target_name = self._session_name(chat_id)
        removed = 0
        for f in SESSIONS_DIR.rglob("*.jsonl"):
            try:
                with f.open("r", encoding="utf-8") as fh:
                    first = fh.readline().strip()
                if not first:
                    continue
                meta = json.loads(first)
            except (OSError, json.JSONDecodeError):
                continue
            # pi's session header keys vary across versions; check the
            # ones that name the session.
            name = (meta.get("name") or meta.get("sessionName")
                    or meta.get("displayName"))
            if name == target_name:
                f.unlink(missing_ok=True)
                removed += 1
        log.info("wiped chat_id=%s (removed %d session file(s))", chat_id, removed)
        return f"🧹 session wiped ({removed} file(s) removed). Next message starts fresh."

    def send(self, chat_id: str, message: str,
             timeout_s: int = 600) -> dict[str, Any]:
        """Send a single message to the chat's long-running pi session.

        Returns the full agent response envelope (text + tool calls).
        Streams internally; the user sees one final answer per prompt,
        not a delta stream — Telegram's edit-message cadence is the
        wrong place for token-by-token updates.
        """
        proc = self._get_or_spawn(chat_id)
        req_id = f"req-{int(time.time() * 1000)}"
        with proc.lock:
            proc.p.stdin.write(json.dumps({"id": req_id, "type": "prompt",
                                           "message": message}) + "\n")
            proc.p.stdin.flush()
            return self._collect(proc, req_id, timeout_s)

    def _collect(self, proc: _Proc, req_id: str,
                 timeout_s: int) -> dict[str, Any]:
        """Read JSONL events from pi's stdout until the prompt resolves.

        The `response` frame just means "I accepted your prompt"; the
        agent keeps streaming after that. We must wait for `agent_end`
        to know the assistant is truly done (could be many turns if the
        agent uses tools). All text/tool deltas across the whole run
        are accumulated into one final answer.

        Returns a dict with keys: text, tool_calls, duration_s, ok, error.
        """
        deadline = time.time() + timeout_s
        text_chunks: list[str] = []
        tool_calls: list[dict] = []
        accepted = False
        agent_end_seen = False
        err: str | None = None
        start = time.time()
        while time.time() < deadline:
            if proc.p.poll() is not None:
                err = f"pi process exited (code={proc.p.returncode}) mid-prompt"
                break
            line = proc.p.stdout.readline()
            if not line:
                time.sleep(0.05)
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                log.warning("pi emitted non-json line: %r", line[:200])
                continue
            et = ev.get("type")
            # `response` = command accepted/queued. NOT done.
            if et == "response" and ev.get("id") == req_id:
                if not ev.get("success"):
                    err = ev.get("error", "pi rejected the prompt")
                    break
                accepted = True
                continue
            if err is not None and et != "agent_end":
                # Already failed; just drain to agent_end to stay framed.
                if et == "agent_end":
                    break
                continue
            if et == "message_update":
                ame = ev.get("assistantMessageEvent", {}) or {}
                if ame.get("type") == "text_delta":
                    text_chunks.append(ame.get("delta", ""))
                if ame.get("type") == "toolcall_end":
                    tc = ame.get("partial", {}).get("toolCall") or {}
                    tool_calls.append({"name": tc.get("name"),
                                       "args": tc.get("arguments")})
            if et == "agent_end":
                agent_end_seen = True
                break
        duration = round(time.time() - start, 2)
        return {
            "text": "".join(text_chunks),
            "tool_calls": tool_calls,
            "duration_s": duration,
            "ok": err is None and accepted and agent_end_seen,
            "error": err,
            "agent_end_seen": agent_end_seen,
        }

    # --- housekeeping ----------------------------------------------------

    def shutdown(self) -> None:
        for cid in list(self._procs.keys()):
            self._kill(cid)


_singleton: PiSessionManager | None = None


def get_manager() -> PiSessionManager:
    global _singleton
    if _singleton is None:
        _singleton = PiSessionManager()
    return _singleton
