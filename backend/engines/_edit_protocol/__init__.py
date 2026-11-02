"""Stdlib wire loop shared by the Audio Edit engine sidecars.

Framing is identical to engines/_echo and SubprocessBackend: a 4-byte
big-endian length, then UTF-8 JSON. Ops: ping -> pong; shutdown; and
edit -> zero or more {"op":"progress","fraction":f}, then
{"op":"audio", ...reply} or {"op":"error","message":...}. The model loads
on the first edit, so the ready frame is instant and a sidecar that is
never asked to edit never touches the GPU.
"""
from __future__ import annotations

import json
import struct
import sys
import traceback
from typing import Callable

MAX_FRAME_BYTES = 64 * 1024 * 1024


def send(stream, obj: dict) -> None:
    body = json.dumps(obj, separators=(",", ":")).encode("utf-8")
    stream.write(struct.pack("!I", len(body)))
    stream.write(body)
    stream.flush()


def recv(stream):
    header = stream.read(4)
    if len(header) < 4:
        return None
    (n,) = struct.unpack("!I", header)
    if n > MAX_FRAME_BYTES:
        raise IOError(f"frame too large: {n}")
    body = bytearray()
    while len(body) < n:
        chunk = stream.read(n - len(body))
        if not chunk:
            raise IOError("short read")
        body.extend(chunk)
    return json.loads(bytes(body).decode("utf-8"))


def serve(engine_name: str, load: Callable[[], object],
          edit: Callable[[object, dict, Callable[[float], None]], dict], out) -> int:
    """Run the wire loop. ``out`` is the private frame stream the entry point
    opened after ``_frame_fd = os.dup(1); os.dup2(2, 1)`` (#1428): the stdout
    isolation guard checks every engines/*/main.py for that pair itself."""
    stdin = sys.stdin.buffer
    send(out, {"op": "ready", "engine": engine_name})
    model = None

    def progress(fraction: float) -> None:
        send(out, {"op": "progress", "fraction": max(0.0, min(1.0, float(fraction)))})

    while True:
        try:
            msg = recv(stdin)
        except Exception as exc:
            send(out, {"op": "error", "stage": "recv", "message": f"{type(exc).__name__}: {exc}"})
            return 1
        if msg is None:
            return 0
        op = msg.get("op")
        if op == "ping":
            send(out, {"op": "pong"})
        elif op == "shutdown":
            return 0
        elif op == "edit":
            try:
                if model is None:
                    model = load()
                reply = edit(model, msg, progress)
                send(out, {"op": "audio", **reply})
            except Exception as exc:
                traceback.print_exc()
                send(out, {"op": "error", "stage": "edit",
                           "message": f"{type(exc).__name__}: {exc}"[:2000]})
        else:
            send(out, {"op": "error", "stage": "dispatch", "message": f"unknown op: {op!r}"})
