"""Serve the latest completed refit's two slice rasters on the GPU host's loopback only."""

from __future__ import annotations

import base64
import json
import logging
import sys
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import numpy as np

from .model import SURVIVAL_DENOMINATOR, SURVIVAL_NUMERATOR, Z_THRESHOLD

logger = logging.getLogger(__name__)


def no_view(*_args) -> None:
    """Run without a browser or terminal prompts."""


def encode(rgb: np.ndarray) -> str:
    """Encode one pixel-exact RGB frame without an optional image package."""
    return base64.b64encode(rgb.tobytes()).decode("ascii")


@contextmanager
def live_view(port: int, trials: int):
    """Keep a responsive local page while the main process fits on the GPU."""
    if not sys.stdin.isatty():
        raise RuntimeError("--live needs an interactive terminal; use batch mode for Slurm")
    page = Path(__file__).with_suffix(".html").read_bytes()
    layout = Path(__file__).with_name("layout.json").read_bytes()
    state = {"active": False, "message": "Loading the IXI reference and compiling the GPU fit…"}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            """Do not log the local page's refresh traffic."""

        def do_GET(self):
            """Expose only this page and its current state, never the workspace."""
            if self.path == "/":
                payload, content_type = page, "text/html; charset=utf-8"
            elif self.path == "/layout.json":
                payload, content_type = layout, "application/json"
            elif self.path == "/state":
                payload, content_type = json.dumps(state).encode(), "application/json"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(payload)

    def publish(trial, surviving, baseline, reference, current):
        """Replace both counters and image together, after a successful trial."""
        nonlocal state
        state = {"active": True, "trial": trial, "trials": trials, "surviving": surviving,
                 "baseline": baseline, "cutoff": Z_THRESHOLD, "recurrence": SURVIVAL_NUMERATOR / SURVIVAL_DENOMINATOR,
                 "width": reference.shape[1], "height": reference.shape[0],
                 "reference": encode(reference), "current": encode(current),
                 "message": "GPU baseline ready · press Enter in the terminal to start" if trial == 0 else ""}
        if trial == 0:
            input("GPU baseline ready. Press Enter to start refits. ")
            state = {**state, "message": "Running the first GPU refit…"}

    with ThreadingHTTPServer(("127.0.0.1", port), Handler) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        logger.info("live gpu viewer: http://127.0.0.1:%d/", server.server_port)
        try:
            yield publish
        finally:
            server.shutdown()
            thread.join()
