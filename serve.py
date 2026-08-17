#!/usr/bin/env python3
"""Serve the project page locally WITH HTTP Range support.

Use this, not `python -m http.server`. SimpleHTTPRequestHandler ignores the Range
header and answers with 200 plus the whole file, and a browser cannot seek inside a
video without 206 Partial Content — so clicking a timeline marker appears to "reset
to the beginning" even though the page is doing the right thing. That wasted a review
cycle. GitHub Pages serves Range correctly, so this makes local review match
production.

    python serve.py [port]        # default 8080, binds 0.0.0.0
"""
import os
import re
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")


class RangeHandler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=HERE, **kw)

    def do_GET(self):
        rng = self.headers.get("Range")
        if not rng:
            return super().do_GET()
        path = self.translate_path(self.path)
        if not os.path.isfile(path):
            return super().do_GET()
        m = RANGE_RE.match(rng.strip())
        if not m:
            self.send_error(400, "malformed Range")
            return
        size = os.path.getsize(path)
        lo_s, hi_s = m.group(1), m.group(2)
        if lo_s == "":                                  # suffix form: bytes=-N
            length = min(int(hi_s or 0), size)
            lo, hi = size - length, size - 1
        else:
            lo = int(lo_s)
            hi = int(hi_s) if hi_s else size - 1
        hi = min(hi, size - 1)
        if lo > hi or lo >= size:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return

        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", f"bytes {lo}-{hi}/{size}")
        self.send_header("Content-Length", str(hi - lo + 1))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        remaining = hi - lo + 1
        with open(path, "rb") as f:
            f.seek(lo)
            while remaining > 0:
                chunk = f.read(min(64 * 1024, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return                              # the browser moved on; normal for video
                remaining -= len(chunk)

    def end_headers(self):
        # advertise range support even on plain GETs, so players know they may seek
        if "Accept-Ranges" not in self._headers_buffer_keys():
            self.send_header("Accept-Ranges", "bytes")
        super().end_headers()

    def _headers_buffer_keys(self):
        return b"".join(getattr(self, "_headers_buffer", []) or []).decode("latin-1", "replace")

    def log_message(self, fmt, *args):
        if "code 4" in (fmt % args) or "code 5" in (fmt % args):
            sys.stderr.write("%s\n" % (fmt % args))


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    print(f"serving {HERE} on 0.0.0.0:{port} with Range support")
    ThreadingHTTPServer(("0.0.0.0", port), RangeHandler).serve_forever()
