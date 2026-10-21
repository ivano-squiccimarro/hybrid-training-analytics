"""Loopback-only desktop server. No general file serving or external requests."""

from __future__ import annotations

import hmac
import json
import secrets
from contextlib import suppress
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import RLock
from urllib.parse import urlsplit

from .local_catalog import LocalCatalog
from .models import DataError
from .strength_catalog import StrengthCatalog
from .strength_evaluation import StrengthEvaluation
from .strength_sessions import StrengthSessions


class LocalServer(ThreadingHTTPServer):
    def __init__(self, workspace: Path, port: int = 8765):
        self.catalog_lock = RLock()
        self.catalog = LocalCatalog(workspace)
        self.strength_catalog = StrengthCatalog(workspace)
        self.strength_sessions = StrengthSessions(workspace)
        self.strength_evaluation = StrengthEvaluation(workspace)
        self.token = secrets.token_urlsafe(32)
        super().__init__(("127.0.0.1", port), LocalHandler)
        self.origin = f"http://127.0.0.1:{self.server_port}"

    @property
    def url(self) -> str:
        return self.origin + "/#" + self.token


class LocalHandler(BaseHTTPRequestHandler):
    server: LocalServer

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(10)

    def log_message(self, format: str, *args) -> None:
        # Never log tokens, request bodies, paths or personal values.
        pass

    def send_content(self, status: int, content: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            (
                "default-src 'self'; script-src 'self'; style-src 'self'; "
                "connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'; "
                "base-uri 'none'; form-action 'self'"
            ),
        )
        self.end_headers()
        self.wfile.write(content)

    def respond(self, status: int, payload: dict | list) -> None:
        self.send_content(
            status,
            json.dumps(payload, ensure_ascii=False, allow_nan=False).encode(),
            "application/json; charset=utf-8",
        )

    def authorized(self) -> bool:
        if self.headers.get("Host") != self.server.origin.removeprefix("http://"):
            return False
        origin = self.headers.get("Origin")
        if origin is not None and origin != self.server.origin:
            return False
        if self.headers.get("Sec-Fetch-Site") == "cross-site":
            return False
        # Compare bytes: str comparison raises TypeError on non-ASCII header values.
        return hmac.compare_digest(
            self.headers.get("X-HTA-Token", "").encode(), self.server.token.encode()
        )

    def do_GET(self) -> None:
        with self.server.catalog_lock:
            self._get()

    def _get(self) -> None:
        path = urlsplit(self.path).path
        if self.headers.get("Host") != self.server.origin.removeprefix("http://"):
            self.respond(403, {"error": "Host non consentito"})
            return
        assets = {
            "/strength": ("strength_reconcile.html", "text/html; charset=utf-8"),
            "/strength.js": ("strength_reconcile.js", "text/javascript; charset=utf-8"),
            "/strength.css": ("strength_reconcile.css", "text/css; charset=utf-8"),
            "/": ("local_ui.html", "text/html; charset=utf-8"),
            "/app.js": ("local_ui.js", "text/javascript; charset=utf-8"),
            "/app.css": ("local_ui.css", "text/css; charset=utf-8"),
        }
        if path in assets:
            name, kind = assets[path]
            self.send_content(200, (Path(__file__).parent / name).read_bytes(), kind)
            return
        if not self.authorized():
            self.respond(403, {"error": "Sessione locale non autorizzata"})
            return
        try:
            if path == "/api/strength-evaluation":
                self.respond(200, self.server.strength_evaluation.metrics())
            elif path == "/api/strength-sessions":
                self.respond(200, self.server.strength_sessions.sessions())
            elif path.startswith("/api/strength-sessions/view/"):
                self.respond(
                    200,
                    self.server.strength_sessions.view(
                        path.removeprefix("/api/strength-sessions/view/")
                    ),
                )
            elif path == "/api/strength-sessions/csv-links":
                self.respond(
                    200,
                    [
                        {"session_id": s}
                        for s in sorted({r.session_id for r in self.server.catalog.strength_sets()})
                    ],
                )
            elif path == "/api/strength-sessions/ux":
                self.respond(200, self.server.strength_sessions.ux_metrics())
            elif path == "/api/strength-fit/sessions":
                self.respond(200, self.server.strength_catalog.sessions())
            elif path == "/api/strength-fit/context":
                self.respond(200, self.server.strength_catalog.context())
            elif path.startswith("/api/strength-fit/draft/"):
                key = path.removeprefix("/api/strength-fit/draft/")
                catalog = self.server.strength_catalog
                self.respond(
                    200,
                    {
                        "raw": catalog.raw(key),
                        "draft": catalog.latest(key),
                        "quality": catalog.quality(key),
                    },
                )
            elif path == "/api/runs":
                self.respond(200, self.server.catalog.runs())
            elif path == "/api/hybrid":
                self.respond(200, self.server.catalog.hybrid())
            else:
                self.respond(404, {"error": "Rotta non disponibile"})
        except (DataError, OSError, ValueError, TypeError, KeyError):
            self.respond(
                400, {"error": "Dati locali non validi o mancanti; conservati per diagnosi"}
            )

    def read_body(self) -> bytes:
        length = self.headers.get("Content-Length", "")
        if not length.isdigit() or self.headers.get("Transfer-Encoding") is not None:
            raise DataError("Content-Length richiesto; chunked non supportato")
        size = int(length)
        if not 0 < size <= 10 * 1024 * 1024:
            raise DataError("Richiesta vuota o oltre 10 MiB")
        content = self.rfile.read(size)
        if len(content) != size:
            raise DataError("Corpo richiesta troncato")
        return content

    def do_POST(self) -> None:
        with self.server.catalog_lock:
            self._post()

    def _post(self) -> None:
        if not self.authorized():
            self.respond(403, {"error": "Sessione/origine locale non autorizzata"})
            return
        try:
            path = urlsplit(self.path).path
            content = self.read_body()
            catalog = self.server.catalog
            if path == "/api/strength-fit/import":
                key = self.server.strength_catalog.import_fit(content)
                self.respond(200, {"session_id": key})
                return
            if path == "/api/import-fit":
                self.respond(200, {"run_id": catalog.import_fit(content)})
                return
            if path == "/api/import-strength":
                self.respond(200, {"exercises": catalog.import_strength(content)})
                return
            if self.headers.get_content_type() != "application/json" or len(content) > 128 * 1024:
                raise DataError("JSON richiesto, massimo 128 KiB")
            request = json.loads(content)
            if not isinstance(request, dict):
                raise DataError("Oggetto JSON richiesto")
            strength = self.server.strength_catalog
            if path == "/api/strength-evaluation/capture":
                trial = self.server.strength_evaluation.capture(
                    request["session_id"], request["slot_id"]
                )
                self.respond(
                    200,
                    {
                        "trial": trial,
                        "label": self.server.strength_evaluation.latest_label(trial["trial_id"]),
                    },
                )
            elif path == "/api/strength-evaluation/label":
                self.respond(
                    200,
                    self.server.strength_evaluation.label(
                        request["trial_id"], request.get("parent_revision_id"), request["values"]
                    ),
                )
            elif path == "/api/strength-evaluation/export":
                self.respond(200, self.server.strength_evaluation.metrics())
            elif path == "/api/strength-sessions/save":
                self.respond(
                    200,
                    self.server.strength_sessions.save(
                        request.get("session_id"),
                        request.get("parent_revision_id"),
                        request["values"],
                    ),
                )
            elif path == "/api/strength-sessions/export":
                self.send_content(
                    200,
                    self.server.strength_sessions.export(request["session_id"]),
                    "application/json; charset=utf-8",
                )
            elif path == "/api/strength-fit/edit":
                self.respond(
                    200,
                    strength.edit(
                        request["session_id"], request["parent_revision_id"], request["operation"]
                    ),
                )
            elif path == "/api/strength-fit/suggest":
                self.respond(
                    200,
                    strength.suggestions(
                        request["session_id"], request["slot_id"], request.get("muscle")
                    ),
                )
            elif path == "/api/strength-fit/context":
                self.respond(200, strength.save_context(request))
            elif path == "/api/strength-fit/export-quality":
                self.send_content(
                    200,
                    strength.export_quality(request["session_id"]),
                    "application/json; charset=utf-8",
                )
            elif path == "/api/strength-fit/export":
                kind = request.get("kind", "json")
                self.send_content(
                    200,
                    strength.export(request["session_id"], kind),
                    "application/json" if kind == "json" else "text/csv",
                )
            elif path == "/api/annotate":
                self.respond(200, catalog.annotate(request["subject"], request["values"]))
            elif path == "/api/analyze":
                self.respond(200, catalog.analyze(request["run_id"], request["config"]))
            elif path == "/api/compare":
                self.respond(200, catalog.compare(request["baseline"], request["comparison"]))
            elif path == "/api/download":
                file = catalog.output_file(request["output_id"], request["filename"])
                self.send_content(200, file.read_bytes(), "application/octet-stream")
            else:
                self.respond(404, {"error": "Rotta non disponibile"})
        except (
            DataError,
            OSError,
            ValueError,
            TypeError,
            KeyError,
            RecursionError,  # deeply nested JSON
        ) as exc:
            # Domain messages contain no raw file bytes or request body.
            message = (
                str(exc) if isinstance(exc, DataError) else "Richiesta non valida o file mancante"
            )
            self.respond(400, {"error": message})


def serve(workspace: Path, port: int = 8765) -> None:
    with LocalServer(workspace, port) as server:
        print(f"Interfaccia locale: {server.url}", flush=True)
        print("Solo loopback. Ctrl+C per arrestare questo server.", flush=True)
        with suppress(KeyboardInterrupt):
            server.serve_forever()
