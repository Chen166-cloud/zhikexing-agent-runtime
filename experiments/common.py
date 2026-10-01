"""Shared experiment harness. Credentials stay in environment variables.

Only child processes started by RuntimeProcess are ever killed. A real Java tool
proxy can withhold an already committed response; it never synthesizes a success.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import psycopg


ROOT = Path(__file__).resolve().parents[1]
TERMINAL = {"SUCCEEDED", "FAILED", "CANCELLED", "TIMED_OUT"}


def stamp():
    return datetime.now(timezone.utc).isoformat()


def required(name):
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"Missing environment variable: {name}")
    return value


def percentile(values, quantile=.95):
    """Nearest-rank quantile; retains all failed cases in separate denominators."""
    values = sorted(values)
    return values[max(0, math.ceil(len(values) * quantile) - 1)] if values else None


def append_json(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def safe_error(exc):
    # Do not serialize exception text: DSNs, credentials and request bodies may occur there.
    if isinstance(exc, httpx.HTTPStatusError):
        return {"type": type(exc).__name__, "status": exc.response.status_code,
                "path": exc.request.url.path}
    return {"type": type(exc).__name__}


@dataclass
class Config:
    database: str
    java: str
    internal_token: str
    auth_token: str
    actor: str
    workspace: str
    port: int
    storage: Path

    @classmethod
    def environment(cls):
        database = required("AGENT_EXPERIMENT_DATABASE_URL")
        if not database.startswith(("postgresql://", "postgresql+psycopg://")):
            raise ValueError("Recovery/evaluation experiments require a dedicated PostgreSQL database")
        if os.getenv("AGENT_EXPERIMENT_ISOLATED") != "yes":
            raise ValueError("Set AGENT_EXPERIMENT_ISOLATED=yes only for dedicated experiment services")
        return cls(database, required("AGENT_EXPERIMENT_JAVA_URL").rstrip("/"),
                   required("AGENT_INTERNAL_TOKEN"), required("AGENT_EXPERIMENT_AUTH_TOKEN"),
                   required("AGENT_EXPERIMENT_ACTOR_ID"), required("AGENT_EXPERIMENT_WORKSPACE_ID"),
                   int(os.getenv("AGENT_EXPERIMENT_RUNTIME_PORT", "18100")),
                   Path(required("AGENT_EXPERIMENT_STORAGE_ROOT")).resolve())

    @property
    def runtime(self):
        return f"http://127.0.0.1:{self.port}"

    @property
    def pg_dsn(self):
        return self.database.replace("postgresql+psycopg://", "postgresql://", 1)

    def headers(self, workspace=None, run=None):
        value = {"X-Internal-Token": self.internal_token, "X-Actor-Id": self.actor,
                 "X-Workspace-Id": workspace or self.workspace}
        if run:
            value["X-Run-Id"] = run
        return value


class RuntimeProcess:
    def __init__(self, config, directory, proxy_url, provider="fixture"):
        self.config, self.directory, self.proxy_url = config, directory, proxy_url
        self.provider, self.process, self.log = provider, None, None
        self.starts = 0

    def start(self, parallel=4):
        if self.process and self.process.poll() is None:
            raise RuntimeError("Owned runtime is already running")
        # Do not take over an existing service at the requested port.
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", self.config.port)) == 0:
                raise RuntimeError("Requested runtime port already has a listener")
        self.directory.mkdir(parents=True, exist_ok=True)
        self.starts += 1
        environment = dict(os.environ)
        for key in list(environment):
            if key.lower() in {"http_proxy", "https_proxy", "all_proxy"}:
                environment.pop(key)
        environment.update({"AGENT_DATABASE_URL": self.config.database,
                            "JAVA_TOOL_BASE_URL": self.proxy_url,
                            "STORAGE_ROOT": str(self.config.storage),
                            "AI_PROVIDER": self.provider, "PORT": str(self.config.port),
                            "AGENT_MAX_PARALLEL": str(parallel),
                            "AGENT_INTERNAL_TOKEN": self.config.internal_token,
                            "S3_ENDPOINT_URL": "", "OTEL_EXPORTER_OTLP_ENDPOINT": "",
                            "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": "", "LANGFUSE_BASE_URL": "",
                            "LANGFUSE_PUBLIC_KEY": "", "LANGFUSE_SECRET_KEY": ""})
        self.log = (self.directory / f"runtime-{self.starts}.log").open("wb")
        self.process = subprocess.Popen([sys.executable, "-m", "experiments.worker"], cwd=ROOT,
                                        env=environment, stdout=self.log, stderr=subprocess.STDOUT,
                                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        started = time.monotonic()
        while time.monotonic() - started < 45:
            if self.process.poll() is not None:
                raise RuntimeError("Experiment runtime exited during startup; inspect its local log")
            try:
                response = httpx.get(self.config.runtime + "/health", timeout=1, trust_env=False)
                if response.status_code == 200:
                    health = response.json()
                    if health.get("storage") != "postgresql" or health.get("provider") != self.provider:
                        raise RuntimeError("Unexpected runtime provider/storage")
                    return (time.monotonic() - started) * 1000
            except httpx.HTTPError:
                pass
            time.sleep(.15)
        raise TimeoutError("Experiment runtime startup timed out")

    def kill(self):
        if self.process and self.process.poll() is None:
            self.process.kill()
            self.process.wait(timeout=15)
        if self.log:
            self.log.close()
        self.process, self.log = None, None

    def restart(self, parallel=4):
        self.kill()
        return self.start(parallel)


class RealJavaProxy:
    """Real HTTP tool forwarding plus an explicitly selected lost-response fault.

    The Java response is retained in the audit before it is withheld. The caller
    kills its runtime child then releases the blocked proxy handler. All normal
    traffic is forwarded unmodified; credentials are never written to the audit.
    """

    def __init__(self, java):
        self.java, self.calls = java, []
        self.lock = threading.Lock()
        self.armed_run = None
        self.committed = threading.Event()
        self.release = threading.Event()
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_GET(self):
                self.forward()

            def do_POST(self):
                self.forward()

            def forward(self):
                started = time.monotonic()
                raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                headers = {key: value for key, value in self.headers.items()
                           if key.lower() not in {"host", "connection", "content-length"}}
                run = headers.get("X-Run-Id", headers.get("x-run-id"))
                try:
                    with httpx.Client(timeout=30, trust_env=False) as client:
                        reply = client.request(self.command, owner.java + self.path,
                                               headers=headers, content=raw)
                    try:
                        result = reply.json()
                    except ValueError:
                        result = {"nonJson": True}
                    record = {"runId": run, "method": self.command, "path": self.path,
                              "status": reply.status_code, "result": result,
                              "latencyMs": round((time.monotonic() - started) * 1000, 3),
                              "responseWithheld": False}
                    with owner.lock:
                        owner.calls.append(record)
                        withhold = (owner.armed_run == run and reply.is_success
                                    and self.path.endswith(("/execute-reservation", "/execute-trial-claim")))
                        if withhold:
                            owner.armed_run = None
                            record["responseWithheld"] = True
                    if withhold:
                        owner.committed.set()
                        owner.release.wait(timeout=60)
                        self.close_connection = True
                        return
                    self.send_response(reply.status_code)
                    self.send_header("Content-Type", reply.headers.get("Content-Type", "application/json"))
                    self.send_header("Content-Length", str(len(reply.content)))
                    self.end_headers()
                    self.wfile.write(reply.content)
                except (httpx.HTTPError, BrokenPipeError, ConnectionResetError):
                    self.close_connection = True

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server.server_port}"

    def arm(self, run):
        self.release.clear()
        self.committed.clear()
        self.armed_run = run

    def for_run(self, run):
        with self.lock:
            return [dict(call) for call in self.calls if call["runId"] == run]

    def close(self):
        self.release.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3)


class Services:
    def __init__(self, config):
        self.config = config
        self.client = httpx.Client(timeout=40, trust_env=False)

    def java(self, method, path, *, workspace=None, **kwargs):
        response = self.client.request(method, self.config.java + "/api/v1" + path,
                                       headers={"Authorization": self.config.auth_token}, **kwargs)
        response.raise_for_status()
        return response.json()

    def runtime(self, method, path, *, workspace=None, **kwargs):
        response = self.client.request(method, self.config.runtime + "/internal/v1" + path,
                                       headers=self.config.headers(workspace), **kwargs)
        response.raise_for_status()
        return response.json()

    def submit(self, prompt, workspace=None, mode="agent", knowledge=None):
        workspace = workspace or self.config.workspace
        conversation = self.java("POST", "/conversations", json={"workspaceId": workspace,
                                  "title": "experiment-" + uuid.uuid4().hex[:12]})
        body = {"workspaceId": workspace, "conversationId": conversation["id"],
                "clientRequestId": "exp-" + uuid.uuid4().hex, "input": prompt,
                "mode": mode, "knowledgeBaseIds": knowledge or []}
        receipt = self.java("POST", "/runs", json=body)
        return receipt["runId"], body

    def wait_run(self, run, states, workspace=None, timeout=100):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            try:
                value = self.runtime("GET", "/runs/" + run, workspace=workspace)
                if value["status"] in states:
                    return value
                if value["status"] in TERMINAL:
                    raise RuntimeError("Run reached unexpected terminal state: " + value["status"])
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code != 404:
                    raise
            time.sleep(.15)
        raise TimeoutError("Run did not reach expected state")

    def approval(self, run, workspace=None):
        approvals = self.java("GET", "/approvals", params={"workspaceId": workspace or self.config.workspace,
                                                           "runId": run})
        if len(approvals) != 1:
            raise RuntimeError("Expected exactly one approval")
        return approvals[0]

    def decide(self, approval, decision, workspace=None):
        return self.java("POST", f"/approvals/{approval['id']}/decision",
                         json={"workspaceId": workspace or self.config.workspace,
                               "decision": decision, "expectedVersion": approval["version"]})

    def tool(self, run, path, workspace=None, method="GET", payload=None, allow_missing=False):
        response = self.client.request(method, self.config.java + "/internal/v1/tools" + path,
                                       headers=self.config.headers(workspace, run), json=payload)
        if allow_missing and response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()

    def events(self, run):
        with psycopg.connect(self.config.pg_dsn) as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT seq,type,data FROM agent_event WHERE run_id=%s ORDER BY seq", (run,))
                return [{"seq": row[0], "type": row[1], "data": row[2]} for row in cursor.fetchall()]

    def runtime_counts(self, run):
        with psycopg.connect(self.config.pg_dsn) as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM agent_run WHERE id=%s", (run,))
                run_count = cursor.fetchone()[0]
                cursor.execute("SELECT message_key,COUNT(*) FROM agent_message WHERE run_id=%s GROUP BY message_key", (run,))
                return {"runCount": run_count, "messageCounts": dict(cursor.fetchall())}

    def close(self):
        self.client.close()


class BusinessOracle:
    """Read-only MySQL oracle, enabled only when all dedicated DB settings exist."""
    def __init__(self):
        self.enabled = bool(os.getenv("AGENT_EXPERIMENT_MYSQL_DATABASE"))
        if self.enabled:
            import pymysql
            self.connection = pymysql.connect(
                host=required("AGENT_EXPERIMENT_MYSQL_HOST"),
                port=int(os.getenv("AGENT_EXPERIMENT_MYSQL_PORT", "3306")),
                user=required("AGENT_EXPERIMENT_MYSQL_USER"),
                password=os.getenv("AGENT_EXPERIMENT_MYSQL_PASSWORD", ""),
                database=required("AGENT_EXPERIMENT_MYSQL_DATABASE"), autocommit=True,
                cursorclass=pymysql.cursors.DictCursor)

    def inspect(self, run):
        if not self.enabled:
            return {"coverage": "api-only", "databaseSideEffectAssertions": None}
        with self.connection.cursor() as cursor:
            cursor.execute("SELECT a.action_id,a.approval_id,a.reservation_id,p.status AS approval_status,"
                           "p.args AS approval_args,p.result AS approval_result,"
                           "r.id AS actual_reservation_id,r.course,r.student_name,r.contact_info,r.school "
                           "FROM agent_reservation_action a "
                           "LEFT JOIN agent_approval p ON p.id=a.approval_id "
                           "LEFT JOIN course_reservation r ON r.id=a.reservation_id WHERE a.run_id=%s", (run,))
            reservations = list(cursor.fetchall())
            cursor.execute("SELECT q.id AS request_id,q.action_id,q.approval_id,q.status,p.status AS approval_status,"
                           "p.args AS approval_args,p.result AS approval_result,o.id AS order_id FROM trial_claim_request q "
                           "LEFT JOIN agent_approval p ON p.id=q.approval_id "
                           "LEFT JOIN trial_order o ON o.request_id=q.id WHERE q.run_id=%s", (run,))
            trials = list(cursor.fetchall())
            cursor.execute("SELECT id,action_id,tool_name,status,args,result FROM agent_approval WHERE run_id=%s", (run,))
            approvals = list(cursor.fetchall())
        for row in reservations + trials + approvals:
            for field in ("approval_args", "approval_result", "args", "result"):
                if isinstance(row.get(field), str):
                    row[field] = json.loads(row[field])
        effects = reservations + trials
        keys = [("reservation", row["action_id"]) for row in reservations]
        keys += [("trial", row["action_id"]) for row in trials]
        return {"coverage": "mysql-read-only", "reservations": reservations, "trials": trials, "approvals": approvals,
                "sideEffectCount": len(effects), "duplicateActionRows": len(keys) - len(set(keys)),
                "unapprovedSideEffects": sum(row["approval_status"] != "EXECUTED" for row in effects),
                "missingReservationRows": sum(row["actual_reservation_id"] is None for row in reservations)}

    def close(self):
        if self.enabled:
            self.connection.close()


def manifest(config, provider, extra=None):
    commit = os.getenv("AGENT_EXPERIMENT_REPOSITORY_COMMIT", "").strip()
    if not commit and shutil.which("git"):
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    if not commit:
        raise ValueError("Container without git requires AGENT_EXPERIMENT_REPOSITORY_COMMIT from host")
    return {"createdAt": stamp(), "repositoryCommit": commit, "provider": provider,
            "runtimePort": config.port, "runtimeWorkerCount": 1,
            "database": "dedicated-postgresql", "toolTransport": "real-java-http",
            "javaHost": urlsplit(config.java).hostname,
            "storageRoot": str(config.storage), "externalModelCalls": provider != "fixture",
            "sourceFilesSha256": {str(path.relative_to(ROOT)).replace("\\", "/"): dataset_hash(path)
                                   for path in sorted((ROOT / "zhikexing_agent").glob("*.py"))},
            **(extra or {})}


def dataset_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
