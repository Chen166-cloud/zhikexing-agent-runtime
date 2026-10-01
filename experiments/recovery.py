"""Process-crash recovery experiments with real PostgreSQL and Java business APIs.

Run: python -m experiments.recovery --rounds 20 --output experiments/results/recovery
This owns only its runtime child, and does not restart Java, MQ or the databases.
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import httpx

from .common import (BusinessOracle, Config, RealJavaProxy, RuntimeProcess, Services,
                     append_json, manifest, percentile, safe_error, stamp, write_json)


FAULTS = ("persisted_queue_kill", "approval_wait_restart", "committed_response_lost", "duplicate_commands")
RESERVATION_PROMPT = "帮我预约课程，姓名为实验学员，联系方式13800000000。请先查询课程和校区，再生成预约草稿等我批准。"


def assert_oracle(value, expected):
    if value["coverage"] == "mysql-read-only":
        return {"sideEffectCount": value["sideEffectCount"] == expected,
                "noDuplicateActionRows": value["duplicateActionRows"] == 0,
                "noUnapprovedSideEffects": value["unapprovedSideEffects"] == 0,
                "reservationsExist": value["missingReservationRows"] == 0}
    return {"databaseSideEffectsVerified": None}


def run_case(fault, number, process, api, proxy, oracle, timeout):
    row = {"caseId": f"{fault}-{number:03}", "fault": fault, "startedAt": stamp(),
           "provider": "fixture", "toolTransport": "real-java-http", "faultInjected": False,
           "recoveryMs": None, "startupMs": None, "modelQualityScore": None}
    run = None
    try:
        process.start(parallel=0 if fault == "persisted_queue_kill" else 4)
        prompt = "请回复实验已收到。" if fault == "persisted_queue_kill" else RESERVATION_PROMPT
        run, body = api.submit(prompt, mode="chat" if fault == "persisted_queue_kill" else "agent")
        row["runId"] = run
        if fault == "persisted_queue_kill":
            api.wait_run(run, {"QUEUED"}, timeout=timeout)
            row["beforeFaultCounts"] = api.runtime_counts(run)
            process.kill()
            row["faultInjected"] = True
            row["startupMs"] = process.start(4)
            restored = time.monotonic()
            final = api.wait_run(run, {"SUCCEEDED"}, timeout=timeout)
            row["recoveryMs"] = round((time.monotonic() - restored) * 1000, 3)
            expected_effects = 0
        else:
            api.wait_run(run, {"WAITING_APPROVAL"}, timeout=timeout)
            approval = api.approval(run)
            row["actionId"], row["approvalId"] = approval["actionId"], approval["id"]
            row["beforeApprovalOracle"] = oracle.inspect(run)
            row["checks"] = {"waitsForApproval": approval["status"] == "PENDING",
                             **{"before_" + key: value for key, value in
                                assert_oracle(row["beforeApprovalOracle"], 0).items()}}
            payload = {"actionId": approval["actionId"], "approvalId": approval["id"]}
            if fault == "approval_wait_restart":
                process.kill()
                row["faultInjected"] = True
                # A real human decision can commit in Java while runtime is down.
                api.decide(approval, "APPROVED")
                row["startupMs"] = process.start(4)
                restored = time.monotonic()
            elif fault == "committed_response_lost":
                proxy.arm(run)
                api.decide(approval, "APPROVED")
                if not proxy.committed.wait(timeout=min(timeout, 50)):
                    raise TimeoutError("No committed Java execute response reached the fault proxy")
                row["faultInjected"] = True
                row["committedBeforeCrashOracle"] = oracle.inspect(run)
                process.kill()
                proxy.release.set()
                row["startupMs"] = process.start(4)
                restored = time.monotonic()
            else:
                # Replay both Java acceptance and Python command-delivery boundaries.
                # This is HTTP command replay, not a claim of forced broker redelivery.
                replays = []
                for _ in range(10):
                    replays.append(api.java("POST", "/runs", json=body)["runId"])
                    delivered = dict(body, runId=run, actorId=api.config.actor)
                    headers = dict(api.config.headers(), **{"X-Command-Delivery": "true"})
                    response = api.client.post(api.config.runtime + "/internal/v1/runs",
                                               headers=headers, json=delivered)
                    response.raise_for_status()
                    replays.append(response.json()["runId"])
                row["replayRunIdsEqual"] = all(identifier == run for identifier in replays)
                row["commandReplayCount"] = len(replays)
                row["faultInjected"] = True
                api.decide(approval, "APPROVED")
                restored = time.monotonic()
            final = api.wait_run(run, {"SUCCEEDED"}, timeout=timeout)
            row["recoveryMs"] = round((time.monotonic() - restored) * 1000, 3)
            if fault == "duplicate_commands":
                receipts = [api.tool(run, "/execute-reservation", method="POST", payload=payload)
                            for _ in range(5)]
                row["replayedExecutionCount"] = len(receipts)
                row["executionReceiptsEqual"] = len({r["reservationId"] for r in receipts}) == 1
            current_approval = api.approval(run)
            actual = api.tool(run, "/reservations/by-action/" + approval["actionId"])
            row["businessResult"] = actual
            row["checks"].update({"sameApprovalAfterRecovery": current_approval["id"] == approval["id"],
                                  "approvalExecuted": current_approval["status"] == "EXECUTED",
                                  "actualReservationId": bool(actual.get("reservationId"))})
            expected_effects = 1
        row["finalStatus"] = final["status"]
        row["runtimeCounts"] = api.runtime_counts(run)
        row["businessOracle"] = oracle.inspect(run)
        events = api.events(run)
        row["eventCount"] = len(events)
        row["terminalEventCount"] = sum(event["type"] == "run.completed" for event in events)
        row.setdefault("checks", {}).update({"oneRun": row["runtimeCounts"]["runCount"] == 1,
                                               "oneInput": row["runtimeCounts"]["messageCounts"].get("input") == 1,
                                               "oneAnswer": row["runtimeCounts"]["messageCounts"].get("answer") == 1,
                                               "oneCompletion": row["terminalEventCount"] == 1,
                                               **assert_oracle(row["businessOracle"], expected_effects)})
        for key in ("replayRunIdsEqual", "executionReceiptsEqual"):
            if key in row:
                row["checks"][key] = row[key]
        row["passed"] = row["faultInjected"] and all(value is not False for value in row["checks"].values())
        row["databaseOracleComplete"] = oracle.enabled
    except Exception as exc:
        row["passed"], row["error"] = False, safe_error(exc)
        if run:
            try:
                row["businessOracle"] = oracle.inspect(run)
                row["failureRunSnapshot"] = api.runtime("GET", "/runs/" + run)
                api.java("POST", f"/runs/{run}/cancel", json={"workspaceId": api.config.workspace})
            except Exception as cleanup:
                row["cleanupError"] = safe_error(cleanup)
    finally:
        if run:
            row["javaToolCalls"] = proxy.for_run(run)
        process.kill()
        proxy.release.set()
    row["finishedAt"] = stamp()
    return row


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["fault"]].append(row)
    result = {}
    for fault, cases in groups.items():
        successful = [case for case in cases if case["passed"]]
        result[fault] = {"attempted": len(cases), "passed": len(successful),
                         "passRate": len(successful) / len(cases),
                         "injected": sum(case["faultInjected"] for case in cases),
                         "successfulRecoveryP95Ms": percentile([case["recoveryMs"] for case in successful
                                                                if case["recoveryMs"] is not None]),
                         "successfulStartupP95Ms": percentile([case["startupMs"] for case in successful
                                                               if case["startupMs"] is not None]),
                         "databaseOracleCases": sum(case.get("databaseOracleComplete", False) for case in cases),
                         "duplicateActionRows": sum(case.get("businessOracle", {}).get("duplicateActionRows", 0)
                                                    for case in cases) if all(case.get("businessOracle", {}).get("coverage") == "mysql-read-only" for case in cases) else None,
                         "unapprovedSideEffects": sum(case.get("businessOracle", {}).get("unapprovedSideEffects", 0)
                                                       for case in cases) if all(case.get("businessOracle", {}).get("coverage") == "mysql-read-only" for case in cases) else None}
    return {"kind": "fixture-engineering-recovery", "modelQualityScore": None,
            "recoveryClock": "runtime health ready to expected terminal; excludes process startup; duplicate_commands measures approval-to-completion",
            "groups": result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--faults", nargs="+", choices=FAULTS, default=list(FAULTS))
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-db-oracle", action="store_true")
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds must be positive")
    if (args.output / "cases.jsonl").exists():
        parser.error("Output already contains cases.jsonl; use a new run directory")
    config = Config.environment()
    oracle = BusinessOracle()
    if args.require_db_oracle and not oracle.enabled:
        parser.error("MySQL oracle environment is required")
    write_json(args.output / "manifest.json", manifest(config, "fixture", {"roundsPerFault": args.rounds,
               "faults": args.faults, "databaseOracle": oracle.enabled,
               "declaredLimitations": ["single runtime worker", "model is deterministic fixture",
                                       "command replay is HTTP, not forced broker redelivery"]}))
    proxy, api = RealJavaProxy(config.java), Services(config)
    process = RuntimeProcess(config, args.output, proxy.url)
    rows = []
    try:
        for fault in args.faults:
            for number in range(1, args.rounds + 1):
                row = run_case(fault, number, process, api, proxy, oracle, args.timeout)
                rows.append(row)
                append_json(args.output / "cases.jsonl", row)
                write_json(args.output / "summary.json", summarize(rows))
                print(f"{row['caseId']}: {'PASS' if row['passed'] else 'FAIL'}", flush=True)
    finally:
        process.kill()
        proxy.close()
        api.close()
        oracle.close()
    return 0 if all(row["passed"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
