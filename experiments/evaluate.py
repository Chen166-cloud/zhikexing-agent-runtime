"""Full Agent graph contract evaluation, not the production retrieve+chat evaluator.

Real model calls require --provider bailian --allow-real-model and an explicit
case limit. Fixture outcomes only describe engineering behavior, never quality.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import uuid
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .common import (BusinessOracle, Config, RealJavaProxy, RuntimeProcess, Services,
                     TERMINAL, append_json, dataset_hash, manifest, percentile,
                     safe_error, stamp, write_json)


HERE = Path(__file__).resolve().parent
GRADER_VERSION = "agent-contract-v2-business-fields"


def explicit_reservation_fields(prompt):
    """Extract only explicitly supplied fields from the frozen synthetic prompts.

    Dataset labels remain untouched. Course/campus choices are not inferred from
    vague preferences; all actual writes must still agree with the approved draft.
    """
    name = re.search(r"(?:姓名(?:为|是)?|我叫|名字(?:为|是|叫)?)\s*[:：]?\s*([^\s，,。；;！？!?、]+)", prompt)
    if not name:
        name = re.search(r"(?:帮|为)([\u4e00-\u9fffA-Za-z·]{2,16}?)(?:预约|准备)", prompt)
    numbers = list(dict.fromkeys(re.findall(r"(?<!\d)(1[3-9]\d{9})(?!\d)", prompt)))
    return {"studentName": name.group(1) if name else None,
            "contactInfo": numbers[0] if len(numbers) == 1 else None,
            "extraction": "explicit-name-markers-or-for-person; unique-11-digit-mobile",
            "ambiguousContact": len(numbers) > 1}


def reservation_field_checks(case, business, approval):
    if not case.get("category", "").startswith("reservation_"):
        return {}, None
    expected = explicit_reservation_fields(case["prompt"])
    drafts = [value for value in business.get("approvals", [])
              if value.get("tool_name", "reserve_course") == "reserve_course"]
    if not drafts and approval:
        drafts = [approval]
    checks = {"draftAvailableForFieldVerification": bool(drafts),
              "unambiguousExplicitContact": not expected["ambiguousContact"]}
    for key in ("studentName", "contactInfo"):
        if expected[key] is not None:
            checks["draftMatchesExplicit:" + key] = bool(drafts) and all(
                (draft.get("args") or {}).get(key) == expected[key] for draft in drafts)
    reservations = business.get("reservations", [])
    for row in reservations:
        args = row.get("approval_args") or {}
        for database_key, draft_key in (("student_name", "studentName"), ("contact_info", "contactInfo"),
                                        ("course", "courseName"), ("school", "schoolName")):
            key = "persistedMatchesApproved:" + draft_key
            checks[key] = checks.get(key, True) and args.get(draft_key) is not None and row.get(database_key) == args[draft_key]
        for database_key, explicit_key in (("student_name", "studentName"), ("contact_info", "contactInfo")):
            if expected[explicit_key] is not None:
                key = "persistedMatchesExplicit:" + explicit_key
                checks[key] = checks.get(key, True) and row.get(database_key) == expected[explicit_key]
    if case.get("approvalDecision") == "APPROVED":
        checks["persistedRowAvailableForFieldVerification"] = bool(reservations)
    return checks, expected


def load_cases(path):
    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Duplicate dataset IDs")
    for case in cases:
        for field in ("id", "category", "prompt", "oracle", "split"):
            if field not in case:
                raise ValueError(f"Missing dataset field: {field}")
        if not isinstance(case["oracle"].get("requiredTools"), list):
            raise ValueError("Each case must declare requiredTools")
    return cases


def prepare(api, case, timeout):
    # A fresh workspace means trial campaigns never compete with earlier cases.
    workspace = api.java("POST", "/workspaces", json={"name": "eval-" + case["id"] + "-" + uuid.uuid4().hex[:6]})["id"]
    context = {"workspaceId": workspace, "knowledgeBaseIds": [], "documents": []}
    if case.get("setup") in {"trial", "trial_future"}:
        catalog = api.java("GET", f"/workspaces/{workspace}/trials/catalog")
        course, campus = catalog["courses"][0], catalog["campuses"][0]
        now = datetime.now(timezone.utc)
        starts = now + timedelta(hours=1) if case["setup"] == "trial_future" else now - timedelta(minutes=1)
        campaign = api.java("POST", f"/workspaces/{workspace}/trials/campaigns", json={
            "title": "评测免费试听-" + case["id"], "courseId": str(course["id"]),
            "schoolId": str(campus["id"]), "capacity": 3,
            "startsAt": starts.isoformat(), "endsAt": (now + timedelta(hours=2)).isoformat()})
        campaign = api.java("POST", f"/workspaces/{workspace}/trials/campaigns/{campaign['id']}/publish")
        context["campaignId"] = str(campaign["id"])
    if case.get("setup") == "rag":
        kb = api.runtime("POST", "/knowledge-bases", workspace=workspace, json={"name": "合成课程政策"})
        context["knowledgeBaseIds"] = [kb["id"]]
        for path in sorted((HERE / "corpus").glob("*.txt")):
            doc = api.runtime("POST", "/documents", workspace=workspace,
                              data={"knowledgeBaseId": kb["id"]},
                              files={"file": (path.name, path.read_bytes(), "text/plain")})
            context["documents"].append({"id": doc["id"], "title": doc["title"]})
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            documents = api.runtime("GET", "/documents", workspace=workspace,
                                    params={"knowledgeBaseId": kb["id"]})
            if len(documents) == len(context["documents"]) and all(doc["status"] == "READY" for doc in documents):
                break
            if any(doc["status"] == "FAILED" for doc in documents):
                raise RuntimeError("Synthetic corpus indexing failed")
            time.sleep(.3)
        else:
            raise TimeoutError("Synthetic corpus indexing timed out")
    return context


def evaluate_contract(case, final, events, business, approval=None, provider="bailian"):
    expected = case["oracle"]
    tools = [event["data"].get("name") for event in events if event["type"] == "tool.started"]
    completed = [event["data"] for event in events if event["type"] == "tool.completed"]
    checks = {"expectedState": final["status"] in expected["states"],
              "requiredTools": set(expected["requiredTools"]).issubset(tools),
              "forbiddenToolsAbsent": not set(expected.get("forbiddenTools", [])).intersection(tools)}
    if "approvalDecision" in case:
        checks["correctApprovalTool"] = (bool(approval and approval.get("toolName", "reserve_course") == expected["approvalTool"])
                                          or bool(not approval and expected.get("approvalOptional")))
        if approval:
            expected_decision = "REJECTED" if case["approvalDecision"] == "REJECTED" else ("APPROVED" if expected.get("approvalOptional") else "EXECUTED")
            checks["persistedApprovalDecision"] = approval["status"] == expected_decision
    if business["coverage"] == "mysql-read-only":
        checks["expectedSideEffects"] = business["sideEffectCount"] == expected["sideEffects"]
        checks["noDuplicateActions"] = business["duplicateActionRows"] == 0
        checks["noUnapprovedSideEffects"] = business["unapprovedSideEffects"] == 0
        checks["reservationExists"] = business["missingReservationRows"] == 0
        if expected.get("trialStatus"):
            checks["trialBusinessState"] = len(business["trials"]) == 1 and business["trials"][0]["status"] == expected["trialStatus"]
            if expected["trialStatus"] == "SUCCEEDED":
                checks["trialHasRealOrder"] = bool(business["trials"] and business["trials"][0]["order_id"])
            else:
                checks["trialHasNoOrder"] = all(not value["order_id"] for value in business["trials"])
    else:
        checks["databaseOracleMissing"] = None
    answer = final.get("answer", "")
    clarification = " ".join(str(event["data"].get("question", "")) for event in events
                             if event["type"] == "input.required")
    reported_error = str((final.get("error") or {}).get("message", ""))
    displayed_text = " ".join((answer, clarification, reported_error))
    for index, alternatives in enumerate(expected.get("answerContainsAnyGroups", [])):
        checks[f"userVisibleFactGroup{index + 1}"] = any(phrase in displayed_text for phrase in alternatives)
    for fact in expected.get("answerContains", []):
        checks["answerContains:" + fact] = fact in answer
    if expected.get("citationTitle"):
        checks["expectedDocumentCited"] = any(citation["title"] == expected["citationTitle"] for citation in final.get("citations", []))
        checks["citationMarkerPresent"] = bool(re.search(r"\[E\d+\]", answer))
    tool_errors = [value for value in completed if isinstance(value.get("result"), dict) and value["result"].get("error")]
    business_fields, explicit_fields = reservation_field_checks(case, business, approval)
    score_business_fields = provider == "bailian" and business["coverage"] == "mysql-read-only"
    if score_business_fields:
        checks.update({"business:" + key: value for key, value in business_fields.items()})
    return {"checks": checks, "contractPassed": all(value is not False for value in checks.values()),
            "expectedBusinessFields": explicit_fields, "businessFieldChecks": business_fields,
            "businessFieldsScoreIncluded": score_business_fields,
            "businessFieldsPassed": all(business_fields.values()) if business_fields else None,
            "databaseOracleComplete": business["coverage"] == "mysql-read-only",
            "toolNames": tools, "toolCallCount": len(tools), "toolErrorCount": len(tool_errors),
            "toolCacheHits": sum(value.get("cached", False) for value in completed),
            "semanticAnswerCorrectness": None,
            "graderVersion": GRADER_VERSION,
            "grader": "graph states + tool policy + business DB + explicit student/contact and approved draft consistency + fact assertions; fixture excludes semantic field scoring; free-form semantics require manual review"}


def run_case(api, proxy, oracle, case, timeout, provider):
    row = {"caseId": case["id"], "category": case["category"], "split": case["split"],
           "startedAt": stamp(), "provider": provider, "taskQualityScore": None,
           "expectedFacts": case.get("expectedFacts", []), "expectedOracle": case["oracle"]}
    run, workspace = None, None
    try:
        context = prepare(api, case, timeout)
        workspace = context["workspaceId"]
        prompt = case["prompt"].replace("{campaignId}", context.get("campaignId", ""))
        started = time.monotonic()
        run, _ = api.submit(prompt, workspace=workspace, knowledge=context["knowledgeBaseIds"])
        row.update({"runId": run, "workspaceId": workspace, "prompt": prompt, "setup": context})
        status = api.wait_run(run, TERMINAL | {"WAITING_INPUT", "WAITING_APPROVAL"}, workspace, timeout)
        approval = None
        if status["status"] == "WAITING_APPROVAL" and case.get("approvalDecision"):
            approval = api.approval(run, workspace)
            row["beforeApprovalOracle"] = oracle.inspect(run)
            row["approval"] = approval
            api.decide(approval, case["approvalDecision"], workspace)
            status = api.wait_run(run, TERMINAL | {"WAITING_INPUT"}, workspace, timeout)
            approval = api.approval(run, workspace)
            row["approvalFinal"] = approval
        row["latencyMs"] = round((time.monotonic() - started) * 1000, 3)
        # Allow asynchronous trial projection to settle; terminal outcome is queried,
        # not inferred from HTTP202 or from runtime SUCCEEDED.
        business = oracle.inspect(run)
        if case["oracle"].get("trialStatus") and oracle.enabled:
            end = time.monotonic() + min(timeout, 40)
            while time.monotonic() < end and any(value["status"] in {"PENDING", "RESERVED"} for value in business["trials"]):
                time.sleep(.25)
                business = oracle.inspect(run)
        events = api.events(run)
        row.update({"finalRun": status, "businessOracle": business, "events": events,
                    **evaluate_contract(case, status, events, business, approval, provider)})
        if row.get("beforeApprovalOracle", {}).get("sideEffectCount", 0):
            row["checks"]["zeroBeforeApproval"] = False
            row["contractPassed"] = False
        row["knownChatCostCny"] = status.get("usage", {}).get("estimatedCostCny")
        row["embeddingCostCny"] = None
        row["costScope"] = "chat estimate only; corpus/query embedding usage is not currently accounted by production runtime"
        row["taskQualityScore"] = row["contractPassed"] if provider == "bailian" and oracle.enabled else None
        if status["status"] not in TERMINAL:
            api.java("POST", f"/runs/{run}/cancel", json={"workspaceId": workspace})
            api.wait_run(run, {"CANCELLED"}, workspace, timeout)
    except Exception as exc:
        row.update({"contractPassed": False, "error": safe_error(exc)})
        if provider == "bailian" and oracle.enabled:
            row["taskQualityScore"] = False
        if run:
            try:
                row["failureRunSnapshot"] = api.runtime("GET", "/runs/" + run, workspace=workspace)
                row["knownChatCostCny"] = row["failureRunSnapshot"].get("usage", {}).get("estimatedCostCny")
                row["businessOracle"] = oracle.inspect(run)
                api.java("POST", f"/runs/{run}/cancel", json={"workspaceId": workspace})
                api.wait_run(run, TERMINAL, workspace, timeout)
            except Exception as cleanup:
                row["cleanupError"] = safe_error(cleanup)
        else:
            row["knownChatCostCny"] = 0
    if run:
        row["javaToolCalls"] = proxy.for_run(run)
    row["finishedAt"] = stamp()
    return row


def summarize(rows, provider):
    scored = [row for row in rows if row.get("taskQualityScore") is not None]
    known = [row["knownChatCostCny"] for row in rows if row.get("knownChatCostCny") is not None]
    return {"provider": provider, "attempted": len(rows),
            "graderVersion": GRADER_VERSION,
            "engineeringContractPassed": sum(row["contractPassed"] for row in rows),
            "scoredRealModelTasks": len(scored),
            "ruleTaskPassRate": sum(row["taskQualityScore"] for row in scored) / len(scored) if scored else None,
            "semanticCorrectness": None, "categories": dict(Counter(row["category"] for row in rows)),
            "latencyP95Ms": percentile([row["latencyMs"] for row in rows if "latencyMs" in row]),
            "knownChatCostCny": sum(known), "unknownChatCostCases": len(rows) - len(known),
            "fullCostCny": None, "totalToolCalls": sum(row.get("toolCallCount", 0) for row in rows),
            "note": "Fixture has no model-quality score. Rule success does not grade unrestricted answer semantics. Latency excludes corpus preparation and includes automatic approval roundtrip."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=HERE / "agent_cases.jsonl")
    parser.add_argument("--provider", choices=["fixture", "bailian"], default="fixture")
    parser.add_argument("--allow-real-model", action="store_true")
    parser.add_argument("--max-real-cases", type=int, default=5)
    parser.add_argument("--max-estimated-chat-cost-cny", "--max-chat-cost-cny", dest="max_chat_cost_cny", type=float, default=5.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--case-ids", nargs="+")
    parser.add_argument("--split", choices=["development", "holdout"])
    parser.add_argument("--fixture-smoke", action="store_true")
    parser.add_argument("--timeout", type=int, default=150)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-db-oracle", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    cases = load_cases(args.dataset)
    if args.case_ids:
        selected = set(args.case_ids)
        if selected - {case["id"] for case in cases}:
            parser.error("Unknown case ID")
        cases = [case for case in cases if case["id"] in selected]
    if args.split:
        cases = [case for case in cases if case["split"] == args.split]
    if args.fixture_smoke:
        if args.provider != "fixture":
            parser.error("--fixture-smoke only applies to fixture")
        cases = [case for case in cases if case.get("fixtureSmokeEligible")]
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be positive")
        cases = cases[:args.limit]
    if args.validate_only:
        print(json.dumps({"validCases": len(cases), "categories": dict(Counter(case["category"] for case in cases)),
                          "datasetSha256": dataset_hash(args.dataset)}, ensure_ascii=False))
        return 0
    if not cases:
        parser.error("No cases selected")
    if args.provider == "bailian" and (not args.allow_real_model or args.limit is None or len(cases) > args.max_real_cases):
        parser.error("Real calls require --allow-real-model, explicit --limit, and selected count <= --max-real-cases")
    if args.max_chat_cost_cny <= 0:
        parser.error("--max-chat-cost-cny must be positive")
    if (args.output / "cases.jsonl").exists():
        parser.error("Use a new output directory")
    config, oracle = Config.environment(), BusinessOracle()
    if args.require_db_oracle and not oracle.enabled:
        parser.error("MySQL oracle is required")
    write_json(args.output / "manifest.json", manifest(config, args.provider, {
        "datasetSha256": dataset_hash(args.dataset), "caseIds": [case["id"] for case in cases],
        "grader": "structured full-agent contract", "graderVersion": GRADER_VERSION, "databaseOracle": oracle.enabled,
        "chatBudgetCny": args.max_chat_cost_cny,
        "budgetBoundary": "known chat estimate checked between cases; reserves configured per-run allowance; vendor billing can differ and single requests can overshoot"}))
    proxy, api = RealJavaProxy(config.java), Services(config)
    process = RuntimeProcess(config, args.output, proxy.url, args.provider)
    rows = []
    stopped_reason = None
    per_run_allowance = float(os.getenv("AGENT_COST_LIMIT_CNY", "0.10"))
    try:
        process.start(4)
        for case in cases:
            if args.provider == "bailian":
                if any(row.get("knownChatCostCny") is None for row in rows):
                    stopped_reason = "unknown_chat_usage"
                    break
                if sum(row.get("knownChatCostCny", 0) for row in rows) + per_run_allowance > args.max_chat_cost_cny:
                    stopped_reason = "insufficient_remaining_chat_budget"
                    break
            row = run_case(api, proxy, oracle, case, args.timeout, args.provider)
            rows.append(row)
            append_json(args.output / "cases.jsonl", row)
            write_json(args.output / "summary.json", summarize(rows, args.provider))
            print(f"{case['id']}: {'PASS' if row['contractPassed'] else 'FAIL'}", flush=True)
        write_json(args.output / "summary.json", {**summarize(rows, args.provider),
                   "selectedCases": len(cases), "notExecutedCaseIds": [case["id"] for case in cases[len(rows):]],
                   "budgetStoppedReason": stopped_reason, "chatBudgetCny": args.max_chat_cost_cny})
    finally:
        process.kill()
        proxy.close()
        api.close()
        oracle.close()
    return 0 if not stopped_reason and all(row["contractPassed"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
