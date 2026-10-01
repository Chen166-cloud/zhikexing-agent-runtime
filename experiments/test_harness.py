"""Harness checks: synthetic loopback server only; no project services or paid APIs."""

import json
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx

from .common import BusinessOracle, Config, RealJavaProxy, percentile, safe_error
from .evaluate import evaluate_contract, explicit_reservation_fields, load_cases, main as eval_main, summarize as eval_summary
from .recovery import summarize as recovery_summary
from .rag import ranking_metrics, read_jsonl, validate as validate_rag, fuse


class ContractOracleTest(unittest.TestCase):
    def test_runtime_success_cannot_replace_committed_order(self):
        case = {"oracle": {"states": ["SUCCEEDED"], "requiredTools": [], "sideEffects": 1,
                           "trialStatus": "SUCCEEDED"}}
        business = {"coverage": "mysql-read-only", "sideEffectCount": 1,
                    "duplicateActionRows": 0, "unapprovedSideEffects": 0,
                    "missingReservationRows": 0, "trials": [{"status": "PENDING", "order_id": None}]}
        result = evaluate_contract(case, {"status": "SUCCEEDED"}, [], business)
        self.assertFalse(result["contractPassed"])
        self.assertFalse(result["checks"]["trialHasRealOrder"])

    def test_read_only_case_rejects_unapproved_write(self):
        case = {"oracle": {"states": ["SUCCEEDED"], "requiredTools": ["search_courses"],
                           "forbiddenTools": ["draft_reservation"], "sideEffects": 0}}
        events = [{"type": "tool.started", "data": {"name": name}} for name in ["search_courses", "draft_reservation"]]
        business = {"coverage": "mysql-read-only", "sideEffectCount": 1, "duplicateActionRows": 0,
                    "unapprovedSideEffects": 1, "missingReservationRows": 0, "trials": []}
        result = evaluate_contract(case, {"status": "SUCCEEDED"}, events, business)
        self.assertFalse(result["contractPassed"])
        self.assertFalse(result["checks"]["forbiddenToolsAbsent"])
        self.assertFalse(result["checks"]["noUnapprovedSideEffects"])

    def test_fixture_has_no_quality_score(self):
        result = eval_summary([{"caseId": "x", "category": "query", "contractPassed": True,
                                "taskQualityScore": None}], "fixture")
        self.assertIsNone(result["ruleTaskPassRate"])
        self.assertIsNone(result["fullCostCny"])

    def test_recovery_failures_remain_in_denominator(self):
        cases = [{"fault": "x", "passed": True, "faultInjected": True, "recoveryMs": 10, "startupMs": 2},
                 {"fault": "x", "passed": False, "faultInjected": True, "recoveryMs": None, "startupMs": None}]
        summary = recovery_summary(cases)["groups"]["x"]
        self.assertEqual(summary["passRate"], .5)
        self.assertEqual(summary["successfulRecoveryP95Ms"], 10)
        self.assertIsNone(summary["duplicateActionRows"])

    def test_dataset_has_explicit_side_effect_expectations(self):
        cases = load_cases(Path(__file__).with_name("agent_cases.jsonl"))
        self.assertEqual(len(cases), 50)
        self.assertTrue(all("sideEffects" in case["oracle"] for case in cases))
        future = [case for case in cases if case["category"] == "trial_future"]
        self.assertTrue(all(case["oracle"]["sideEffects"] == 0 for case in future))

    def test_real_model_gate_precedes_environment_or_service_access(self):
        with patch.dict(os.environ, {}, clear=True), patch("sys.argv", ["evaluate", "--provider", "bailian",
                                                                            "--output", "unused-test-output"]):
            with self.assertRaises(SystemExit) as failure:
                eval_main()
        self.assertEqual(failure.exception.code, 2)

    def test_errors_do_not_serialize_credentials(self):
        value = safe_error(ValueError("postgresql://secret:secret@host/private"))
        self.assertNotIn("secret", json.dumps(value))
        self.assertEqual(percentile([1, 2, 3, 4, 5]), 5)

    def test_rag_recall_requires_all_relevant_chunks(self):
        result = ranking_metrics(["a", "x", "b", "z", "q"], ["a", "b", "c"])
        self.assertEqual(result["recallAt5"], 2 / 3)
        self.assertEqual(result["reciprocalRankAt5"], 1)
        self.assertFalse(result["allEvidenceAt5"])
        self.assertEqual(fuse(["a", "b"], ["b", "c"])[0], "b")

    def test_rag_labels_and_holdout_are_consistent(self):
        root = Path(__file__).parent
        corpus, questions = read_jsonl(root / "rag_corpus.jsonl"), read_jsonl(root / "rag_questions.jsonl")
        validate_rag(corpus, questions)
        self.assertEqual((len(corpus), len(questions)), (100, 50))
        self.assertEqual(sum(q["split"] == "holdout" for q in questions), 30)

    def test_future_trial_can_explain_without_draft(self):
        case = next(value for value in load_cases(Path(__file__).with_name("agent_cases.jsonl"))
                    if value["id"] == "trial_future-01")
        business = {"coverage": "mysql-read-only", "sideEffectCount": 0, "duplicateActionRows": 0,
                    "unapprovedSideEffects": 0, "missingReservationRows": 0, "trials": []}
        result = evaluate_contract(case, {"status": "SUCCEEDED", "answer": "该活动尚未开始，现在不能领取名额。"}, [], business)
        self.assertTrue(result["contractPassed"])

    def test_missing_information_can_be_requested_in_completed_answer(self):
        case = next(value for value in load_cases(Path(__file__).with_name("agent_cases.jsonl"))
                    if value["id"] == "missing_information-01")
        business = {"coverage": "mysql-read-only", "sideEffectCount": 0, "duplicateActionRows": 0,
                    "unapprovedSideEffects": 0, "missingReservationRows": 0, "trials": []}
        result = evaluate_contract(case, {"status": "SUCCEEDED", "answer": "请提供您的姓名和联系方式。"}, [], business)
        self.assertTrue(result["contractPassed"])


class BusinessFieldOracleTest(unittest.TestCase):
    def setUp(self):
        self.case = next(value for value in load_cases(Path(__file__).with_name("agent_cases.jsonl"))
                         if value["id"] == "reservation_approve-02")
        self.args = {"studentName": "测试甲", "contactInfo": "13800000001", "courseName": "合成课程", "schoolName": "合成校区"}
        self.approval = {"status": "EXECUTED", "toolName": "reserve_course", "args": self.args}
        self.events = [{"type": "tool.started", "data": {"name": name}} for name in self.case["oracle"]["requiredTools"]]
        self.business = {"coverage": "mysql-read-only", "sideEffectCount": 1, "duplicateActionRows": 0,
                         "unapprovedSideEffects": 0, "missingReservationRows": 0, "trials": [],
                         "approvals": [{"tool_name": "reserve_course", "args": self.args}],
                         "reservations": [{"approval_args": self.args, "student_name": "测试甲",
                                            "contact_info": "13800000001", "course": "合成课程", "school": "合成校区"}]}

    def test_explicit_name_variants_and_contact(self):
        for prompt, expected in (("姓名为测试甲，电话13800000001。", "测试甲"),
                                 ("姓名是测试乙；手机13800000001", "测试乙"),
                                 ("我叫测试丙，电话13800000001", "测试丙"),
                                 ("帮实验学员预约课程，手机号13800000001", "实验学员"),
                                 ("请为测试丁准备预约，电话13800000001", "测试丁")):
            value = explicit_reservation_fields(prompt)
            self.assertEqual(value["studentName"], expected)
            self.assertEqual(value["contactInfo"], "13800000001")

    def test_mysql_empty_tuple_and_populated_list_results(self):
        oracle = object.__new__(BusinessOracle)
        oracle.enabled = True
        oracle.connection = MagicMock()
        cursor = oracle.connection.cursor.return_value.__enter__.return_value
        cursor.fetchall.side_effect = [(), (), [{"id": "a1", "tool_name": "reserve_course", "status": "PENDING", "args": '{"studentName":"测试甲"}', "result": None}]]
        result = oracle.inspect("r1")
        self.assertEqual(result["sideEffectCount"], 0)
        self.assertEqual(result["approvals"][0]["args"]["studentName"], "测试甲")

    def test_matching_business_fields_pass(self):
        result = evaluate_contract(self.case, {"status": "SUCCEEDED"}, self.events, self.business, self.approval)
        self.assertTrue(result["contractPassed"])
        self.assertTrue(result["businessFieldsPassed"])

    def test_wrong_name_contact_or_approved_course_fails_real_score(self):
        for key, value in (("student_name", "错误姓名"), ("contact_info", "13800000000"), ("course", "另一课程")):
            previous = self.business["reservations"][0][key]
            self.business["reservations"][0][key] = value
            real = evaluate_contract(self.case, {"status": "SUCCEEDED"}, self.events, self.business, self.approval)
            fixture = evaluate_contract(self.case, {"status": "SUCCEEDED"}, self.events, self.business, self.approval, "fixture")
            self.assertFalse(real["contractPassed"])
            self.assertFalse(real["businessFieldsPassed"])
            self.assertTrue(fixture["contractPassed"])
            self.assertFalse(fixture["businessFieldsScoreIncluded"])
            self.business["reservations"][0][key] = previous


class ResponseLossProxyTest(unittest.TestCase):
    def test_committed_response_is_withheld_without_fabricating_java_result(self):
        writes = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                writes.append(body)
                raw = json.dumps({"reservationId": "synthetic-test-only", "status": "SUCCEEDED"}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        proxy = RealJavaProxy(f"http://127.0.0.1:{server.server_port}")
        errors = []
        proxy.arm("r1")

        def request():
            try:
                httpx.post(proxy.url + "/internal/v1/tools/execute-reservation", timeout=5, trust_env=False,
                           headers={"X-Run-Id": "r1"}, json={"actionId": "a1"})
            except httpx.HTTPError as error:
                errors.append(type(error).__name__)

        caller = threading.Thread(target=request)
        caller.start()
        try:
            self.assertTrue(proxy.committed.wait(3))
            self.assertEqual(writes, [{"actionId": "a1"}])
            self.assertTrue(caller.is_alive())
            self.assertTrue(proxy.for_run("r1")[0]["responseWithheld"])
            proxy.release.set()
            caller.join(3)
            self.assertEqual(errors, ["RemoteProtocolError"])
            self.assertEqual(proxy.for_run("r1")[0]["result"]["reservationId"], "synthetic-test-only")
        finally:
            proxy.close()
            server.shutdown()
            server.server_close()
            thread.join(3)


if __name__ == "__main__":
    unittest.main()
