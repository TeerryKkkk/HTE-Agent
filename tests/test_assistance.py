import copy
from dataclasses import replace
import io
import json
from types import SimpleNamespace
import urllib.error
import urllib.request

import pytest

from hte_agent.config import SharedRunConstants, create_flow_config
from hte_agent.llm_modules.client import OpenRouterJSONClient
from hte_agent.llm_modules.precedent_critic import run_precedent_critic, calibrate_precedent_critique
from hte_agent.llm_modules.result_summarizer import summarize_results
from hte_agent.llm_modules.topk_rerank_gate import rerank_topk, _sanitize_actions
from hte_agent.pipeline_runner import _apply_gate, _select_gated_batch, run_flow
from hte_agent.shared.candidate_schema import Candidate, LLMS1Critique, LLMS2Summary, Stage2Score
from hte_agent.shared.io_utils import read_csv_rows
from hte_agent.shared.offline_simulator import simulate_observation
from hte_agent.shared.task_representation import build_task_representation
from hte_agent.stage1.retrieval_initializer import RetrievedSupportBundle
from hte_agent.stage1.rule_filter import apply_rule_filter
from hte_agent.stage1.transfer_scorer import score_candidates
from hte_agent.stage2.optimizer_loop import MixedBatchOptimizer
from hte_agent.stage2.result_aggregator import aggregate_round


C = SharedRunConstants()
TASK = build_task_representation(C.supported_profiles[0], C.reaction_family, C.objective)
SUPPORT = RetrievedSupportBundle((), (), (), ())


def response(parsed, model="synthetic/model", status="success", invoked=True):
    return {"status": status, "invoked": invoked, "model": model, "parsed": parsed, "error": "", "attempts": []}


def fake_client(parsed):
    return SimpleNamespace(complete_json=lambda *args: response(parsed))


def candidates(n):
    return [Candidate(str(i), {"ligand_label": f"L{i // 10}", "base_label": f"B{i // 10}",
                              "solvent_label": "Dioxane", "temperature_c": 78.0,
                              "catalyst_mol_pct": .15, "base_equiv": 2.0}) for i in range(n)]


def test_missing_and_failed_assistance_is_decision_identical_to_baseline(tmp_path, monkeypatch):
    constants = replace(C, candidate_pool_cap=240, initial_batch_size=16, stage2_batch_size=6, stage2_rounds=2)
    baseline = create_flow_config(results_root=tmp_path, constants=constants)
    run_flow(baseline)
    baseline_dir = baseline.protocols[0][1]
    reference = read_csv_rows(baseline_dir / "experiments.csv")
    fingerprint = lambda rows: [{k: v for k, v in r.items() if k != "flow_name"} for r in rows]
    # The real client must not touch urlopen with a missing credential. The autouse fixture denies it.
    for failure in ("missing_api_key", "request_failed", "invalid_response"):
        if failure == "request_failed":
            monkeypatch.setattr(OpenRouterJSONClient, "complete_json", lambda *args: response(
                {"bonuses": {"ligand_label": {"BrettPhos": .15}}}, model="", status="request_failed"))
        elif failure == "invalid_response":
            monkeypatch.setattr(OpenRouterJSONClient, "complete_json", lambda *args: response(
                {"bonuses": [], "promising_patterns": 123, "candidate_actions": {}}))
        for mode in ("llm-s1", "llm-s2", "llm-s1s2"):
            flow = create_flow_config(mode, results_root=tmp_path, constants=constants)
            summary = run_flow(flow)[0]
            output = flow.protocols[0][1]
            assert summary["optimization_succeeded"]
            assert not summary["llm_evaluation_succeeded"]
            assert summary["llm_evaluation_status"] == ("unavailable" if failure == "missing_api_key" else "failed")
            for prefix, enabled in (("llm_s1", flow.llm_stage1_enabled), ("llm_s2", flow.llm_stage2_enabled)):
                assert summary[f"{prefix}_status"] == (failure if enabled else "disabled")
                assert summary[f"{prefix}_invoked"] == (enabled and failure != "missing_api_key")
                assert summary[f"{prefix}_batch_delta_count"] == 0
            assert summary["llm_s1_nonzero_candidate_count"] == summary["llm_s1_rejected_count"] == summary["llm_s2_action_count"] == 0
            assert fingerprint(read_csv_rows(output / "experiments.csv")) == fingerprint(reference)
            for path in output.rglob("*topk_after_gate.csv"):
                assert all(float(row["llm_delta"]) == 0 for row in read_csv_rows(path))


def test_successful_stage1_uses_only_model_actions_and_respects_bounds():
    pool = candidates(400)
    # A one-candidate combination is permitted; the broad rejection is not.
    pool[-1].parameters["base_label"] = "rare"
    parsed = {"bonuses": {"ligand_label": {"L0": 10, "L1": 10, "unknown": 9, "L2": float("nan")}},
              "penalties": {"ligand_label": {"L0": 10, "L3": 10}, "base_label": {"B3": 10}},
              "reject_values": {"base_label": ["B4"]},
              "reject_combinations": [{"ligand_label": "L39", "base_label": "rare"},
                                      {"ligand_label": "L38"}, {}]}
    raw = run_precedent_critic(TASK, SUPPORT, fake_client(parsed), C)
    calibrated = calibrate_precedent_critique(raw, pool, C)
    assert calibrated.status == "success_calibrated" and calibrated.model == "synthetic/model"
    assert calibrated.bonuses == {"ligand_label": {"L1": .03}}
    assert calibrated.diagnostics["contradiction_resolution_count"] == 1
    assert calibrated.penalties.get("ligand_label", {}).get("L0", 0) <= C.s1_contradiction_penalty_magnitude
    contradiction_only = run_precedent_critic(TASK, SUPPORT, fake_client({
        "bonuses": {"ligand_label": {"L0": 10}}, "penalties": {"ligand_label": {"L0": 10}}}), C)
    assert calibrate_precedent_critique(contradiction_only, pool, C).penalties["ligand_label"]["L0"] == .02
    scored = score_candidates(TASK, pool, SUPPORT, calibrated)
    feasible, rejected = apply_rule_filter(TASK, scored, calibrated)
    assert all(-C.s1_max_penalty_magnitude <= c.llm_stage1_delta <= C.s1_max_bonus_magnitude for c in scored)
    changed = {c.candidate_id for c in scored if c.llm_stage1_delta} | {c.candidate_id for c in rejected}
    assert len(changed) <= int(len(pool) * C.s1_max_modified_fraction)
    assert len(rejected) == 1 <= int(len(pool) * C.s1_max_reject_fraction)
    assert not calibrated.reject_values
    empty = run_precedent_critic(TASK, SUPPORT, fake_client({}), C)
    assert empty.invoked and empty.status == "success" and not empty.bonuses and not empty.penalties
    zero = calibrate_precedent_critique(raw, pool, replace(C, s1_max_modified_fraction=0))
    assert not zero.bonuses and not zero.penalties and not zero.reject_combinations
    tight = calibrate_precedent_critique(raw, pool, replace(C, s1_max_modified_fraction=.05))
    assert tight.diagnostics["estimated_modified_candidate_count"] <= 20
    assert tight.diagnostics["dropped_for_budget_count"] > 0
    failed = replace(raw, status="request_failed")
    assert not any(c.llm_stage1_delta for c in score_candidates(TASK, candidates(400), SUPPORT, failed))


def test_stage2_successful_actions_are_sparse_valid_and_bounded_after_selection():
    pool = candidates(18)
    scores = [Stage2Score(c.candidate_id, .70, .20, .70-i*.001, .5, 0., 0., final_score=.70-i*.001)
              for i, c in enumerate(pool)]
    actions = [{"candidate_id": str(i), "decision": "downweight", "confidence": .99, "score_delta": -99}
               for i in range(8)]
    actions += [{"candidate_id": "17", "decision": "prefer_for_exploration", "confidence": 1., "score_delta": 99},
                {"candidate_id": "unknown", "decision": "reject", "confidence": 1.},
                {"candidate_id": "8", "decision": "reject", "confidence": float("nan")},
                {"candidate_id": "9", "decision": "downweight", "confidence": 1., "score_delta": "nonsense"},
                {"candidate_id": "10", "decision": "downweight", "score_delta": -.04}, 123, None]
    aggregate = aggregate_round([], 1)
    summary = LLMS2Summary(True, "success", "synthetic/summary", "")
    gate = rerank_topk(TASK, aggregate, summary, pool, scores, 12, fake_client({"candidate_actions": actions}), C)
    assert gate.status == "success" and gate.invoked
    assert 0 < gate.diagnostics["nonkeep_action_count"] <= C.s2_max_nonkeep_actions
    assert gate.diagnostics["action_counts"].get("downweight", 0) <= C.s2_max_downweight_actions
    assert {a["candidate_id"] for a in gate.candidate_actions} == {c.candidate_id for c in pool}
    assert all(-.08 <= a["score_delta"] <= .02 for a in gate.candidate_actions)
    assert gate.diagnostics["filtered_counts"]["invalid_candidate"] == 1
    hindered_task = build_task_representation(C.supported_profiles[1], C.reaction_family, C.objective)
    hindered = rerank_topk(hindered_task, aggregate, summary, pool, scores, 12,
                          fake_client({"candidate_actions": actions}), C)
    assert hindered.diagnostics["nonkeep_action_count"] <= 2
    assert hindered.diagnostics["action_counts"].get("downweight", 0) <= 1
    assert not hindered.diagnostics["action_counts"].get("reject", 0)
    assert not hindered.diagnostics["action_counts"].get("prefer_for_exploration", 0)
    assert hindered.diagnostics["max_final_batch_changes"] == 1
    optimizer = MixedBatchOptimizer(C.acquisition_beta, C.diversity_weight, C.prior_blend)
    by_id = {c.candidate_id: c for c in pool}
    before, _ = optimizer.select_diverse_batch(scores, by_id, 12)
    _, after, _ = _select_gated_batch(optimizer, scores, by_id, 12, gate)
    assert len({c.candidate_id for c in after} - {c.candidate_id for c in before}) <= C.s2_max_final_batch_changes
    # Force the real post-diversity guard to exercise full rollback.
    gate.diagnostics["max_final_batch_changes"] = 0
    adjusted, after, _ = _select_gated_batch(optimizer, scores, by_id, 12, gate)
    assert [c.candidate_id for c in after] == [c.candidate_id for c in before]
    assert gate.diagnostics["actual_batch_limit_reverted"]
    assert all(s.llm_delta == 0 for s in adjusted)
    assert not gate.diagnostics["nonkeep_action_count"]


def test_empty_and_failed_summaries_never_inject_heuristic_recommendations():
    aggregate = aggregate_round([simulate_observation(c, TASK, 7, 1) for c in candidates(10)], 1)
    for payload in ({}, {"summary": "Model summary", "promising_patterns": []}):
        result = summarize_results(TASK, aggregate, fake_client(payload))
        assert result.status == "success"
        assert result.promising_patterns == result.risky_patterns == result.exploration_axes == []


def test_client_attempt_provenance_and_error_redaction(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "private-test-value")
    calls = []
    def transport(request, **kwargs):
        calls.append(json.loads(request.data)["model"])
        if len(calls) == 1:
            raise urllib.error.HTTPError(request.full_url, 429, "private-test-value", {}, io.BytesIO(b"private-test-value"))
        return io.BytesIO(json.dumps({"model": "provider/resolved-model", "choices": [{"message": {"content": "{}"}}]}).encode())
    monkeypatch.setattr(urllib.request, "urlopen", transport)
    client = OpenRouterJSONClient("https://example.invalid", 1, ("model/first", "model/second"))
    result = client.complete_json("test", "test", {})
    assert result["invoked"] and result["status"] == "success"
    assert result["model"] == "model/second" and result["response_model"] == "provider/resolved-model"
    assert [a["status"] for a in result["attempts"]] == ["request_failed", "success"]
    assert "private-test-value" not in json.dumps(result)
    def failure(request, **kwargs):
        raise urllib.error.HTTPError(request.full_url, 429, "private-test-value", {}, io.BytesIO(b"private-test-value"))
    monkeypatch.setattr(urllib.request, "urlopen", failure)
    failed = client.complete_json("test", "test", {})
    assert failed["status"] == "request_failed" and failed["error"] == "HTTP 429"
    assert "private-test-value" not in json.dumps(failed)
    assert not replace(client, model_fallbacks=()).complete_json("test", "test", {})["invoked"]


def test_future_summary_preserves_partial_rounds_and_models(tmp_path, monkeypatch):
    calls = []
    def completion(self, role, system_prompt, payload):
        calls.append(role)
        if role == "result_summarizer" and calls.count(role) == 1:
            return response({}, status="request_failed", model="")
        return response({}, model=f"synthetic/{role}")
    monkeypatch.setattr(OpenRouterJSONClient, "complete_json", completion)
    constants = replace(C, candidate_pool_cap=160, initial_batch_size=12, stage2_batch_size=6, stage2_rounds=2)
    row = run_flow(create_flow_config("llm-s1s2", constants=constants, results_root=tmp_path))[0]
    assert row["optimization_succeeded"] and not row["llm_evaluation_succeeded"]
    assert row["llm_evaluation_status"] == "partial" and row["llm_s2_status"] == "mixed"
    assert row["llm_s2_gate_succeeded"] and not row["llm_s2_summary_succeeded"]
    assert row["llm_s2_summary_model"] == "synthetic/result_summarizer"
    assert row["llm_s2_gate_model"] == "synthetic/topk_rerank_gate"
    assert row["llm_s2_model"] == ""
    assert [c["status"] for c in json.loads(row["llm_s2_summary_calls"])] == ["request_failed", "success"]
    assert row["llm_s2_call_count"] == 4


def test_simulator_is_deterministic_by_profile_seed_and_candidate():
    for profile in C.supported_profiles:
        task = build_task_representation(profile, C.reaction_family, C.objective)
        for candidate in candidates(8):
            first = simulate_observation(candidate, task, 17, 1)
            assert first == simulate_observation(copy.deepcopy(candidate), task, 17, 1)
            later = simulate_observation(candidate, task, 17, 3)
            assert first.yield_value == later.yield_value and first.raw_measurements == later.raw_measurements
