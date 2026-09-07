import csv
import socket

from hte_agent.config import create_flow_config
from hte_agent.pipeline_runner import run_flow


def test_offline_run_respects_budget_and_never_connects(tmp_path, monkeypatch):
    def fail_network(*args, **kwargs):
        raise AssertionError("The offline baseline attempted a network connection")

    monkeypatch.setattr(socket.socket, "connect", fail_network)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    flow = create_flow_config(results_root=tmp_path)
    run_flow(flow)
    with (tmp_path / "baseline_smoke/experiments.csv").open(encoding="utf-8-sig") as handle:
        experiments = list(csv.DictReader(handle))
    assert len(experiments) == flow.constants.initial_batch_size + flow.constants.stage2_batch_size
    candidate_ids = [row["candidate_id"] for row in experiments]
    assert len(set(candidate_ids)) == len(candidate_ids)
    with (tmp_path / "baseline_smoke/run_summary.csv").open(encoding="utf-8-sig") as handle:
        summary = next(csv.DictReader(handle))
    assert float(summary["best_observed_yield"]) <= float(summary["oracle_best_yield"])
    assert summary["optimization_succeeded"] == "True"
    assert summary["llm_evaluation_status"] == "not_requested"
    assert {p.name for p in tmp_path.iterdir()} == {"baseline_smoke"}
