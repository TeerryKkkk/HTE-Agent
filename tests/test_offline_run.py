import csv
from dataclasses import replace
import socket

from pipeline_v2.config import backbone_flow_config
from pipeline_v2.pipeline_runner import run_flow


def test_offline_run_respects_budget_and_never_connects(tmp_path, monkeypatch):
    def fail_network(*args, **kwargs):
        raise AssertionError("The offline baseline attempted a network connection")

    monkeypatch.setattr(socket.socket, "connect", fail_network)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    flow = replace(backbone_flow_config(), smoke_output_dir=tmp_path / "smoke",
                   validation_output_dir=tmp_path / "validation")
    run_flow(flow)
    with (tmp_path / "smoke/experiments.csv").open(encoding="utf-8-sig") as handle:
        experiments = list(csv.DictReader(handle))
    assert len(experiments) == flow.constants.initial_batch_size + flow.constants.stage2_batch_size
    candidate_ids = [row["candidate_id"] for row in experiments]
    assert len(set(candidate_ids)) == len(candidate_ids)
    with (tmp_path / "smoke/run_summary.csv").open(encoding="utf-8-sig") as handle:
        summary = next(csv.DictReader(handle))
    assert float(summary["best_observed_yield"]) <= float(summary["oracle_best_yield"])
