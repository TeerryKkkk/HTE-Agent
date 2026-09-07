import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from statistics import mean

import pytest

from hte_agent.config import ASSISTANCE_SEMANTICS, MODES, PROTOCOLS, PROJECT_ROOT, SharedRunConstants, create_flow_config
from hte_agent.pipeline_runner import run_flow
from hte_agent.reporting import build_comparison
from hte_agent.shared.io_utils import read_csv_rows
from hte_agent.stage1.retrieval_initializer import _normalize_ligand_label


def test_historical_snapshot_schema_hashes_and_readme_agree():
    root = PROJECT_ROOT / "benchmarks"
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["schema_version"] == 1 and metadata["snapshot_is_current_semantics"] is False
    assert metadata["legacy_assistance"]["stage1_deterministic_backstop_merged_on_success"] is True
    assert metadata["historical_model_provenance"]["status"] == "recovered_from_matching_local_run_artifacts"
    tables = {}
    for name, artifact in metadata["data"]["files"].items():
        path = PROJECT_ROOT / "data/buchwald_hartwig" / name
        assert hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest() == artifact["sha256"]
        assert len(read_csv_rows(path)) == artifact["rows"]
    for name, artifact in metadata["artifacts"].items():
        path = root / name
        assert hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest() == artifact["sha256"]
        rows = read_csv_rows(path)
        assert len(rows) == artifact["rows"] and all(None not in row for row in rows)
        tables[name] = rows
    comparison = tables["final_benchmark_comparison.csv"]
    protocol = metadata["protocol"]
    assert len(comparison) == 72
    assert {int(r["seed"]) for r in comparison} == set(protocol["seeds"])
    assert {r["profile_name"] for r in comparison} == set(protocol["profiles"])
    assert {int(r["total_experiments"]) for r in comparison} == {120}
    assert len({(r["flow_name"], r["seed"], r["profile_name"]) for r in comparison}) == 72
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    for row in tables["final_benchmark_overall_summary.csv"]:
        source = [r for r in comparison if r["flow_name"] == row["flow_name"]]
        assert len(source) == int(row["num_profile_runs"]) == 18
        values = []
        for key, raw_key in (("best_observed_mean", "best_observed_yield"), ("oracle_regret_mean", "regret_to_oracle"),
                             ("cumulative_regret_mean", "cumulative_regret"), ("stage2_mean", "stage2_batch_mean_yield")):
            assert float(row[key]) == pytest.approx(round(mean(float(r[raw_key]) for r in source), 4), abs=.0001)
            values.append(f"{float(row[key]):.4f}")
        assert " | ".join(values) in readme


def test_ligand_normalization_preserves_specific_brettphos_labels():
    assert _normalize_ligand_label("tBuBrettPhos Pd G3")[0] == "tBuBrettPhos"
    assert _normalize_ligand_label("t-BuBrettPhos")[0] == "tBuBrettPhos"
    assert _normalize_ligand_label("BrettPhos")[0] == "BrettPhos"


def test_all_cli_modes_preserve_protocol_budgets_and_aggregation_provenance(tmp_path):
    for mode in MODES:
        for protocol in PROTOCOLS:
            flow = create_flow_config(mode, protocol, results_root=tmp_path)
            assert len(flow.protocol_specs) == {"smoke": 1, "validation": 1, "quick-benchmark": 2, "benchmark": 6}[protocol]
            assert all(p.initial_batch_size + p.stage2_rounds * p.stage2_batch_size == (120 if "benchmark" in protocol else 108) for p in flow.protocol_specs)
    constants = replace(SharedRunConstants(), candidate_pool_cap=160, initial_batch_size=12, stage2_batch_size=6)
    for mode in ("baseline", "llm-s1s2"):
        run_flow(create_flow_config(mode, results_root=tmp_path, constants=constants))
    output = build_comparison(tmp_path, "smoke")
    comparison = read_csv_rows(output / "comparison.csv")
    assert {r["llm_evaluation_status"] for r in comparison} == {"not_requested", "unavailable"}
    assert all(r["assistance_semantics"] == ASSISTANCE_SEMANTICS for r in comparison)
    llm_row = next(r for r in comparison if r["flow_name"] == "llm_s1s2")
    assert not any(call["invoked"] for call in json.loads(llm_row["llm_s2_gate_calls"]))
    assert "llm_s2_summary_models" in read_csv_rows(output / "overall_summary.csv")[0]
    # Unknown legacy summaries must not silently enter current comparisons.
    summary_path = tmp_path / "baseline_smoke/run_summary.csv"
    summary_path.write_text(summary_path.read_text().replace(ASSISTANCE_SEMANTICS, "legacy"))
    with pytest.raises(ValueError, match="legacy"):
        build_comparison(tmp_path, "smoke")
