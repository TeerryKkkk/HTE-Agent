"""Run a simulator protocol: python -m hte_agent.run --help."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path

from hte_agent.config import MODES, PROTOCOLS, RESULTS_ROOT, create_flow_config
from hte_agent.offline import offline_execution
from hte_agent.pipeline_runner import run_flow
from hte_agent.reporting import build_comparison


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline reaction-condition simulation with optional bounded model assistance.")
    parser.add_argument("--mode", choices=MODES, default="baseline")
    parser.add_argument("--protocol", choices=PROTOCOLS, default="smoke")
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--offline", action="store_true", help="Clear the model credential temporarily and deny network access.")
    parser.add_argument("--aggregate", action="store_true", help="After execution, explicitly aggregate available runs of this protocol.")
    args = parser.parse_args(argv)
    config = create_flow_config(args.mode, args.protocol, results_root=args.results_root)
    with offline_execution() if args.offline else nullcontext():
        summaries = run_flow(config)
        if args.aggregate:
            build_comparison(args.results_root, args.protocol)
    for row in summaries:
        print(f"{row['protocol']} / {row['profile_name']}: optimization_succeeded={row['optimization_succeeded']}; "
              f"llm_evaluation={row['llm_evaluation_status']}; s1={row['llm_s1_status']}; s2={row['llm_s2_status']}")
    if any(not row["optimization_succeeded"] for row in summaries):
        return 1
    if any(row["llm_assistance_requested"] and not row["llm_evaluation_succeeded"] for row in summaries):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
