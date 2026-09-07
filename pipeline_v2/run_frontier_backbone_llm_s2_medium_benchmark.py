from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_PARENT = Path(__file__).resolve().parent.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from pipeline_v2.config import llm_s2_medium_benchmark_flow_config
from pipeline_v2.pipeline_runner import run_flow


FLOW_CONFIG = llm_s2_medium_benchmark_flow_config()
FLOW_NAME = FLOW_CONFIG.flow_name
BENCHMARK_PROTOCOLS = FLOW_CONFIG.protocol_specs


def main() -> None:
    run_flow(FLOW_CONFIG)


if __name__ == "__main__":
    main()
