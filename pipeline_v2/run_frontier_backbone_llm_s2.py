from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_PARENT = Path(__file__).resolve().parent.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from pipeline_v2.config import llm_s2_flow_config
from pipeline_v2.pipeline_runner import run_flow


FLOW_CONFIG = llm_s2_flow_config()
FLOW_NAME = FLOW_CONFIG.flow_name
SMOKE_OUTPUT_DIR = FLOW_CONFIG.smoke_output_dir
VALIDATION_OUTPUT_DIR = FLOW_CONFIG.validation_output_dir
INITIAL_BATCH_SIZE = FLOW_CONFIG.constants.initial_batch_size
STAGE2_BATCH_SIZE = FLOW_CONFIG.constants.stage2_batch_size
STAGE2_ROUNDS = FLOW_CONFIG.constants.stage2_rounds


def main() -> None:
    run_flow(FLOW_CONFIG)


if __name__ == "__main__":
    main()
