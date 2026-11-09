#!/usr/bin/env python3
"""Run one PLCBench-style S1--S4 SITL episode with GPT-5.5 via OpenRouter.

This is intentionally a thin locked-model wrapper around the canonical runner:
the scenario runtime, prompt, tools, sandbox, action limits, and evaluator are
identical to the Qwen condition.  It reads only the gitignored
PLCBENCH_OPENROUTER_API_KEY2 credential.
"""

from __future__ import annotations

import sys

from run_plcbench_sitl_agent import main


if __name__ == "__main__":
    # Appending the locked values ensures callers cannot accidentally change
    # this condition's model or credential source via duplicate CLI options.
    main([*sys.argv[1:], "--model", "openai/gpt-5.5", "--api-key-env",
          "PLCBENCH_OPENROUTER_API_KEY2"])
