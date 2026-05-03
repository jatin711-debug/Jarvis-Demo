"""JARVIS entry point — python -m jarvis"""

# ── Suppress ALL noisy library warnings BEFORE any imports ──
import os
import sys

# Set env vars FIRST - before any imports
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "0"
if not sys.warnoptions:
    os.environ["PYTHONWARNINGS"] = "ignore"

# NOW suppress warnings and logging
import warnings
warnings.simplefilter("ignore")

import logging

# Suppress all noisy loggers by attaching null handler
_null_handler = logging.NullHandler()
for _name in ["tensorflow", "NumExpr", "torch", "transformers",
              "httpx", "httpcore", "requests"]:
    _logger = logging.getLogger(_name)
    _logger.addHandler(_null_handler)
    _logger.setLevel(50)

import argparse
import asyncio
import sys

from jarvis.utils.logger import get_logger
from jarvis.utils.config import load_config
from jarvis.core.orchestrator import Orchestrator

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="JARVIS — Local Voice AI Assistant")
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config.yaml (default: auto-detect)"
    )
    return parser.parse_args()


async def main() -> None:
    """Boot Jarvis voice assistant."""
    args = parse_args()

    logger.info("🚀 Starting JARVIS v0.2.0 ...")

    config = load_config(args.config)
    logger.info("✅ Configuration loaded")

    orchestrator = Orchestrator(config)
    await orchestrator.startup()

    try:
        await orchestrator.run()
    except KeyboardInterrupt:
        logger.info("🛑 Shutdown requested by user")
    finally:
        await orchestrator.shutdown()
        logger.info("👋 JARVIS stopped. Goodbye!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)
