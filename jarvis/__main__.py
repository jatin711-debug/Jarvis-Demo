"""JARVIS entry point — python -m jarvis"""

# ── Suppress ALL noisy library warnings before any imports ──
import warnings
import os
import sys

# Nuclear option: silence everything (DeprecationWarning, FutureWarning, UserWarning)
warnings.simplefilter("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTHONWARNINGS"] = "ignore"
# Also suppress via sys.warnoptions so child threads respect it
if not sys.warnoptions:
    os.environ["PYTHONWARNINGS"] = "ignore"

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
