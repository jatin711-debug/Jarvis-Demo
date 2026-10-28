"""
JARVIS entry point: ``python -m jarvis`` or the ``jarvis`` console script.

Exit codes: 0 ok, 1 startup failed (or --doctor found a problem), 2 invalid
configuration, 130 interrupted.
"""

import os
import sys

# Quiet native-library chatter. These must be set before the libraries import.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
if sys.platform == "win32":
    # Child processes started by tools emit UTF-8, which is how their output is decoded.
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import warnings

# Dependency deprecation noise is not actionable for users. Other warnings are
# captured by the logging system (file only unless --verbose).
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import argparse
import asyncio

import yaml
from pydantic import ValidationError

from jarvis import __version__
from jarvis.core.orchestrator import Orchestrator
from jarvis.utils.config import expand_path, load_config, resolve_config_path
from jarvis.utils.logger import configure_stdio, get_logger, setup_logging

logger = get_logger(__name__)

EXIT_OK = 0
EXIT_STARTUP_FAILED = 1
EXIT_BAD_CONFIG = 2
EXIT_INTERRUPTED = 130


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="jarvis", description="JARVIS - local voice AI assistant")
    parser.add_argument("--config", type=str, default=None,
                        help="path to config.yaml (default: auto-detect)")
    parser.add_argument("--text", type=str, default=None,
                        help="run one text turn without microphone, wake word, or speech output")
    parser.add_argument("--list-tools", action="store_true",
                        help="print every tool by surface, effect and confirmation, then exit")
    parser.add_argument("--doctor", action="store_true",
                        help="check Ollama, the model, audio devices and packages, then exit")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="show DEBUG diagnostics on the console (the log file always has them)")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    args = parser.parse_args(argv)
    if args.text is not None and not args.text.strip():
        parser.error("--text must be non-empty")
    return args


async def main(argv: list[str] | None = None) -> int:
    """Boot Jarvis; returns the process exit code."""
    args = parse_args(argv)

    try:
        config = load_config(args.config)
    except (OSError, yaml.YAMLError, ValidationError) as exc:
        # Logging is not configured yet, so report directly.
        print(f"jarvis: cannot load configuration: {exc}", file=sys.stderr)
        return EXIT_BAD_CONFIG

    if args.list_tools:
        from jarvis.agent.inventory import render_tool_table

        print(render_tool_table(config), end="")
        return EXIT_OK

    if args.doctor:
        from jarvis import doctor

        checks = await doctor.run_checks(config, resolve_config_path(args.config))
        print(doctor.render(checks), end="")
        return doctor.exit_code(checks)

    setup_logging(
        level="DEBUG" if args.verbose else config.system.log_level,
        log_dir=str(expand_path(config.system.log_dir)),
        file_format=config.system.log_format,
    )
    logger.info("starting jarvis version=%s config=%s", __version__,
                resolve_config_path(args.config) or "defaults")

    orchestrator = Orchestrator(config)
    exit_code = EXIT_OK
    try:
        try:
            await orchestrator.startup(voice_enabled=args.text is None)
        except RuntimeError as exc:
            # Preflight failures (Ollama down, model not pulled, ...) fail fast
            # with an actionable message instead of booting into broken turns.
            logger.error("startup failed: %s", exc)
            return EXIT_STARTUP_FAILED

        if args.text is None:
            await orchestrator.run()
        else:
            print(await orchestrator.ask(args.text))
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("interrupted by user")
        exit_code = EXIT_INTERRUPTED
    finally:
        await orchestrator.shutdown()
        logger.info("goodbye")
    return exit_code


def cli() -> None:
    """Console-script entry point (``jarvis`` command)."""
    configure_stdio()
    try:
        code = asyncio.run(main())
    except KeyboardInterrupt:
        code = EXIT_INTERRUPTED
    sys.exit(code)


if __name__ == "__main__":
    cli()
