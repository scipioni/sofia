"""Entrypoint for s2s.service.

Forwards to the livekit-agents CLI, so all its subcommands still work:

    sofia-s2s start          # production worker (what the container runs)
    sofia-s2s dev            # local, with hot reload
    sofia-s2s console        # talk to the agent in your terminal, no LiveKit room
    sofia-s2s download-files # fetch turn-detector / VAD weights
"""

from __future__ import annotations

from livekit.agents import cli

from sofia_galileo.core.logging import configure_logging
from sofia_galileo.s2s.agent import worker_options
from sofia_galileo.s2s.config import S2SSettings


def main() -> None:
    settings = S2SSettings()
    # The livekit CLI owns the root handler and the log level (`--log-level`);
    # we only route our own structured events into it.
    configure_logging("s2s", settings.log_level, attach_handler=False)
    cli.run_app(worker_options())


if __name__ == "__main__":
    main()
