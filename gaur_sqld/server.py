"""Auto-provisioning of the GAUR-instrumented MySQL server."""

import logging
import os
import shutil
import socket
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

GAUR_APPS_REPO = "https://github.com/gquetel/gaur-instrumented-apps"
# Override to point the builder at a different gaur-instrumented-apps checkout
# (e.g. one on a feature branch) without touching the auto-cloned default.
GAUR_APPS_DIR = Path(
    os.environ.get("GAUR_APPS_DIR")
    or Path.home() / ".local" / "share" / "gaur-sqld" / "gaur-instrumented-apps"
)


class GaurServerError(RuntimeError):
    """Raised when the GAUR MySQL server cannot be started."""


class GaurServerManager:
    """Manages the lifecycle of a GAUR-instrumented MySQL server.

    The Nix run-server script computes its own base path from hostname -s and
    the configured prefix; it daemonizes mysqld itself.  This class simply
    invokes the script and checks that the socket appeared afterwards.

    Workflow:
    1. Check whether a server is already reachable at the configured socket.
    2. If not, attempt to build via Nix and run the provisioning script.
    3. If Nix is unavailable, raise GaurServerError with actionable instructions.
    """

    def __init__(self, trace_type: str):
        from gaur_sqld import config as cfg

        self.trace_type = trace_type
        self.socket_path = cfg.mysql_info.socket_path

    def _server_is_running(self) -> bool:
        """Return True if the MySQL socket file exists and accepts connections."""
        if not Path(self.socket_path).exists():
            return False
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                sock.settimeout(1)
                sock.connect(self.socket_path)
            return True
        except OSError:
            return False

    def _log_output(self, stdout: str, stderr: str) -> None:
        for line in stdout.splitlines():
            if line.strip():
                logger.debug("[stdout] %s", line)
        for line in stderr.splitlines():
            if line.strip():
                logger.debug("[stderr] %s", line)

    def _ensure_apps_cloned(self) -> None:
        if GAUR_APPS_DIR.exists():
            logger.info(f"gaur-instrumented-apps already present at {GAUR_APPS_DIR}")
            return
        logger.info(f"Cloning {GAUR_APPS_REPO} → {GAUR_APPS_DIR}")
        GAUR_APPS_DIR.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            ["git", "clone", "--depth", "1", GAUR_APPS_REPO, str(GAUR_APPS_DIR)],
            capture_output=True,
            text=True,
        )
        self._log_output(result.stdout, result.stderr)
        if result.returncode != 0:
            raise GaurServerError(
                f"Failed to clone gaur-instrumented-apps:\n{result.stderr}"
            )

    def _nix_build(self) -> Path:
        app_dir = GAUR_APPS_DIR / "apps" / f"mysql-{self.trace_type}"
        if not app_dir.exists():
            raise GaurServerError(
                f"No Nix app found for trace type '{self.trace_type}' at {app_dir}"
            )
        logger.info(f"Running nix-build in {app_dir}")
        result = subprocess.run(
            ["nix-build", "default.nix"],
            cwd=str(app_dir),
            capture_output=True,
            text=True,
        )
        self._log_output(result.stdout, result.stderr)
        if result.returncode != 0:
            raise GaurServerError(
                f"nix-build failed for trace type '{self.trace_type}':\n{result.stderr}"
            )
        return app_dir / "result" / "bin" / "run-server"

    def _start_server(self, run_server_bin: Path) -> None:
        """Run the Nix provisioning script.

        The script uses 'hostname -s' to build its own base path, initialises
        the datadir, daemonizes mysqld, and waits ~2 s before returning.
        We do not pass paths to it — they are baked in at Nix build time.
        """
        logger.info(f"Starting GAUR MySQL server ({self.trace_type})…")
        result = subprocess.run(
            [str(run_server_bin)],
            capture_output=True,
            text=True,
        )
        self._log_output(result.stdout, result.stderr)
        if result.returncode != 0:
            raise GaurServerError(
                f"run-server failed for trace type '{self.trace_type}':\n{result.stderr}"
            )
        logger.info(result.stdout.strip())

        if not self._server_is_running():
            raise GaurServerError(
                f"run-server exited successfully but socket not found at "
                f"{self.socket_path}.\n"
                "Check that the prefix in your config matches the path used by "
                "the Nix script (~/tmp/<hostname>/mysqld-<trace_type>/)."
            )
        logger.info(f"GAUR MySQL server ready at socket {self.socket_path}")

    def ensure_running(self) -> None:
        """Ensure the GAUR MySQL server for this trace type is running.

        Raises:
            GaurServerError: If the server cannot be started.
        """
        if self._server_is_running():
            logger.info(
                f"GAUR MySQL server already running (socket: {self.socket_path})"
            )
            return

        if not shutil.which("nix-build"):
            raise GaurServerError(
                f"No GAUR MySQL server found at {self.socket_path}.\n"
                "Install Nix (https://nixos.org/download) to auto-provision, or "
                "manually start a server and set the prefix via:\n"
                "  gaur_sqld.configure(prefix='~/tmp/')"
            )

        self._ensure_apps_cloned()
        run_server_bin = self._nix_build()
        self._start_server(run_server_bin)

    def stop(self) -> None:
        """Kill the daemonized mysqld for this trace type.

        Mirrors the pkill the Nix script itself uses at startup.
        """
        from gaur_sqld import config as cfg

        hostname = subprocess.check_output(["hostname", "-s"], text=True).strip()
        base = str(Path(cfg.mysql_info.prefix) / hostname / f"mysqld-{self.trace_type}")
        result = subprocess.run(["pkill", "-f", base])
        if result.returncode == 0:
            logger.info(f"Stopped GAUR MySQL server ({self.trace_type})")
        else:
            logger.warning(
                f"No running GAUR MySQL process matched filter: {base}"
            )


def ensure_server(trace_type: str | None = None) -> None:
    """Ensure a GAUR MySQL server is running for the given trace type.

    Args:
        trace_type: One of 'expert', 'chatgpt', 'claude', 'gpt-oss', 'llama',
            'mistral'. Defaults to the value in the current configuration.

    Raises:
        GaurServerError: If the server cannot be found or started.
    """
    from gaur_sqld import config as cfg

    GaurServerManager(trace_type or cfg.trace_type).ensure_running()


def stop_server(trace_type: str | None = None) -> None:
    """Stop a GAUR MySQL server that was started via the Nix provisioning script.

    Args:
        trace_type: Trace type of the server to stop.
            If None, stops servers for all known trace types.
    """
    from gaur_sqld.config import ExistingTraces
    from typing import get_args

    targets = [trace_type] if trace_type else list(get_args(ExistingTraces))
    for tt in targets:
        GaurServerManager(tt).stop()
