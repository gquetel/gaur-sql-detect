"""Definition of ML models configuration."""

import os
import socket
import tomllib
from pathlib import Path
from typing import Literal, get_args
from .utils.constants import DotDict, ProjectPaths

ExistingTraces = Literal[
    "expert", "chatgpt", "claude", "gpt-oss", "llama", "mistral"
]


def _load_config() -> dict:
    """Load bundled default configuration."""
    bundled_config = Path(__file__).parent / "config" / "config.toml"
    with open(bundled_config, "rb") as f:
        return tomllib.load(f)


# ------------ Static configuration  ------------
raw_cfg = _load_config()

_generic = raw_cfg.get("generic", {})
seed = _generic.get("seed", 7)
# Default trace type, given by config file.
trace_type = _generic.get("trace_type", "expert")
mysql_info = DotDict(raw_cfg.get("mysql", {}))

# Expand ~ in prefix
if "prefix" in mysql_info:
    mysql_info.prefix = str(Path(mysql_info.prefix).expanduser())


def update_location_mysqlfiles(
    type: ExistingTraces,
) -> None:
    """Update the location of the MySQL server socket, and the datadir where we find
    the gaur.log file given a trace type.

    Args:
        type (ExistingTraces): The trace type to use.
    """
    global trace_type

    if type not in get_args(ExistingTraces):
        raise ValueError(f"Unknown trace type: {type}")
    trace_type = type

    ppths.set_trace_type(trace_type)
    hostname = socket.gethostname()
    base = Path(mysql_info.prefix) / hostname / f"mysqld-{trace_type}"
    mysql_info.socket_path = str(base / "socket")
    mysql_info.datadir_path = str(base / "datadir") + "/"


def configure(
    prefix: str | None = None,
    trace_type: str | None = None,
    seed: int | None = None,
) -> None:
    """Programmatically override configuration values.

    Args:
        prefix: Path prefix for MySQL server files
            (e.g. '~/.local/share/gaur-sqld/servers/'). Tilde is expanded.
        trace_type: Default trace type to use
            ('expert', 'chatgpt', 'claude', 'gpt-oss', 'llama', 'mistral').
        seed: Random seed for reproducibility.
    """
    import gaur_sqld.config as _cfg

    if prefix is not None:
        _cfg.mysql_info.prefix = str(Path(prefix).expanduser())

    effective_trace = trace_type if trace_type is not None else _cfg.trace_type
    if prefix is not None or trace_type is not None:
        update_location_mysqlfiles(effective_trace)

    if seed is not None:
        _cfg.seed = seed


def configure_from_file(path: str) -> None:
    """Load configuration from a TOML file, replacing all current settings.

    Args:
        path: Path to a TOML configuration file.
    """
    import gaur_sqld.config as _cfg

    with open(path, "rb") as f:
        raw = tomllib.load(f)

    _cfg.raw_cfg = raw

    generic = raw.get("generic", {})
    _cfg.seed = generic.get("seed", 7)
    _cfg.trace_type = generic.get("trace_type", "expert")

    _cfg.mysql_info = DotDict(raw.get("mysql", {}))
    if "prefix" in _cfg.mysql_info:
        _cfg.mysql_info.prefix = str(Path(_cfg.mysql_info.prefix).expanduser())

    update_location_mysqlfiles(_cfg.trace_type)


# ------------ Dynamic configuration  ------------
# Root path, where output/ or logs/ folders will be created.
# Use the working directory of the calling script so that cache/output/logs
# are always written to a writable location (avoids issues with read-only
# install paths such as the Nix store).
base_path = os.getcwd()

# Bootstrap a custom object path. Access this object from any file using:
# > import gaur_sqld.config as config
# > m = MyObject(log_path=config.ppths.logs_path)
ppths = ProjectPaths(base_path)

# Bootstrap the location of mysql files, it can be changed later using the same func
update_location_mysqlfiles(trace_type)
