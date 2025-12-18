"""Definition of ML models configuration."""

import os
import tomllib
from typing import Literal, get_args
from .utils.constants import DotDict, ProjectPaths

ExistingTraces = Literal[
    "expert", "chatgpt", "claude", "gpt-oss", "llama", "mistral"
]

# TODO: Should we be able to override this ?
cfg_path = os.path.join(os.path.dirname(__file__), "../config/config.toml")

# ------------ Static configuration  ------------
with open(cfg_path, "rb") as f:
    raw_cfg = tomllib.load(f)

_generic = raw_cfg.get("generic")
seed = _generic["seed"]
# Default trace type, given by config file.
trace_type = _generic["trace_type"]
mysql_info = DotDict(raw_cfg.get("mysql", {}))


def update_location_mysqlfiles(
    type: ExistingTraces,
) -> None:
    """Update the location of the MySQL server socket, and the datadir where we find
    the gaur.log file given a trace type.

    Args:
        type (ExistingTraces): _description_
    """
    global trace_type

    if type not in get_args(ExistingTraces):
        raise ValueError(f"Unknown trace type: {type}")
    trace_type = type

    ppths.set_trace_type(trace_type)
    base_path = mysql_info.prefix + "mysqld-" + trace_type + "/"
    mysql_info.socket_path = base_path + "socket"
    mysql_info.datadir_path = base_path + "datadir/"


# ------------ Dynamic configuration  ------------
# Root path, where outpur/ or logs/ folders will be created.
base_path = os.path.join(os.path.dirname(__file__), "../")

# Bootstrap a custom object path. Access this object from any file using:
# > import config
# > m = MyObject(log_path=config.ppths.logs_path)
# Source: https://discuss.python.org/t/global-variables-shared-across-modules/16833/2
ppths = ProjectPaths(base_path)

# Bootstrap the location of mysql files, it can be changed later using the same func
update_location_mysqlfiles(trace_type)
