import os
from pathlib import Path

config_dir = Path(os.getenv("XDG_CONFIG_HOME", Path.home() / ".config"))
data_dir = Path(os.getenv("XDG_DATA_HOME", Path.home() / ".local/share"))
state_dir = Path(os.getenv("XDG_STATE_HOME", Path.home() / ".local/state"))

c_config_dir = config_dir / "caelestia"
c_data_dir = data_dir / "caelestia"
c_state_dir = state_dir / "caelestia"

cli_data_dir = Path(__file__).parent.parent / "data"
templates_dir = cli_data_dir / "templates"

scheme_path = c_state_dir / "scheme.json"
scheme_data_path = cli_data_dir / "schemes"
