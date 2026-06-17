import json
from dataclasses import dataclass, field

from caelestia.utils.dots.packages import DEFAULT_AUR_HELPER
from caelestia.utils.io import warn
from caelestia.utils.paths import atomic_dump, dots_state_path


@dataclass
class DotsState:
    # The AUR helper selected selected at install time
    aur_helper: str = "paru"

    # The git rev of currently applied dots version
    applied_rev: str | None = None

    # The currently enabled components
    enabled_components: list[str] = field(default_factory=list)

    # Previously installed packages/local packages
    packages: list[str] = field(default_factory=list)
    local_packages: dict[str, list[str]] = field(default_factory=dict)

    # Files placed by the last deploy. Only files, not directories
    # Maps dest -> src
    deployed_files: dict[str, str] = field(default_factory=dict)

    @staticmethod
    def load() -> "DotsState":
        try:
            data = json.loads(dots_state_path.read_text())
        except FileNotFoundError:
            return DotsState()
        except json.JSONDecodeError:
            warn("failed to parse current dots state.")
            return DotsState()

        return DotsState(
            aur_helper=data.get("aur_helper", DEFAULT_AUR_HELPER),
            applied_rev=data.get("applied_rev"),
            enabled_components=data.get("enabled_components", []),
            packages=data.get("packages", []),
            local_packages=data.get("local_packages", {}),
            deployed_files=data.get("deployed_files", {}),
        )

    def save(self) -> None:
        atomic_dump(
            dots_state_path,
            {
                "aur_helper": self.aur_helper,
                "applied_rev": self.applied_rev,
                "enabled_components": self.enabled_components,
                "packages": self.packages,
                "local_packages": self.local_packages,
                "deployed_files": self.deployed_files,
            },
        )
