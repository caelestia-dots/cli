import glob
import os
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from string import Template
from typing import Any

_XDG_DEFAULTS = {
    "XDG_CONFIG_HOME": str(Path.home() / ".config"),
    "XDG_DATA_HOME": str(Path.home() / ".local/share"),
    "XDG_STATE_HOME": str(Path.home() / ".local/state"),
    "XDG_CACHE_HOME": str(Path.home() / ".cache"),
}
_GLOB_MAGIC = re.compile(r"[*?[]")
_LOCAL_PREFIX = "local:"


class ManifestError(Exception):
    """Raised when manifest.toml is malformed."""


class ComponentError(Exception):
    """Raised when component flags are invalid or contradictory."""


def expand(text: str) -> Path:
    """Expand $VAR/${VAR} env vars (with XDG defaults) and ~ in a path."""

    env = {**_XDG_DEFAULTS, **os.environ}
    return Path(Template(text).safe_substitute(env)).expanduser()


def expand_dests(dest: str) -> list[Path]:
    """Expand globs within a dest path.

    Globs from the start until the segment with the last glob so subdirs are
    created if they didn't exist previously.
    """

    expanded = expand(dest)
    if not _GLOB_MAGIC.search(str(expanded)):
        return [expanded]

    parts = expanded.parts
    glob_idx = max(i for i, part in enumerate(parts) if _GLOB_MAGIC.search(part))
    pattern = str(Path(*parts[: glob_idx + 1]))
    tail = parts[glob_idx + 1 :]
    return [Path(match, *tail) for match in sorted(glob.glob(pattern))]


@dataclass(frozen=True)
class ManifestEntry:
    src: str
    dest: str


@dataclass(frozen=True)
class ManifestComponent:
    name: str
    default: bool = False
    packages: list[str] = field(default_factory=list)
    entries: list[ManifestEntry] = field(default_factory=list)
    post_install: list[str] = field(default_factory=list)
    post_update: list[str] = field(default_factory=list)


@dataclass
class _ManifestData:
    enabled_comps: list[str] = field(default_factory=list)
    disabled_comps: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Manifest:
    components: dict[str, ManifestComponent] = field(default_factory=dict)
    packages: list[str] = field(default_factory=list)
    post_install: list[str] = field(default_factory=list)
    post_update: list[str] = field(default_factory=list)
    _data: _ManifestData = field(default_factory=_ManifestData, init=False, repr=False)

    @property
    def enabled_components(self) -> list[str]:
        return self._data.enabled_comps

    @property
    def disabled_components(self) -> list[str]:
        return self._data.disabled_comps

    @staticmethod
    def parse(text: str) -> "Manifest":
        try:
            raw = tomllib.loads(text)
        except tomllib.TOMLDecodeError as e:
            raise ManifestError(f"invalid TOML: {e}") from e

        hooks = raw.get("hooks", {})
        post_install = _validate_str_list(hooks.get("post_install", []), "hooks.post_install")
        post_update = _validate_str_list(hooks.get("post_update", []), "hooks.post_update")

        packages = _validate_str_list(raw.get("packages", []), "packages")

        components = {}
        for comp in raw.get("components", []):
            parsed = _parse_component(comp)
            components[parsed.name] = parsed

        return Manifest(
            components=components,
            packages=packages,
            post_install=post_install,
            post_update=post_update,
        )

    def resolve_components(
        self,
        enable: list[str] | None = None,
        disable: list[str] | None = None,
    ) -> None:
        """Resolves enabled/disabled components. This MUST be called before calling any other method."""

        enable_set = set(enable or [])
        disable_set = set(disable or [])
        known = set(self.components)

        for name in enable_set | disable_set:
            if name not in known:
                raise ComponentError(f"unknown component: {name}")

        conflict = enable_set & disable_set
        if conflict:
            raise ComponentError(f"component(s) both enabled and disabled: {', '.join(sorted(conflict))}")

        enabled = {name for name, comp in self.components.items() if comp.default}
        enabled |= enable_set
        enabled -= disable_set

        self._data.enabled_comps.clear()
        self._data.disabled_comps.clear()
        for name in self.components:
            if name in enabled:
                self._data.enabled_comps.append(name)
            else:
                self._data.disabled_comps.append(name)

    def enabled_entries(self) -> list[ManifestEntry]:
        """The entries of every enabled component."""

        entries: list[ManifestEntry] = []
        for name in self._data.enabled_comps:
            entries.extend(self.components[name].entries)
        return entries

    def enabled_hooks(self, kind: str) -> list[str]:
        """Global + enabled components' hooks of the given kind."""

        hooks = list(getattr(self, kind))
        for name in self._data.enabled_comps:
            hooks.extend(getattr(self.components[name], kind))
        return hooks

    def enabled_packages(self) -> list[str]:
        """Repo/AUR packages to install."""
        return [p for p in self._all_packages() if not p.startswith(_LOCAL_PREFIX)]

    def enabled_local_packages(self) -> list[str]:
        """Local PKGBUILD dirs to build.

        Local packages are determined by a local: prefix and are
        relative dirs instead of package names.
        """
        return [p[len(_LOCAL_PREFIX) :] for p in self._all_packages() if p.startswith(_LOCAL_PREFIX)]

    def _all_packages(self) -> list[str]:
        """The manifest's top-level packages plus enabled components'."""
        return list(set(self.packages) | set(p for c in self._data.enabled_comps for p in self.components[c].packages))


def _require_key(d: dict[str, Any], key: str, ctx: str) -> Any:
    if key not in d:
        raise ManifestError(f"{ctx}: missing required key '{key}'")
    return d[key]


def _validate_str_list(value: Any, ctx: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ManifestError(f"{ctx}: expected a list of strings")
    return value


def _parse_entry(d: Any) -> ManifestEntry:
    if not isinstance(d, dict):
        raise ManifestError("entry: expected a table")
    return ManifestEntry(src=_require_key(d, "src", "entry"), dest=_require_key(d, "dest", "entry"))


def _parse_component(d: dict[str, Any]) -> ManifestComponent:
    name = _require_key(d, "name", "component")
    return ManifestComponent(
        name=name,
        default=bool(d.get("default", False)),
        packages=_validate_str_list(d.get("packages", []), f"component '{name}' packages"),
        entries=[_parse_entry(e) for e in d.get("entries", [])],
        post_install=_validate_str_list(d.get("post_install", []), f"component '{name}' post_install"),
        post_update=_validate_str_list(d.get("post_update", []), f"component '{name}' post_update"),
    )
