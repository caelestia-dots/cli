import os
import shutil
import subprocess
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path

from caelestia.utils.io import fatal, info, warn

DEFAULT_AUR_HELPER = "paru"
AUR_HELPERS = DEFAULT_AUR_HELPER, "yay"


class PackageError(Exception):
    """Raised when a package operation (install/remove/build/update) fails."""


def _try_run(cmd: list[str], error_msg: str, **kwargs) -> None:
    """Run a subprocess, raising `PackageError` if it fails."""

    try:
        subprocess.run(cmd, check=True, **kwargs)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        raise PackageError(error_msg) from e


def _read_srcinfo(directory: Path) -> dict[str, list[str]]:
    """Run `makepkg --printsrcinfo` in `directory`, grouping each key to its list of values."""

    try:
        srcinfo = subprocess.check_output(["makepkg", "--printsrcinfo"], cwd=directory, text=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        raise PackageError(f"failed to read package metadata in {directory}") from e

    fields: dict[str, list[str]] = {}
    for line in srcinfo.splitlines():
        key, sep, value = line.partition("=")
        if not sep:
            continue
        fields.setdefault(key.strip(), []).append(value.strip())
    return fields


def _srcinfo_version(fields: dict[str, list[str]]) -> str | None:
    """Build the `[epoch:]pkgver-pkgrel` version string from parsed .SRCINFO fields, or None if absent."""

    pkgver = next(iter(fields.get("pkgver", [])), None)
    pkgrel = next(iter(fields.get("pkgrel", [])), None)
    if pkgver is None or pkgrel is None:
        return None

    version = f"{pkgver}-{pkgrel}"
    epoch = next(iter(fields.get("epoch", [])), None)
    return f"{epoch}:{version}" if epoch else version


def _vercmp(a: str, b: str) -> int:
    """Use pacman's `vercmp` to compare to package versions."""

    try:
        return int(subprocess.check_output(["vercmp", a, b], text=True).strip())
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError) as e:
        warn(f"vercmp failed, assuming equal: {e}")
        return 0  # Don't rebuild when unable to check version


def _install_aur_helper(helper: str, noconfirm: bool = False) -> None:
    pacman_cmd = ["sudo", "pacman", "-S", "--needed", "git", "base-devel"]
    if noconfirm:
        pacman_cmd.append("--noconfirm")
    _try_run(pacman_cmd, "failed to install AUR helper build dependencies")

    repo_url = f"https://aur.archlinux.org/{helper}.git"
    with tempfile.TemporaryDirectory() as repo_dir:
        _try_run(["git", "clone", repo_url, repo_dir], f"failed to clone {helper} from the AUR")

        makepkg_cmd = ["makepkg", "-si"]
        if noconfirm:
            makepkg_cmd.append("--noconfirm")
        _try_run(makepkg_cmd, f"failed to build and install {helper}", cwd=repo_dir)

    try:
        if helper == "yay":
            subprocess.run(["yay", "-Y", "--gendb"], check=True)
            subprocess.run(["yay", "-Y", "--devel", "--save"], check=True)
        elif helper == "paru":
            subprocess.run(["paru", "--gendb"], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        warn(f"failed to run AUR helper post install actions: {e}")


class PackageInstaller(ABC):
    @staticmethod
    def get(helper: str | None = None, noconfirm: bool = False) -> "PackageInstaller":
        """Pick a package installer: the requested/detected AUR helper on Arch, else a no-op."""

        # Not on Arch, can't install packages
        if shutil.which("pacman") is None:
            return NoopInstaller()

        # Explicitly given
        if helper:
            if not shutil.which(helper):
                if helper not in AUR_HELPERS:
                    fatal(f"given AUR helper {helper} is not installed and is unable to be installed automatically.")

                info(f"Given AUR helper not installed. Installing {helper}...")
                _install_aur_helper(helper, noconfirm)
            return ArchInstaller(helper, noconfirm)

        # Not given, find installed one
        for candidate in AUR_HELPERS:
            if shutil.which(candidate):
                return ArchInstaller(candidate, noconfirm)

        info(f"No AUR helper found. Installing {DEFAULT_AUR_HELPER}...")
        _install_aur_helper(DEFAULT_AUR_HELPER, noconfirm)
        return ArchInstaller(DEFAULT_AUR_HELPER, noconfirm)

    # --- Abstract methods ---

    @abstractmethod
    def install(self, packages: list[str]) -> None: ...

    @abstractmethod
    def remove(self, packages: list[str]) -> None: ...

    @abstractmethod
    def build_install(self, directory: Path) -> list[str]:
        """Build and install the PKGBUILD in `directory`, returning the installed package names."""

    @abstractmethod
    def installed_version(self, package: str) -> str | None:
        """Return the installed version of `package`, or None if it is not installed."""

    def is_installed(self, package: str) -> bool:
        return self.installed_version(package) is not None

    @abstractmethod
    def needs_rebuild(self, directory: Path, packages: list[str]) -> bool:
        """Whether the PKGBUILD in `directory` would build a version differing from the installed `packages`."""

    @abstractmethod
    def system_update(self) -> None: ...


class NoopInstaller(PackageInstaller):
    """Used off Arch, where the dots' packages are not available via pacman/AUR."""

    def install(self, packages: list[str]) -> None:
        if packages:
            info(f"Skipping package install (not on Arch): {', '.join(packages)}")

    def remove(self, packages: list[str]) -> None:
        if packages:
            info(f"Skipping package removal (not on Arch): {', '.join(packages)}")

    def build_install(self, directory: Path) -> list[str]:
        info(f"Skipping local package build (not on Arch): {directory}")
        return []

    def installed_version(self, package: str) -> str | None:
        return None

    def needs_rebuild(self, directory: Path, packages: list[str]) -> bool:
        return False

    def system_update(self) -> None:
        info("Skipping system update (not on Arch)")


class ArchInstaller(PackageInstaller):
    def __init__(self, helper: str, noconfirm: bool = False) -> None:
        self.helper = helper
        self.flags = ["--noconfirm"] if noconfirm else []

    def install(self, packages: list[str], explicit: bool = True) -> None:
        if not packages:
            return

        cmd = [self.helper, "-S", "--needed", *self.flags]
        if not explicit:
            cmd.append("--asdeps")  # Set install reason to dep (does not affect already installed packages)
        _try_run(cmd + packages, f"failed to install packages: {', '.join(packages)}")

        # Force install reason to explicit install
        if explicit:
            # `-D` only accepts real installed names, so resolve any virtual/`provides` names (e.g. awk -> gawk)
            resolved = [self._installed_name(pkg) for pkg in packages]
            try:
                subprocess.run([self.helper, "-D", "--asexplicit", *self.flags, *resolved], check=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                warn(f"failed to mark packages as explicitly installed: {', '.join(resolved)}")

    def remove(self, packages: list[str]) -> None:
        if not packages:
            return
        _try_run([self.helper, "-Rns", *self.flags, *packages], f"failed to remove packages: {', '.join(packages)}")

    def build_install(self, directory: Path) -> list[str]:
        fields = _read_srcinfo(directory)
        names = fields.get("pkgname", [])
        depends = fields.get("depends", [])

        self.install(depends, explicit=False)

        # Stop makepkg from resetting sudo
        env = {**os.environ, "PACMAN_AUTH": "sudo"}
        # -f = force, -s = sync deps, -i = install
        _try_run(
            ["makepkg", "-fsi", *self.flags], f"failed to build local package in {directory}", cwd=directory, env=env
        )

        # Clean build artifacts
        for artifact in directory.glob("*.pkg.tar*"):
            try:
                artifact.unlink()
            except OSError as e:
                warn(f"failed to remove build artifact {artifact}: {e}")

        return names

    def _query(self, package: str) -> tuple[str, str] | None:
        """Return the installed (name, version) of `package`, resolving `provides` (e.g. awk -> gawk), or None."""

        result = subprocess.run(
            ["pacman", "-Q", package],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        if result.returncode != 0:
            return None
        # `pacman -Q` resolves provides and prints "<real name> <version>"
        parts = result.stdout.split()
        return (parts[0], parts[1]) if len(parts) >= 2 else None

    def _installed_name(self, package: str) -> str:
        """Resolve `package` to its real installed name (handles provides), falling back to the given name."""

        query = self._query(package)
        return query[0] if query else package

    def installed_version(self, package: str) -> str | None:
        query = self._query(package)
        return query[1] if query else None

    def needs_rebuild(self, directory: Path, packages: list[str]) -> bool:
        built = _srcinfo_version(_read_srcinfo(directory))
        if built is None:
            return False  # Can't determine the source version, leave as is

        # Rebuild when installed version < repo version
        # Don't rebuild packages that have been removed
        return any(
            (installed := self.installed_version(pkg)) is not None and _vercmp(built, installed) > 0 for pkg in packages
        )

    def system_update(self) -> None:
        _try_run([self.helper, "-Syu", *self.flags], "failed to perform system update")
