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
    def is_installed(self, package: str) -> bool: ...

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

    def is_installed(self, package: str) -> bool:
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
            try:
                subprocess.run([self.helper, "-D", "--asexplicit", *self.flags, *packages], check=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                warn(f"failed to mark packages as explicitly installed: {', '.join(packages)}")

    def remove(self, packages: list[str]) -> None:
        if not packages:
            return
        _try_run([self.helper, "-Rns", *self.flags, *packages], f"failed to remove packages: {', '.join(packages)}")

    def build_install(self, directory: Path) -> list[str]:
        try:
            srcinfo = subprocess.check_output(["makepkg", "--printsrcinfo"], cwd=directory, text=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raise PackageError(f"failed to read package metadata in {directory}") from e

        names = []
        depends = []
        for line in srcinfo.splitlines():
            key, sep, value = line.partition("=")
            if not sep:
                continue

            key = key.strip()
            if key == "pkgname":
                names.append(value.strip())
            elif key == "depends":
                depends.append(value.strip())

        self.install(depends, explicit=False)

        # Stop makepkg from resetting sudo
        env = {**os.environ, "PACMAN_AUTH": "sudo"}
        # -f = force, -s = sync deps, -i = install
        _try_run(
            ["makepkg", "-fsi", *self.flags], f"failed to build local package in {directory}", cwd=directory, env=env
        )

        return names

    def is_installed(self, package: str) -> bool:
        return (
            subprocess.run(
                ["pacman", "-Q", package],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ).returncode
            == 0
        )

    def system_update(self) -> None:
        _try_run([self.helper, "-Syu", *self.flags], "failed to perform system update")
