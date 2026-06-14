import os
import shutil
import subprocess
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path

from caelestia.utils.io import fatal, info

DEFAULT_AUR_HELPER = "paru"
AUR_HELPERS = DEFAULT_AUR_HELPER, "yay"


def _install_aur_helper(helper: str, noconfirm: bool = False) -> None:
    pacman_cmd = ["sudo", "pacman", "-S", "--needed", "git", "base-devel"]
    if noconfirm:
        pacman_cmd.append("--noconfirm")
    subprocess.run(pacman_cmd, check=True)

    repo_url = f"https://aur.archlinux.org/{helper}.git"
    with tempfile.TemporaryDirectory() as repo_dir:
        subprocess.run(["git", "clone", repo_url, repo_dir], check=True)

        makepkg_cmd = ["makepkg", "-si"]
        if noconfirm:
            makepkg_cmd.append("--noconfirm")
        subprocess.run(makepkg_cmd, cwd=repo_dir, check=True)

    if helper == "yay":
        subprocess.run(["yay", "-Y", "--gendb"], check=True)
        subprocess.run(["yay", "-Y", "--devel", "--save"], check=True)
    elif helper == "paru":
        subprocess.run(["paru", "--gendb"], check=True)


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


class ArchInstaller(PackageInstaller):
    def __init__(self, helper: str, noconfirm: bool = False) -> None:
        self.helper = helper
        self.flags = ["--noconfirm"] if noconfirm else []

    def install(self, packages: list[str], extra_flags: list[str] = []) -> None:
        if not packages:
            return
        subprocess.run([self.helper, "-S", "--needed", *self.flags, *extra_flags, *packages], check=True)

    def remove(self, packages: list[str]) -> None:
        if not packages:
            return
        subprocess.run([self.helper, "-Rns", *self.flags, *packages], check=True)

    def build_install(self, directory: Path) -> list[str]:
        srcinfo = subprocess.check_output(["makepkg", "--printsrcinfo"], cwd=directory, text=True)
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

        self.install(depends, extra_flags=["--asdeps"])

        # Stop makepkg from resetting sudo
        env = {**os.environ, "PACMAN_AUTH": "sudo"}
        # -f = force, -s = sync deps, -i = install
        subprocess.run(["makepkg", "-fsi", *self.flags], cwd=directory, env=env, check=True)

        return names
