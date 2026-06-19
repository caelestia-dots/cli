import shutil
import textwrap
from argparse import Namespace
from pathlib import Path

from caelestia.utils.dots.deployer import Deployer
from caelestia.utils.dots.legacy import (
    LEGACY_META_PKG,
    detect_legacy_repo,
    legacy_config_symlinks,
    legacy_symlinks,
    legacy_to_delete,
)
from caelestia.utils.dots.manifest import ComponentError, Manifest, ManifestError
from caelestia.utils.dots.misc import build_local_packages, run_hooks
from caelestia.utils.dots.packages import DEFAULT_AUR_HELPER, PackageError, PackageInstaller
from caelestia.utils.dots.source import DotsSource, SourceError
from caelestia.utils.dots.state import DotsState
from caelestia.utils.io import confirm, disable_input, fatal, info, log, pause, prompt_selection, warn
from caelestia.utils.paths import (
    config_backup_dir,
    config_dir,
)


def _parse_list_arg(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _deref_symlink(link: Path, target: Path) -> None:
    """Replace symlink `link` with a real copy of `target`'s content."""

    bak = link.rename(link.parent / f"{link.name}.bak")
    try:
        if target.is_dir():
            shutil.copytree(target, link, symlinks=True)
        else:
            shutil.copy2(target, link)
    except OSError:
        bak.rename(link)
        raise
    bak.unlink()


class Command:
    args: Namespace

    def __init__(self, args: Namespace) -> None:
        self.args = args

    def run(self) -> None:
        if self.args.noconfirm:
            disable_input()

        self.print_greeting()
        self.create_backup()
        legacy_dir = detect_legacy_repo()  # Detect legacy repo first cause deploy overwrites legacy syms

        source, tip, manifest = self.fetch_manifest()
        try:
            installer, packages, local_packages = self.install_packages(source, manifest)
        except PackageError as e:
            fatal(e)
        run_hooks(manifest, "post_package")
        self.dereference_legacy(legacy_dir)  # Copy legacy content into place before deploy overwrites the symlinks
        deployed = self.deploy_configs(source, manifest)
        run_hooks(manifest, "post_install")

        DotsState(
            aur_helper=getattr(installer, "helper", DEFAULT_AUR_HELPER),
            applied_rev=tip,
            enabled_components=manifest.enabled_components,
            packages=packages,
            local_packages=local_packages,
            deployed_files=deployed,
        ).save()

        self.migrate_legacy(installer, legacy_dir)
        self.print_done()

    def print_greeting(self) -> None:
        print(
            "\033[38;2;150;241;241m"  # Caelestia colour
            + textwrap.dedent(
                r"""
                ╭─────────────────────────────────────────────────╮
                │      ______           __          __  _         │
                │     / ____/___ ____  / /__  _____/ /_(_)___ _   │
                │    / /   / __ `/ _ \/ / _ \/ ___/ __/ / __ `/   │
                │   / /___/ /_/ /  __/ /  __(__  ) /_/ / /_/ /    │
                │   \____/\__,_/\___/_/\___/____/\__/_/\__,_/     │
                │                                                 │
                ╰─────────────────────────────────────────────────╯
                """
            )
            + "\033[0m"
        )
        info("Welcome to the Caelestia dotfiles installer!")
        info("Here's a quick overview on what this command is going to do:")
        info("  - Install dependencies")
        info("  - Install config files")
        info("The installer does NOT set up hardware/system level configs (e.g. drivers). Please do this yourself.")
        pause()
        print()

    def create_backup(self) -> None:
        if config_dir.exists():
            if not confirm("Back up the config directory?", default=True):
                return

            log(f"Creating a backup of {config_dir}...")
            if config_backup_dir.exists():
                if not confirm("A backup already exists, overwrite?", default=False):
                    info("Not creating backup.")
                    return

                log("Deleting old backup...")
                shutil.rmtree(config_backup_dir)

            shutil.copytree(config_dir, config_backup_dir, symlinks=True)
            info(f"Created backup at {config_backup_dir}")

    def fetch_manifest(self) -> tuple[DotsSource, str, Manifest]:
        print()
        log("Fetching dots repo...")
        source = DotsSource()
        try:
            source.ensure()
            tip = source.checkout_tip()
        except SourceError as e:
            fatal(e)

        enable = _parse_list_arg(self.args.enable_components)
        disable = _parse_list_arg(self.args.disable_components)
        try:
            manifest = source.manifest_at(tip)

            # No flags given, prompt user for non-default components
            if enable is None and disable is None:
                optional = [name for name, comp in manifest.components.items() if not comp.default]
                if optional:
                    enable = prompt_selection(optional, "Components to enable?")

            manifest.resolve_components(enable=enable, disable=disable)
        except (SourceError, ManifestError, ComponentError) as e:
            fatal(e)

        names = ", ".join(manifest.enabled_components) or "none"
        info(f"Enabled components: {names}")

        return source, tip, manifest

    def deploy_configs(self, source: DotsSource, manifest: Manifest) -> dict[str, str]:
        print()
        log("Installing configs...")
        deployer = Deployer()
        for entry in manifest.enabled_entries():
            src = source.working_path(entry.expanded_src())
            if not src.exists():
                warn(f"missing in source, skipping: {entry.src}")
                continue

            dests = entry.expanded_dests()
            if not dests:
                warn(f"dest glob matched nothing, skipping: {entry.dest}")
                continue

            for dest in dests:
                deployer.place(src, Path(dest))
                info(f"{entry.src} -> {dest}")

        return deployer.deployed_files

    def install_packages(
        self, source: DotsSource, manifest: Manifest
    ) -> tuple[PackageInstaller, list[str], dict[str, list[str]]]:
        installer = PackageInstaller.get(self.args.aur_helper, self.args.noconfirm)

        packages = manifest.enabled_packages()
        if packages:
            print()
            log("Installing packages...")
            installer.install(packages)

        local_packages = {}
        local_dirs = manifest.enabled_local_packages()
        if local_dirs:
            print()
            log("Building local packages...")
            local_packages = build_local_packages(installer, source, local_dirs)

        return installer, packages, local_packages

    def dereference_legacy(self, legacy_dir: Path | None) -> None:
        """Replace legacy symlinks with real copies of their targets."""

        symlinks = legacy_symlinks(legacy_dir)
        if not symlinks:
            return

        print()
        log("Preserving content from legacy symlinks...")
        for path in symlinks:
            target = path.resolve()
            if not target.exists():
                continue

            try:
                _deref_symlink(path, target)
                info(f"Copied {target} -> {path}")
            except OSError as e:
                warn(f"failed to preserve {path}: {e}")

    def deref_backup_syms(self, legacy_dir: Path | None) -> None:
        """Deref the backup's legacy symlinks before the repo is cleared, so the backup keeps real content."""

        if not config_backup_dir.is_dir():
            return

        for link in legacy_config_symlinks(config_backup_dir, legacy_dir):
            target = link.resolve()
            if not target.exists():
                continue

            try:
                _deref_symlink(link, target)
            except OSError as e:
                warn(f"failed to preserve {link} in backup: {e}")

    def migrate_legacy(self, installer: PackageInstaller, legacy_dir: Path | None) -> None:
        """Clean up a previous install.fish setup (repo, symlinks and metapackage)."""

        to_delete = legacy_to_delete(legacy_dir)
        meta_installed = installer.is_installed(LEGACY_META_PKG)
        if not to_delete and not meta_installed:
            return

        print()
        log("Found a legacy Caelestia installation...")
        if not confirm("Clear legacy installation?"):
            return

        deployer = Deployer()
        try:
            self.deref_backup_syms(legacy_dir)
            for path in to_delete:
                deployer.remove(path)
                info(f"Deleted {path}")

            if meta_installed:
                log("Removing legacy meta package...")
                installer.remove([LEGACY_META_PKG])
        except (OSError, PackageError) as e:
            warn(f"could not fully clear the legacy installation: {e}")

    def print_done(self) -> None:
        print()
        info("All done! Caelestia has been installed.")
        info("A few things to finish up:")
        info("  - A reboot is recommended for all changes take effect")
        info("  - Edit `~/.config/caelestia/hypr-vars.conf` to set default apps, keybinds and much more")
        info("  - Edit `~/.config/caelestia/hypr-user.conf` to set your monitor layout and other Hyprland configs")
        info("  - Run `caelestia update` later to pull in the latest changes")
        info("Enjoy! For support (or to just hang out), join our Discord server: https://discord.gg/BGDCFCmMBk")
