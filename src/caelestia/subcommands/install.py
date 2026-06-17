import shutil
import textwrap
from argparse import Namespace
from pathlib import Path

from caelestia.utils.dots.deployer import Deployer
from caelestia.utils.dots.manifest import ComponentError, Manifest, ManifestError
from caelestia.utils.dots.misc import build_local_packages, run_hooks
from caelestia.utils.dots.packages import DEFAULT_AUR_HELPER, PackageInstaller
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


class Command:
    args: Namespace

    def __init__(self, args: Namespace) -> None:
        self.args = args

    def run(self) -> None:
        if self.args.noconfirm:
            disable_input()

        self.print_greeting()
        self.create_backup()

        source, tip, manifest = self.fetch_manifest()
        deployed = self.deploy_configs(source, manifest)
        helper, packages, local_packages = self.install_packages(source, manifest)
        run_hooks(manifest, "post_install")

        DotsState(
            aur_helper=helper,
            applied_rev=tip,
            enabled_components=manifest.enabled_components,
            packages=packages,
            local_packages=local_packages,
            deployed_files=deployed,
        ).save()

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

    def install_packages(self, source: DotsSource, manifest: Manifest) -> tuple[str, list[str], dict[str, list[str]]]:
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

        return getattr(installer, "helper", DEFAULT_AUR_HELPER), packages, local_packages

    def print_done(self) -> None:
        print()
        info("All done! Caelestia has been installed.")
        info("A few things to finish up:")
        info("  - A reboot is recommended for all changes take effect")
        info("  - Edit `~/.config/caelestia/hypr-vars.conf` to set default apps, keybinds and much more")
        info("  - Edit `~/.config/caelestia/hypr-user.conf` to set your monitor layout and other Hyprland configs")
        info("  - Run `caelestia update` later to pull in the latest changes")
        info("Enjoy! For support (or to just hang out), join our Discord server: https://discord.gg/BGDCFCmMBk")
