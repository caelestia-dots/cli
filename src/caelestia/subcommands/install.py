import os
import shutil
import subprocess
import textwrap
from argparse import Namespace
from pathlib import Path

from caelestia.utils.dots.deployer import Deployer
from caelestia.utils.dots.manifest import ComponentError, Manifest, ManifestError, expand, expand_dests
from caelestia.utils.dots.packages import DEFAULT_AUR_HELPER, PackageInstaller
from caelestia.utils.dots.source import DotsSource, SourceError
from caelestia.utils.dots.state import DotsState
from caelestia.utils.io import PROMPT_COLOUR, confirm, disable_input, fatal, format_msg, info, log, pause, prompt, warn
from caelestia.utils.paths import (
    config_backup_dir,
    config_dir,
    dots_dir,
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
        self.deploy_configs(source, manifest)
        helper, packages, local_packages = self.install_packages(source, manifest)
        self.run_hooks(manifest)

        DotsState(
            aur_helper=helper,
            applied_rev=tip,
            enabled_components=manifest.enabled_components,
            packages=packages,
            local_packages=local_packages,
        ).save()

        info("Done!")

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
            manifest.resolve_components(enable=enable, disable=disable)

            if enable is None and disable is None:
                self.prompt_optional_components(manifest)
        except (SourceError, ManifestError, ComponentError) as e:
            fatal(e)

        names = ", ".join(manifest.enabled_components) or "none"
        info(f"Enabled components: {names}")

        return source, tip, manifest

    def prompt_optional_components(self, manifest: Manifest) -> None:
        comp_arr = manifest.disabled_components
        if not comp_arr:
            return

        print(format_msg(PROMPT_COLOUR, "Components to enable?"))
        max_idx_w = len(str(len(comp_arr)))
        for i, comp in enumerate(comp_arr):
            print(format_msg(PROMPT_COLOUR, f"  {i + 1:<{max_idx_w}}\t{comp}"))
        print(format_msg(PROMPT_COLOUR, "[A]ll or (1 2 3, 1-3, ^4)"))
        ans = prompt("", end="").lower().strip()

        def _valid_v(v: str) -> int:
            try:
                i_v = int(v, base=10) - 1  # -1 to translate to 0 index
            except ValueError:
                fatal(f'Invalid input. Given value "{v}" must be an integer.')
            if i_v < 0 or i_v >= len(comp_arr):
                fatal(f'Invalid input. Given value "{v}" must be between 1 and {len(comp_arr)} inclusive.')
            return i_v

        if ans in ("a", "all"):
            manifest.resolve_components(enable=list(manifest.components))
        elif ans:
            enabled: list[str] = []
            toks = ans.split()
            for tok in toks:
                fr, sep, to = tok.partition("-")
                if sep:
                    fr = _valid_v(fr)
                    to = _valid_v(to)
                    if fr > to:
                        fatal(f'Invalid input. Given range "{tok}" must be lo-hi.')
                    enabled += comp_arr[fr : to + 1]
                elif tok.startswith("^"):
                    t = _valid_v(tok[1:])
                    enabled += comp_arr[:t] + comp_arr[t + 1 :]
                else:
                    t = _valid_v(tok)
                    enabled.append(comp_arr[t])
            manifest.resolve_components(enable=list(set(enabled)))

    def deploy_configs(self, source: DotsSource, manifest: Manifest) -> None:
        log("Installing configs...")
        deployer = Deployer()
        for entry in manifest.enabled_entries():
            src = source.working_path(expand(entry.src))
            if not src.exists():
                warn(f"missing in source, skipping: {entry.src}")
                continue

            dests = expand_dests(entry.dest)
            if not dests:
                warn(f"dest glob matched nothing, skipping: {entry.dest}")
                continue

            for dest in dests:
                deployer.place(src, Path(dest))
                info(f"{entry.src} -> {dest}")

    def install_packages(self, source: DotsSource, manifest: Manifest) -> tuple[str, list[str], dict[str, list[str]]]:
        installer = PackageInstaller.get(self.args.aur_helper, self.args.noconfirm)

        packages = manifest.enabled_packages()
        if packages:
            log("Installing packages...")
            installer.install(packages)

        local_packages = {}
        local_dirs = manifest.enabled_local_packages()
        if local_dirs:
            log("Building local packages...")
            for path in local_dirs:
                directory = source.working_path(path)
                if not directory.is_dir():
                    warn(f"missing in repo, skipping: {path}")
                    continue

                log(f"Building {path}...")
                local_packages[path] = installer.build_install(directory)

        return getattr(installer, "helper", DEFAULT_AUR_HELPER), packages, local_packages

    def run_hooks(self, manifest: Manifest) -> None:
        hooks = manifest.enabled_hooks("post_install")
        if not hooks:
            return

        log("Running post-install hooks...")
        env = {**os.environ, "CAELESTIA_DOTS": str(dots_dir)}
        for hook in hooks:
            log(f"Running hook: {hook}")
            result = subprocess.run(hook, shell=True, env=env)
            if result.returncode != 0:
                warn(f"hook exited with {result.returncode}")
