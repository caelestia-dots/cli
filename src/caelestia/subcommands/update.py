import sys
from argparse import Namespace
from pathlib import Path

from caelestia.utils.dots.deployer import Deployer
from caelestia.utils.dots.diff import Changeset
from caelestia.utils.dots.manifest import ComponentError, Manifest, ManifestError
from caelestia.utils.dots.misc import build_local_packages, run_hooks
from caelestia.utils.dots.packages import PackageError, PackageInstaller
from caelestia.utils.dots.source import DotsSource, SourceError
from caelestia.utils.dots.state import DotsState
from caelestia.utils.io import disable_input, fatal, info, log, prompt_selection, warn


class Command:
    args: Namespace

    def __init__(self, args: Namespace) -> None:
        self.args = args

    def run(self) -> None:
        if self.args.noconfirm:
            disable_input()

        state = DotsState.load()
        if state.applied_rev is None:
            fatal("dots not installed yet. Run `caelestia install` first.")

        # Run system update
        try:
            installer = PackageInstaller.get(self.args.aur_helper or state.aur_helper, self.args.noconfirm)
            installer.system_update()
        except PackageError as e:
            fatal(e)

        # Get manifest or exit if up to date
        source, tip, manifest = self.fetch_manifest(state, state.applied_rev)

        # Apply file changes
        entries = manifest.enabled_entries()
        try:
            changeset = Changeset.compute(source, state.applied_rev, tip, entries, state.deployed_files)
            source.checkout_tip()
        except SourceError as e:
            fatal(e)
        new_files, revived_files, placed = self.deploy_changeset(source, changeset)

        # Persist file changes immediately so a later failure can't lose track of them
        deployed = dict(state.deployed_files)
        for dest in (*changeset.deletes, *changeset.stale, *changeset.untracked):
            deployed.pop(str(dest), None)
        for repofile, dest in changeset.remap:
            deployed[str(dest)] = repofile
        deployed.update(placed)
        state.deployed_files = deployed
        state.save()

        # Install new/remove old packages
        desired = manifest.enabled_packages()
        desired_local = manifest.enabled_local_packages()
        try:
            state.packages = self.sync_packages(installer, state.packages, desired)
            state.save()
            state.local_packages = self.sync_local_packages(installer, source, state.local_packages, desired_local)
            state.save()
        except PackageError as e:
            fatal(e)

        # Run hooks
        run_hooks(manifest, "post_update")

        # Mark the new revision applied
        state.applied_rev = tip
        state.enabled_components = manifest.enabled_components
        state.aur_helper = getattr(installer, "helper", state.aur_helper)
        state.save()

        self.summarize(changeset, new_files, revived_files)

    def fetch_manifest(self, state: DotsState, applied_rev: str) -> tuple[DotsSource, str, Manifest]:
        print()
        log("Fetching dots repo...")
        source = DotsSource()
        try:
            source.ensure()
            tip = source.tip_rev()
            if tip == applied_rev:
                info("Dots already up to date.")
                sys.exit(0)

            manifest = source.manifest_at(tip)
            if source.has_rev(applied_rev):
                known = set(source.manifest_at(applied_rev).components)
            else:
                # Treat all components as known if rev is invalid so we don't overwrite existing prefs
                known = set(manifest.components)
        except (SourceError, ManifestError) as e:
            fatal(e)

        # Enable components recorded at install time + any new components that are default on
        enabled = [
            name
            for name, comp in manifest.components.items()
            if name in state.enabled_components or (name not in known and comp.default)
        ]

        # Let the user opt into any new optional components
        new_comps = [name for name, comp in manifest.components.items() if name not in known and not comp.default]
        if new_comps:
            info(f"New components: {', '.join(new_comps)}")
            enabled += prompt_selection(new_comps, "Components to enable?")

        disabled = [name for name in manifest.components if name not in enabled]
        try:
            manifest.resolve_components(enable=enabled, disable=disabled)
        except ComponentError as e:
            fatal(e)

        info(f"Enabled components: {', '.join(enabled) or 'none'}")

        return source, tip, manifest

    def deploy_changeset(
        self, source: DotsSource, changeset: Changeset
    ) -> tuple[list[Path], list[Path], dict[str, str]]:
        print()

        if changeset.is_empty():
            info("No configs to update.")
            return [], [], {}

        log("Updating configs...")
        deployer = Deployer()

        for repofile, dest in changeset.place:
            src = source.working_path(repofile)
            if not src.exists():
                warn(f"missing in source, skipping: {repofile}")
                continue
            deployer.place_file(src, dest)
            info(f"{repofile} -> {dest}")

        new_files = []
        for repofile, dest in changeset.conflicts:
            src = source.working_path(repofile)
            if not src.exists():
                warn(f"missing in source, skipping: {repofile}")
                continue
            new_path = deployer.write_new(src, dest)
            new_files.append(new_path)
            warn(f"{dest} has local changes; upstream version written as {new_path.name}")

        revived_files = []
        for repofile, dest in changeset.deleted_changed:
            src = source.working_path(repofile)
            if not src.exists():
                warn(f"missing in source, skipping: {repofile}")
                continue
            new_path = deployer.write_new(src, dest)
            revived_files.append(new_path)
            warn(f"{dest} was removed but changed upstream; upstream version written as {new_path.name}")

        for dest in changeset.deletes:
            deployer.remove(dest)
            deployer.prune_empty_dirs(dest, Path.home())
            info(f"Removed {dest}")

        return new_files, revived_files, deployer.deployed_files

    def sync_packages(self, installer: PackageInstaller, current: dict[str, str], desired: list[str]) -> dict[str, str]:
        to_install = [p for p in desired if p not in current]
        to_remove = [p for p in current if p not in desired]
        installed = dict(current)

        if to_install:
            print()
            info(f"Installing new packages: {', '.join(to_install)}")
            # Record each desired name -> its real installed name so removal later is exact
            installed.update(zip(to_install, installer.install(to_install)))

        if to_remove:
            print()
            info(f"Packages no longer required: {', '.join(to_remove)}")
            selected = prompt_selection(to_remove, "Packages to remove?")
            if selected:
                installer.remove([current[p] for p in selected])
                for p in selected:
                    installed.pop(p, None)

        return installed

    def sync_local_packages(
        self, installer: PackageInstaller, source: DotsSource, current: dict[str, list[str]], desired: list[str]
    ) -> dict[str, list[str]]:
        to_build = [p for p in desired if p not in current]
        to_rebuild = self.outdated_local_packages(installer, source, current, desired)
        to_remove = [p for p in current if p not in desired]
        installed = dict(current)

        if to_build:
            print()
            log(f"Building new local packages: {', '.join(to_build)}")
            installed.update(build_local_packages(installer, source, to_build))

        if to_rebuild:
            print()
            log(f"Rebuilding updated local packages: {', '.join(to_rebuild)}")
            installed.update(build_local_packages(installer, source, to_rebuild))

        if to_remove:
            print()
            info(f"Local packages no longer required: {', '.join(to_remove)}")
            selected = prompt_selection(to_remove, "Local packages to remove?")
            if selected:
                installer.remove([pkg for path in selected for pkg in current[path]])
                for path in selected:
                    installed.pop(path, None)

        return installed

    def outdated_local_packages(
        self, installer: PackageInstaller, source: DotsSource, current: dict[str, list[str]], desired: list[str]
    ) -> list[str]:
        """Repo paths whose installed packages are older than what the repo would build (skipped when off Arch)."""

        outdated = []
        for path in desired:
            if path not in current:
                continue

            directory = source.working_path(path)
            if not directory.is_dir():
                continue

            try:
                if installer.needs_rebuild(directory, current[path]):
                    outdated.append(path)
            except PackageError as e:
                # Failed to read PKGBUILD, leave it as-is
                warn(f"could not check {path} for updates, leaving as-is: {e}")

        return outdated

    def summarize(self, changeset: Changeset, new_files: list[Path], revived_files: list[Path]) -> None:
        print()
        conflicts = len(new_files) + len(revived_files)
        info(f"Updated {len(changeset.place)} file(s), removed {len(changeset.deletes)}, {conflicts} conflict(s).")
        if new_files:
            info("The following files were changed upstream but you had edited them locally.")
            info("Your versions were kept; the upstream versions were written alongside as .new:")
            for path in new_files:
                info(f"  {path}")
        if revived_files:
            info("These files were removed by you but changed upstream, so were not restored.")
            info("The upstream versions were written alongside as .new:")
            for path in revived_files:
                info(f"  {path}")
        if changeset.stale:
            info("These files are no longer managed but differ from what was installed, so were kept:")
            for path in changeset.stale:
                info(f"  {path}")
