from dataclasses import dataclass, field
from pathlib import Path

from caelestia.utils.dots.manifest import ManifestEntry
from caelestia.utils.dots.source import DotsSource, SourceError
from caelestia.utils.io import warn


class _Continue(Exception):
    """Signals the deployed-files loop to skip to the next entry."""


@dataclass(frozen=True)
class Changeset:
    place: list[tuple[str, Path]] = field(default_factory=list)  # (repofile, dest) to fast-forward
    conflicts: list[tuple[str, Path]] = field(default_factory=list)  # (repofile, dest) -> write .new
    deletes: list[Path] = field(default_factory=list)  # We placed it, upstream removed it, unmodified
    stale: list[Path] = field(default_factory=list)  # Upstream removed it but user modified it
    deleted_changed: list[tuple[str, Path]] = field(default_factory=list)  # User deleted it, upstream changed -> .new
    untracked: list[Path] = field(default_factory=list)  # Gone + no longer managed; drop from state
    remap: list[tuple[str, Path]] = field(default_factory=list)  # Up to date but source path moved; restate mapping

    def is_empty(self) -> bool:
        return not (self.place or self.conflicts or self.deletes or self.stale or self.deleted_changed)

    @staticmethod
    def compute(
        source: DotsSource,
        applied_rev: str,
        tip: str,
        entries: list[ManifestEntry],
        deployed: dict[str, str],
    ) -> "Changeset":
        """Collect all file changes needed into a Changeset."""

        has_base = source.has_rev(applied_rev)
        if not has_base:
            warn(
                "the previously applied revision is missing from the dots clone; files that differ "
                "from the latest version will be written as .new instead of updated in place."
            )

        changed = set(source.changed_files(applied_rev, tip)) if has_base else set()
        place: list[tuple[str, Path]] = []
        conflicts: list[tuple[str, Path]] = []
        deletes: list[Path] = []
        stale: list[Path] = []
        deleted_changed: list[tuple[str, Path]] = []
        untracked: list[Path] = []
        remap: list[tuple[str, Path]] = []

        # Collect all files to deploy (entry sources can be dirs so we recurse into them)
        to_deploy: dict[Path, str] = {}
        for entry in entries:
            src_root = str(entry.expanded_src())
            repo_files = source.files_at(tip, src_root)
            for dest in entry.expanded_dests():
                for repo_file in repo_files:
                    to_deploy[dest / Path(repo_file).relative_to(src_root)] = repo_file
        files_to_deploy = set(to_deploy)

        # Already deployed files
        for dest, src in deployed.items():
            dest_path = Path(dest)

            def try_read(rev: str, path: str) -> bytes:
                try:
                    return source.blob_at(rev, path)
                except SourceError:
                    # Read failed, keep it just in case
                    stale.append(dest_path)
                    raise _Continue

            try:
                if dest_path not in files_to_deploy:  # No longer managed by any entry
                    if not dest_path.exists():
                        # Gone from disk and no entry manages it
                        untracked.append(dest_path)
                        continue

                    if has_base and try_read(applied_rev, src) == dest_path.read_bytes():
                        deletes.append(dest_path)
                    else:
                        stale.append(dest_path)
                else:  # Still managed; `src` is what we last placed, `new_src` the current source
                    new_src = to_deploy[dest_path]
                    if not dest_path.exists():
                        # User deleted a managed file locally
                        if has_base and new_src == src and new_src not in changed:
                            continue  # Respect the deletion; upstream has nothing new to offer
                        # Upstream changed it (or base is unknown): surface as .new, don't restore
                        deleted_changed.append((new_src, dest_path))
                        continue

                    if has_base and new_src == src and new_src not in changed:
                        continue  # Unchanged upstream

                    dest_content = dest_path.read_bytes()
                    if try_read(tip, new_src) == dest_content:
                        # Already up to date; restate the mapping if the source path moved
                        if new_src != src:
                            remap.append((new_src, dest_path))
                        continue

                    # Fast-forward only when the user hasn't edited since last deploy
                    if has_base and try_read(applied_rev, src) == dest_content:
                        place.append((new_src, dest_path))
                    else:
                        conflicts.append((new_src, dest_path))
            except _Continue:
                continue

        # New files to deploy
        for dest in files_to_deploy - set(Path(d) for d in deployed):
            src = to_deploy[dest]
            try:
                new_content = source.blob_at(tip, src)
            except SourceError:
                # Failed to read the upstream blob; skip rather than abort the whole update
                warn(f"could not read from source, skipping: {src}")
                continue
            if not dest.exists() or new_content == dest.read_bytes():
                # Dest nonexistent or already equal to new content
                place.append((src, dest))
            else:
                conflicts.append((src, dest))

        return Changeset(
            place=place,
            conflicts=conflicts,
            deletes=deletes,
            stale=stale,
            deleted_changed=deleted_changed,
            untracked=untracked,
            remap=remap,
        )
