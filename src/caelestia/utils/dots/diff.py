from dataclasses import dataclass, field
from pathlib import Path

from caelestia.utils.dots.manifest import ManifestEntry
from caelestia.utils.dots.source import DotsSource, SourceError


class _Continue(Exception):
    """Signals the deployed-files loop to skip to the next entry."""


@dataclass(frozen=True)
class Changeset:
    place: list[tuple[str, Path]] = field(default_factory=list)  # (repofile, dest) to fast-forward
    conflicts: list[tuple[str, Path]] = field(default_factory=list)  # (repofile, dest) -> write .new
    deletes: list[Path] = field(default_factory=list)  # We placed it, upstream removed it, unmodified
    stale: list[Path] = field(default_factory=list)  # Upstream removed it but user modified it

    def is_empty(self) -> bool:
        return not (self.place or self.conflicts or self.deletes or self.stale)

    @staticmethod
    def compute(
        source: DotsSource,
        applied_rev: str,
        tip: str,
        entries: list[ManifestEntry],
        deployed: dict[str, str],
    ) -> "Changeset":
        """Collect all file changes needed into a Changeset."""

        changed = set(source.changed_files(applied_rev, tip))
        place: list[tuple[str, Path]] = []
        conflicts: list[tuple[str, Path]] = []
        deletes: list[Path] = []
        stale: list[Path] = []

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
                if dest_path not in files_to_deploy:  # Removed file
                    if not dest_path.exists():
                        continue

                    if try_read(applied_rev, src) == dest_path.read_bytes():
                        deletes.append(dest_path)
                    else:
                        stale.append(dest_path)
                elif src in changed:  # Existing file that needs updating
                    if not dest_path.exists():
                        place.append((src, dest_path))
                        continue

                    dest_content = dest_path.read_bytes()
                    if try_read(tip, src) == dest_content:
                        continue  # File is already up to date

                    if try_read(applied_rev, src) == dest_content:
                        place.append((src, dest_path))
                    else:
                        conflicts.append((src, dest_path))
            except _Continue:
                continue

        # New files to deploy
        for dest in files_to_deploy - set(Path(d) for d in deployed):
            src = to_deploy[dest]
            if not dest.exists() or source.blob_at(tip, src) == dest.read_bytes():
                # Dest nonexistent or already equal to new content
                place.append((src, dest))
            else:
                conflicts.append((src, dest))

        return Changeset(place=place, conflicts=conflicts, deletes=deletes, stale=stale)
