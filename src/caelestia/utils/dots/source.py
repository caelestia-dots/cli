import shutil
import subprocess
from pathlib import Path

from caelestia.utils.dots.manifest import Manifest
from caelestia.utils.paths import dots_dir, get_config


class SourceError(Exception):
    """Raised when a git operation against the dots clone fails."""


class DotsSource:
    _fetched_source: bool = False

    def __init__(self) -> None:
        cfg = get_config().get("dots", {})
        self.url = cfg.get("url", "https://github.com/caelestia-dots/caelestia.git")
        self.branch = cfg.get("branch", "main")

    @property
    def remote_ref(self) -> str:
        return f"origin/{self.branch}"

    def exists(self) -> bool:
        return (dots_dir / ".git").is_dir()

    def working_path(self, relpath: str | Path) -> Path:
        """Get a Path relative to the dots dir."""
        return dots_dir / relpath

    def ensure(self) -> None:
        """Clone the repo if absent, otherwise fetch the latest refs.

        If the configured url changed, the stale clone is removed and re-cloned
        from the new source.
        """

        if self.exists():
            if self.current_url() == self.url:
                if DotsSource._fetched_source:
                    return

                self._git("fetch", "--prune", "origin", self.branch)
                DotsSource._fetched_source = True
                return
            shutil.rmtree(dots_dir)

        dots_dir.parent.mkdir(parents=True, exist_ok=True)
        self._run("git", "clone", "--branch", self.branch, self.url, str(dots_dir))
        DotsSource._fetched_source = True

    def current_url(self) -> str:
        return self._git("remote", "get-url", "origin").strip()

    def checkout_tip(self) -> str:
        """Reset the working tree to the fetched tip and return its commit hash."""

        self._git("reset", "--hard", self.remote_ref)
        return self.tip_rev()

    def tip_rev(self) -> str:
        return self._git("rev-parse", self.remote_ref).strip()

    def changed_files(self, base: str, head: str) -> list[str]:
        """Repo-relative paths that differ between two revisions."""

        out = self._git("diff", "--name-only", base, head)
        return [line for line in out.splitlines() if line]

    def has_rev(self, rev: str) -> bool:
        """Whether `rev` resolves to a commit."""

        try:
            self._git("rev-parse", "--verify", "--quiet", f"{rev}^{{commit}}")
            return True
        except SourceError:
            return False

    def clean(self) -> None:
        """Remove all untracked files in the git repo."""
        self._git("clean", "-fdx")

    # --- Accessors ---

    def manifest_at(self, ref: str) -> Manifest:
        return Manifest.parse(self.text_at(ref, "manifest.toml"))

    def text_at(self, ref: str, relpath: str) -> str:
        return self._git("show", f"{ref}:{relpath}")

    def blob_at(self, ref: str, relpath: str) -> bytes:
        return self._git_bytes("show", f"{ref}:{relpath}")

    def files_at(self, ref: str, relpath: str) -> list[str]:
        """Repo-relative paths of all files under relpath at ref (the path itself if it is a file)."""

        out = self._git("ls-tree", "-r", "--name-only", ref, "--", relpath)
        return [line for line in out.splitlines() if line]

    # --- Helpers ---

    def _git(self, *args: str) -> str:
        return self._run("git", "-C", str(dots_dir), *args)

    def _git_bytes(self, *args: str) -> bytes:
        result = subprocess.run(["git", "-C", str(dots_dir), *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            raise SourceError(result.stderr.decode().strip() or f"git {' '.join(args)} failed")
        return result.stdout

    def _run(self, *cmd: str) -> str:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            raise SourceError(result.stderr.strip() or f"{' '.join(cmd)} failed")
        return result.stdout
