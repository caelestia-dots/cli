import shutil
import tempfile
from pathlib import Path


class Deployer:
    """Places files from the dots clone into their destinations."""

    def place(self, src: Path, dest: Path) -> None:
        """Place a whole entry (file or directory tree), replacing any existing dest."""

        if src.is_dir():
            self.place_dir(src, dest)
        else:
            self.place_file(src, dest)

    def place_dir(self, src: Path, dest: Path) -> None:
        """Place a directory tree recursively, overwriting any existing dest files."""

        if dest.is_symlink() or dest.is_file():
            self.remove(dest)

        dest.mkdir(parents=True, exist_ok=True)
        for path in src.rglob("*"):
            if path.is_file():
                self.place_file(path, dest / path.relative_to(src))
            elif path.is_dir():
                (dest / path.relative_to(src)).mkdir(parents=True, exist_ok=True)

    def place_file(self, src: Path, dest: Path) -> None:
        """Atomically place a single file, replacing any existing dest."""

        if dest.is_dir() and not dest.is_symlink():
            self.remove(dest)

        dest.parent.mkdir(parents=True, exist_ok=True)
        f = tempfile.NamedTemporaryFile(dir=dest.parent, delete=False)
        f.close()
        try:
            shutil.copyfile(src, f.name)
            shutil.copymode(src, f.name)
            Path(f.name).replace(dest)
        except BaseException:
            Path(f.name).unlink()
            raise

    def write_new(self, src: Path, dest: Path) -> Path:
        """Write the upstream version alongside dest as <dest>.new and return that path."""

        new_path = dest.parent / f"{dest.name}.new"
        self.place_file(src, new_path)
        return new_path

    def remove(self, path: Path) -> None:
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
