import subprocess

from caelestia.utils.paths import config_dir


def print_version() -> None:
    print("Packages:")
    pkgs = ["caelestia-shell-git", "caelestia-cli-git", "caelestia-meta"]
    versions = subprocess.run(
        ["pacman", "-Q", *pkgs], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
    ).stdout

    for pkg in pkgs:
        if pkg not in versions:
            print(f"    {pkg} not installed")
    print("\n".join(f"    {pkg}" for pkg in versions.splitlines()))

    caelestia_dir = (config_dir / "hypr").resolve().parent
    print("\nCaelestia:")
    caelestia_ver = subprocess.check_output(
        ["git", "--git-dir", caelestia_dir / ".git", "rev-list", "--format=%B", "--max-count=1", "HEAD"], text=True
    )
    print("    Last commit:", caelestia_ver.split()[1])
    print("    Commit message:", *caelestia_ver.splitlines()[1:])

    print("\nQuickshell:")
    print("   ", subprocess.check_output(["qs", "--version"], text=True).strip())

    local_shell_dir = config_dir / "quickshell/caelestia"
    if local_shell_dir.exists():
        print("\nLocal copy of shell found:")

        try:
            shell_ver = subprocess.check_output(
                [
                    "git",
                    "--git-dir",
                    local_shell_dir / ".git",
                    "rev-list",
                    "--format=%B",
                    "--max-count=1",
                    "upstream/main",
                ],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            print("    Last merged upstream commit:", shell_ver.split()[1])
            print("    Commit message:", *shell_ver.splitlines()[1:])
        except subprocess.CalledProcessError:
            print("    Unable to determine last merged upstream commit.")

        shell_ver = subprocess.check_output(
            ["git", "--git-dir", local_shell_dir / ".git", "rev-list", "--format=%B", "--max-count=1", "HEAD"],
            text=True,
        )
        print("\n    Last commit:", shell_ver.split()[1])
        print("    Commit message:", *shell_ver.splitlines()[1:])
