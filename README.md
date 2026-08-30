# xiu-cli

The Python CLI control script, system manager, and upstream synchronization suite for the **xiu** ecosystem.

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](/LICENSE)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)

---

## Features & Subcommands

* **Upstream Drift Monitoring (`xiu check`)**:
  * Scans `xiu` (dotfiles), `xiu-shell`, and `xiu-cli`.
  * Reports ahead/behind commit counts against `upstream/main`.
  * Verifies working tree cleanliness and detects core file modifications.
* **Automated Upstream Sync (`xiu sync`)**:
  * Fetches `upstream/main` across all repositories.
  * Performs conflict-free `ort` merge into local `xiu` branch.
  * Pushes clean merges directly to GitHub (`origin/xiu`).
* **Clipboard Manager Integration (`xiu clipboard`)**:
  * Seamless adapter for `clipvault` with fallback to `cliphist`.
* **Dotfiles Deployment (`xiu install`)**:
  * Manifest-driven interactive component installer and updater.
* **Desktop Control Suite**:
  * `xiu shell`: Start, daemonize, inspect, or send IPC calls to the shell.
  * `xiu scheme`: Dynamic Material You color scheme generation and switching.
  * `xiu wallpaper`: Wallpaper management with color palette extraction.
  * `xiu screenshot`: Fullscreen, interactive region, or freeze screenshot captures.
  * `xiu record`: Screen and audio recording triggers.

---

## Installation Guide

### Method 1: Arch Linux (AUR)

```sh
# Using paru:
paru -S xiu-cli

# Or using yay:
yay -S xiu-cli
```

---

### Method 2: Nix Flake

Run directly without installing:
```sh
nix run github:yrpcaro/xiu-cli -- check
```

Or add to your NixOS / Home Manager flake inputs:
```nix
{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    xiu-cli = {
      url = "github:yrpcaro/xiu-cli";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, xiu-cli, ... }: {
    # Add to system packages:
    environment.systemPackages = [
      xiu-cli.packages.x86_64-linux.default
    ];
  };
}
```

---

### Method 3: Python pip / pipx / Local Source

On any Linux distribution with Python 3.13+:

```sh
# 1. Clone repository
git clone -b xiu https://github.com/yrpcaro/xiu-cli.git ~/dots/xiu-cli
cd ~/dots/xiu-cli

# 2. Install using pip:
pip install --user .

# Or using pipx:
pipx install .

# Or run directly from local source:
./bin/xiu --help
```

---

## Quick Command Reference

```sh
# Upstream Maintenance
xiu check                    # Report upstream drift and branch health
xiu sync                     # Automate upstream fetch, merge, and push

# Shell & Desktops
xiu shell -d                 # Start shell in detached background mode
xiu shell -k                 # Terminate running shell instances
xiu shell -l                 # Stream live shell logs

# Wallpapers & Themes
xiu wallpaper set /path/img  # Set wallpaper and regenerate dynamic colors
xiu scheme set dark-oceanic  # Switch color scheme

# Utilities
xiu screenshot -r            # Capture region screenshot
xiu record -r                # Record selected screen region
xiu clipboard                # Launch fuzzy clipboard history selector
```

---

## Credits & License

* **Upstream Base**: [caelestia-dots/cli](https://github.com/caelestia-dots/cli) (GNU General Public License v3.0)
* **License**: [GNU General Public License v3.0 (GPL-3.0)](/LICENSE)
