# caelestia-cli

The main control script for the Caelestia dotfiles.

<details><summary id="dependencies">External dependencies</summary>

-   [`libnotfy`](https://gitlab.gnome.org/GNOME/libnotify) - sending notifications
-   [`swappy`](https://github.com/jtheoof/swappy) - screenshot editor
-   [`grim`](https://gitlab.freedesktop.org/emersion/grim) - taking screenshots
-   [`dart-sass`](https://github.com/sass/dart-sass) - discord theming
-   [`app2unit`](https://github.com/Vladimir-csp/app2unit) - launching apps
-   [`wl-clipboard`](https://github.com/bugaevc/wl-clipboard) - copying to clipboard
-   [`slurp`](https://github.com/emersion/slurp) - selecting an area
-   [`gpu-screen-recorder`](https://git.dec05eba.com/gpu-screen-recorder/about) - screen recording
-   `glib2` - closing notifications
-   [`cliphist`](https://github.com/sentriz/cliphist) - clipboard history
-   [`fuzzel`](https://codeberg.org/dnkl/fuzzel) - clipboard history/emoji picker

### Optional dependencies for OCR click-to-copy (`clicktodo` command)

-   [`grim`](https://gitlab.freedesktop.org/emersion/grim) - taking screenshots (already listed above)
-   [`wl-clipboard`](https://github.com/bugaevc/wl-clipboard) - copying to clipboard (already listed above)
-   Python packages: `rapidocr-onnxruntime`, `onnxruntime`, `PyQt6`, `numpy`, `threadpoolctl` (install via `pip install caelestia[ocr]`)

**Performance Note:** The OCR feature uses RapidOCR with ONNXRuntime for optimal CPU performance (5-15x faster than EasyOCR). For best results on high-resolution displays, run the setup script to configure the persistent daemon:

</details>

<details><summary id="optional-dependencies">Optional dependencies</summary>

-   [`papirus-folders`](https://github.com/PapirusDevelopmentTeam/papirus-folders) - automatic folder icon color syncing with theme

> [!NOTE]
> For automatic Papirus folder icon color syncing, `papirus-folders` needs to be able to run with `sudo` without a password prompt.
> 
> **Recommended** - Create a sudoers file:
> ```fish
> # Fish shell
> echo "$USER ALL=(ALL) NOPASSWD: "(which papirus-folders) | sudo tee /etc/sudoers.d/papirus-folders
> sudo chmod 440 /etc/sudoers.d/papirus-folders
> ```
> ```sh
> # Bash/other shells
> echo "$USER ALL=(ALL) NOPASSWD: $(which papirus-folders)" | sudo tee /etc/sudoers.d/papirus-folders
> sudo chmod 440 /etc/sudoers.d/papirus-folders
> ```
> 
> **Alternatively** - Edit the main sudoers file by running `sudo visudo` and adding at the end:
> ```
> your_username ALL=(ALL) NOPASSWD: /usr/bin/papirus-folders
> ```

</details>

## Installation

### Arch linux

The CLI is available from the AUR as `caelestia-cli`. You can install it with an AUR helper
like [`yay`](https://github.com/Jguer/yay) or manually downloading the PKGBUILD and running `makepkg -si`.

A package following the latest commit also exists as `caelestia-cli-git`. This is bleeding edge
and likely to be unstable/have bugs. Regular users are recommended to use the stable package
(`caelestia-cli`).

### Nix

You can run the CLI directly via `nix run`:

```sh
nix run github:caelestia-dots/cli
```

Or add it to your system configuration:

```nix
{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";

    caelestia-cli = {
      url = "github:caelestia-dots/cli";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };
}
```

The package is available as `caelestia-cli.packages.<system>.default`, which can be added to your
`environment.systemPackages`, `users.users.<username>.packages`, `home.packages` if using home-manager,
or a devshell. The CLI can then be used via the `caelestia` command.

> [!TIP]
> The default package does not have the shell enabled by default, which is required for full functionality.
> To enable the shell, use the `with-shell` package. This is the recommended installation method, as
> the CLI exposes the shell via the `shell` subcommand, meaning there is no need for the shell package
> to be exposed.

For home-manager, you can also use the Caelestia's home manager module (explained in
[configuring](https://github.com/caelestia-dots/shell?tab=readme-ov-file#home-manager-module)) that
installs and configures the shell and the CLI.

### Manual installation

Install all [dependencies](#dependencies), then install
[`python-build`](https://github.com/pypa/build),
[`python-installer`](https://github.com/pypa/installer),
[`python-hatch`](https://github.com/pypa/hatch) and
[`python-hatch-vcs`](https://github.com/ofek/hatch-vcs).

e.g. via an AUR helper (yay)

```sh
yay -S libnotify swappy grim dart-sass app2unit wl-clipboard slurp gpu-screen-recorder glib2 cliphist fuzzel python-build python-installer python-hatch python-hatch-vcs
```

Now, clone the repo, `cd` into it, build the wheel via `python -m build --wheel`
and install it via `python -m installer dist/*.whl`. Then, to install the `fish`
completions, copy the `completions/caelestia.fish` file to
`/usr/share/fish/vendor_completions.d/caelestia.fish`.

```sh
git clone https://github.com/caelestia-dots/cli.git
cd cli
python -m build --wheel
sudo python -m installer dist/*.whl
sudo cp completions/caelestia.fish /usr/share/fish/vendor_completions.d/caelestia.fish
```

## Usage

All subcommands/options can be explored via the help flag.

```
$ caelestia -h
usage: caelestia [-h] [-v] COMMAND ...

Main control script for the Caelestia dotfiles

options:
  -h, --help     show this help message and exit
  -v, --version  print the current version

subcommands:
  valid subcommands

  COMMAND        the subcommand to run
    shell        start or message the shell
    toggle       toggle a special workspace
    scheme       manage the colour scheme
    screenshot   take a screenshot
    record       start a screen recording
    clipboard    open clipboard history
    emoji        emoji/glyph utilities
    wallpaper    manage the wallpaper
    resizer      window resizer daemon
    clicktodo    OCR-based click-to-copy from screen
```

### OCR Click-to-Copy (`clicktodo`)

The `clicktodo` command provides an OCR-based workflow for extracting and copying text from anywhere on your screen:

1. Captures a fullscreen screenshot
2. Runs OCR to detect all text on screen (via persistent daemon for speed)
3. Shows an interactive overlay with detected text regions highlighted
4. Click any text region to copy it to clipboard
5. Press `ESC` or right-click to cancel

**Performance:** Uses RapidOCR + ONNXRuntime for 5-15x faster processing than traditional OCR engines. Typical latency: 300-600ms on a 2880x1800 display.

**Setup:**

1. Install OCR dependencies:
   ```sh
   pip install caelestia[ocr]
   # Or manually: pip install rapidocr-onnxruntime onnxruntime PyQt6 numpy
   ```

2. Run the setup script to configure the OCR daemon:
   ```sh
   ./setup-ocr.sh
   ```

   This will:
   - Install dependencies if missing
   - Set up a systemd user service for the OCR daemon
   - Create default configuration at `~/.config/caelestia/ocr.json`
   - Start the daemon (models stay hot in memory for instant responses)

**Requirements:**
- Requires `grim` and `wl-clipboard` (already needed for other features)
- Python 3.13+ with pip

**Hyprland keybinding example:**

Add to your `hyprland.conf`:
```
# Standard mode
bind = SUPER, O, exec, caelestia clicktodo

# Fast mode (more aggressive optimizations)
bind = SUPER SHIFT, O, exec, caelestia clicktodo --fast
```

**Usage:**
```sh
# Standard mode
caelestia clicktodo

# Fast mode (downscales more aggressively, limits max boxes)
caelestia clicktodo --fast --live
```

**Configuration:**

Edit `~/.config/caelestia/ocr.json` to customize:
```json
{
  "provider": "cpu-ort",    // cpu-ort, gpu-rocm, npu-xdna (future)
  "downscale": 0.6,         // Detection downscale factor (0.5-1.0)
  "tiles": 1,               // Parallel tiles (future feature)
  "max_boxes": 300,         // Maximum text boxes to detect
  "use_gpu": false,         // Enable GPU (experimental on AMD)
  "warm_start": true,       // Run warm-up on daemon start
  "performance": {
    "idle_threads": 1,      // Background thread budget when idle
    "standard_threads": 4,  // Default thread budget during normal OCR
    "fast_threads": 0,      // 0 = auto, otherwise specific thread count
    "idle_cores": 1,        // CPU cores kept active when idle
    "standard_cores": 0,    // 0 = auto mid-range core count
    "fast_cores": 0         // 0 = all available cores during bursts
  }
}
```

Set any value to `0` (or omit the key) to allow the daemon to auto-detect from the host CPU. Leave the entire `performance` block out to use adaptive defaults.

**Daemon Management:**
```sh
# Check status
systemctl --user status caelestia-ocrd

# Restart daemon
systemctl --user restart caelestia-ocrd

# Stop daemon
systemctl --user stop caelestia-ocrd

# View logs
journalctl --user -u caelestia-ocrd -f
```

**Future Optimizations:**
- NPU acceleration via AMD XDNA (when ONNX Runtime EP is stable on Linux)
- GPU acceleration via ROCm (when Radeon 890M iGPU is officially supported)
- Parallel tile processing for ultra-high-resolution displays

## Configuring

All configuration options are in `~/.config/caelestia/cli.json`.

<details><summary>Example configuration</summary>

```json
{
    "record": {
        "extraArgs": []
    },
    "wallpaper": {
        "postHook": "echo $WALLPAPER_PATH"  
    },
    "theme": {
        "enableTerm": true,
        "enableHypr": true,
        "enableDiscord": true,
        "enableSpicetify": true,
        "enableFuzzel": true,
        "enableBtop": true,
        "enableGtk": true,
        "enableQt": true
    },
    "toggles": {
        "communication": {
            "discord": {
                "enable": true,
                "match": [{ "class": "discord" }],
                "command": ["discord"],
                "move": true
            },
            "whatsapp": {
                "enable": true,
                "match": [{ "class": "whatsapp" }],
                "move": true
            }
        },
        "music": {
            "spotify": {
                "enable": true,
                "match": [{ "class": "Spotify" }, { "initialTitle": "Spotify" }, { "initialTitle": "Spotify Free" }],
                "command": ["spicetify", "watch", "-s"],
                "move": true
            },
            "feishin": {
                "enable": true,
                "match": [{ "class": "feishin" }],
                "move": true
            }
        },
        "sysmon": {
            "btop": {
                "enable": true,
                "match": [{ "class": "btop", "title": "btop", "workspace": { "name": "special:sysmon" } }],
                "command": ["foot", "-a", "btop", "-T", "btop", "fish", "-C", "exec btop"]
            }
        },
        "todo": {
            "todoist": {
                "enable": true,
                "match": [{ "class": "Todoist" }],
                "command": ["todoist"],
                "move": true
            }
        }
    }
}
```

</details>
