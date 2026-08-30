# xiu-cli

The Python CLI control script and system manager for the **xiu** ecosystem.

## Credits & License

- **Upstream Base**: [caelestia-dots/cli](https://github.com/caelestia-dots/cli) (GNU GPL-3.0)
- **License**: [GNU General Public License v3.0 (GPL-3.0)](/LICENSE)

## Upstream Drift Check & Sync

`xiu-cli` provides built-in tools to manage and monitor upstream sync across all three xiu repositories:

```sh
# Report upstream drift, ahead/behind counts, and core file modifications
xiu check

# Automatically fetch upstream, merge upstream/main into xiu, and push to origin
xiu sync
```

## Installation

`xiu-cli` is distro-agnostic:

### 1. Arch Linux / AUR
Install `xiu-cli` (or `caelestia-cli`):

```sh
paru -S xiu-cli
```

### 2. Nix Flake
Run directly or import via flake:

```sh
nix run github:yrpcaro/xiu-cli
```

### 3. Generic Python Install (pip / pipx)
Install on any Linux distribution:

```sh
pip install .
# or run directly from source via:
./bin/xiu --help
```
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
yay -S libnotify swappy grim dart-sass wl-clipboard slurp gpu-screen-recorder glib2 cliphist fuzzel python-build python-installer python-hatch python-hatch-vcs
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

### Additional steps

#### Auto folder colour theming

For automatic Papirus folder icon colour syncing, you must have [`papirus-folders`](https://github.com/PapirusDevelopmentTeam/papirus-folders)
installed, and `papirus-folders` must to be able to run with `sudo` without a password prompt.

You can allow this by creating a sudoers file:

```sh
echo "$USER ALL=(ALL) NOPASSWD: $(which papirus-folders)" | sudo tee /etc/sudoers.d/papirus-folders
sudo chmod 440 /etc/sudoers.d/papirus-folders
```

#### Chromium-based browser theming

For live Chromium-based browser theming, the CLI must be allowed to create certain directories in `/etc`
and write to them via `sudo` without a password prompt.

You can allow this by creating a sudoers file:

```fish
# Fish shell
for dir in /etc/chromium/policies/managed /etc/brave/policies/managed /etc/opt/chrome/policies/managed
    echo "$USER ALL=(ALL) NOPASSWD: $(which mkdir) -p $dir" | sudo tee -a /etc/sudoers.d/caelestia-chromium
    echo "$USER ALL=(ALL) NOPASSWD: $(which tee) $dir/caelestia.json" | sudo tee -a /etc/sudoers.d/caelestia-chromium
end
sudo chmod 440 /etc/sudoers.d/caelestia-chromium
```

```sh
# Bash/other shells
for dir in /etc/chromium/policies/managed /etc/brave/policies/managed /etc/opt/chrome/policies/managed; do
    echo "$USER ALL=(ALL) NOPASSWD: $(which mkdir) -p $dir" | sudo tee -a /etc/sudoers.d/caelestia-chromium
    echo "$USER ALL=(ALL) NOPASSWD: $(which tee) $dir/caelestia.json" | sudo tee -a /etc/sudoers.d/caelestia-chromium
done
sudo chmod 440 /etc/sudoers.d/caelestia-chromium
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
    install      install the Caelestia dotfiles
    update       update the Caelestia dotfiles
```

### User templates

Custom user templates can be defined in `~/.config/caelestia/templates/`.

#### Template syntax

`{{ <color>.<format> }}`

- `<color>` is a theme color role derived from the Material You color system (e.g. `primary`, `secondary`, `background`)
- `<format>` is the output format: `hex`, `rgb`, `hsl`, or a single channel (`red`, `green`, `blue`, `hue`, `saturation`, `lightness`)

#### Examples

- `{{ primary.hex }}` outputs `3f4ba2`
- `{{ primary.rgb }}` outputs `rgb(193, 132, 207)`
- `{{ primary.red }}` outputs `193`
- `{{ primary.hsl }}` outputs `hsl(268,41%,66%)`
- `{{ primary.hue }}` outputs `268`

Output files are written to `~/.local/state/caelestia/theme/`. You can symlink them to your desired locations.

## Configuring

All configuration options are in `~/.config/caelestia/cli.json`.

<details><summary>Example configuration</summary>

```json
{
    "record": {
        "extraArgs": []
    },
    "wallpaper": {
        "postHook": "echo $WALLPAPER_PATH $SCHEME_NAME $SCHEME_FLAVOUR $SCHEME_MODE $SCHEME_VARIANT $SCHEME_COLOURS"
    },
    "theme": {
        "enableTerm": true,
        "enableHypr": true,
        "enableDiscord": true,
        "enableSpicetify": true,
        "enablePandora": true,
        "enableFuzzel": true,
        "enableBtop": true,
        "enableNvtop": true,
        "enableHtop": true,
        "enableGtk": true,
        "enableQt": true,
        "enableWarp": true,
        "enableChromium": true,
        "enableZed": true,
        "enableCava": true,
        "iconTheme": "Papirus-Dark",
        "iconThemeLight": "Papirus-Light",
        "iconThemeDark": "Papirus-Dark",
        "postHook": "echo $SCHEME_NAME $SCHEME_FLAVOUR $SCHEME_MODE $SCHEME_VARIANT $SCHEME_COLOURS"
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
    },
    "dots": {
        "url": "https://github.com/caelestia-dots/caelestia.git",
        "branch": "main"
    }
}
```

</details>
