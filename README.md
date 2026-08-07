# caelestia-cli

The main control script for the Caelestia dotfiles.

<details><summary id="dependencies">External dependencies</summary>

- [`libnotify`](https://gitlab.gnome.org/GNOME/libnotify) - sending notifications
- [`swappy`](https://github.com/jtheoof/swappy) - screenshot editor
- [`grim`](https://gitlab.freedesktop.org/emersion/grim) - taking screenshots
- [`dart-sass`](https://github.com/sass/dart-sass) - discord theming
- [`wl-clipboard`](https://github.com/bugaevc/wl-clipboard) - copying to clipboard
- [`slurp`](https://github.com/emersion/slurp) - selecting an area
- [`gpu-screen-recorder`](https://git.dec05eba.com/gpu-screen-recorder/about) - screen recording
- `glib2` - closing notifications
- [`cliphist`](https://github.com/sentriz/cliphist) - clipboard history
- [`fuzzel`](https://codeberg.org/dnkl/fuzzel) - clipboard history/emoji picker

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
- `<format>` is the output format: `hex` or `rgb`

#### Examples

- `{{ primary.hex }}` outputs `3f4ba2`
- `{{ primary.rgb }}` outputs `rgb(193, 132, 207)`

Output files are written to `~/.local/state/caelestia/theme/`. You can symlink them to your desired locations.

### User colour schemes

Custom user colour schemes can be added in `~/.config/caelestia/schemes/`.

#### Directory structure

`~/.config/caelestia/schemes/<name>/<flavour>/<mode>.txt`

- `<name>` is the name of your scheme (e.g. `mytheme`).
- `<flavour>` is the flavour (e.g. `default`, `mocha`).
- `<mode>` is the mode: `dark` or `light`.

Once created, these schemes will then be available in the launcher `>scheme`, in `caelestia scheme list`, and in shell autocompletions.

> [!TIP]
> User scheme will replace any existing one with the same name. You can also extend existing scheme by creating new flavours or modes (for example new flavour for catpuccin, `~/.config/caelestia/schemes/catppuccin/newflavour/dark.txt`).

#### Examples

Each file is a plain text list of `key value` pairs (no `#` prefixes).
- `background 0a0f0f`
- `primary 9bd0cc`

<details><summary>Example colour scheme content</summary>

```text
background 0a0f0f
onBackground dce8e6
surface 0a0f0f
surfaceDim 0a0f0f
surfaceBright 242e2d
surfaceContainerLowest 000000
surfaceContainerLow 0e1514
surfaceContainer 131b1a
surfaceContainerHigh 192120
surfaceContainerHighest 1d2827
onSurface dce8e6
surfaceVariant 1d2827
onSurfaceVariant a2adac
outline 6d7876
outlineVariant 3f4a49
inverseSurface f6faf9
inverseOnSurface 515655
shadow 000000
scrim 000000
surfaceTint 9bd0cc
primary 9bd0cc
primaryDim 8ec2bf
onPrimary 0d4845
primaryContainer 255b58
onPrimaryContainer b8ede9
inversePrimary 336764
primaryFixed b7ede9
primaryFixedDim a9deda
onPrimaryFixed 0c4744
onPrimaryFixedVariant 306461
secondary b0ccc9
secondaryDim a3bebc
onSecondary 2c4543
secondaryContainer 27403e
onSecondaryContainer a9c5c2
secondaryFixed cce8e5
secondaryFixedDim bedad7
onSecondaryFixed 2b4442
onSecondaryFixedVariant 47605e
tertiary d5efff
tertiaryDim b6e3fe
onTertiary 2e5c72
tertiaryContainer b6e3fe
onTertiaryContainer 255369
tertiaryFixed b6e3fe
tertiaryFixedDim a8d5ef
onTertiaryFixed 0b4156
onTertiaryFixedVariant 2f5d73
error fa746f
errorDim c54d4a
onError 490006
errorContainer 871f21
onErrorContainer ff9993
primaryPaletteKeyColor 4c807d
secondaryPaletteKeyColor 627c7a
tertiaryPaletteKeyColor 517d94
neutralPaletteKeyColor 737877
neutralVariantPaletteKeyColor 6e7978
errorPaletteKeyColor c84f4c
primary_paletteKeyColor 4c807d
secondary_paletteKeyColor 627c7a
tertiary_paletteKeyColor 517d94
neutral_paletteKeyColor 737877
neutral_variant_paletteKeyColor 6e7978
term0 343434
term1 769e00
term2 56e2c0
term3 81fcce
term4 76b6b3
term5 7aaee9
term6 83d8c9
term7 cddcd3
term8 9aa59e
term9 85b900
term10 41f7d0
term11 cdffe9
term12 a3c8c3
term13 a2c0f7
term14 8bedd9
term15 ffffff
rosewater f1f3e5
flamingo e3e4c5
pink bae2ff
mauve 60cfe8
red 8ab5ff
maroon abbef0
peach a9daac
yellow d3fae8
green 8df1df
teal 9feee7
sky 93eae9
sapphire 70d7db
blue 57cdda
lavender 86d9e7
klink 00969e
klinkSelection 00969e
kvisited 008ca9
kvisitedSelection 008ca9
knegative 838f00
knegativeSelection 838f00
kneutral 34c359
kneutralSelection 34c359
kpositive 00beab
kpositiveSelection 00beab
text dce8e6
subtext1 a2adac
subtext0 6d7876
overlay2 5f6967
overlay1 505958
overlay0 434b4a
surface2 353d3c
surface1 282e2e
surface0 191f1e
base 0a0f0f
mantle 0a0f0f
crust 090e0e
success B5CCBA
onSuccess 213528
successContainer 374B3E
onSuccessContainer D1E9D6
```

</details>

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
