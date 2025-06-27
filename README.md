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
-   [`wl-screenrec`](https://github.com/russelltg/wl-screenrec) - screen recording
-   `glib2` - closing notifications
-   `libpulse` - getting audio device
-   [`cliphist`](https://github.com/sentriz/cliphist) - clipboard history
-   [`fuzzel`](https://codeberg.org/dnkl/fuzzel) - clipboard history/emoji picker

</details>

## Installation

### Package manager (recommended)

The cli is available from the AUR as `caelestia-cli-git`. To install it you can use
an AUR helper like [`yay`](https://github.com/Jguer/yay), or manually download the
PKGBUILD and run `makepkg -si`.

e.g. using yay

```sh
yay -S caelestia-cli-git
```

### Manual installation

Install all [dependencies](#dependencies), then install
[`python-build`](https://github.com/pypa/build),
[`python-installer`](https://github.com/pypa/installer),
[`python-hatch`](https://github.com/pypa/hatch) and
[`python-hatch-vcs`](https://github.com/ofek/hatch-vcs).

e.g. via an AUR helper (yay)

```sh
yay -S libnotify swappy grim dart-sass app2unit wl-clipboard slurp wl-screenrec glib2 libpulse cliphist fuzzel python-build python-installer python-hatch python-hatch-vcs
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
> caelestia -h
usage: caelestia [-h] COMMAND ...

Main control script for the Caelestia dotfiles

options:
  -h, --help    show this help message and exit

subcommands:
  valid subcommands

  COMMAND       the subcommand to run
    shell       start or message the shell
    toggle      toggle a special workspace
    scheme      manage the colour scheme
    screenshot  take a screenshot
    record      start a screen recording
    clipboard   open clipboard history
    emoji       emoji/glyph utilities
    wallpaper   manage the wallpaper
    pip         picture in picture utilities
```
