import subprocess
from pathlib import Path

from caelestia.utils.paths import c_state_dir, config_dir, templates_dir


def gen_conf(colours: dict[str, str]) -> str:
    conf = ""
    for name, colour in colours.items():
        conf += f"${name} = {colour}\n"
    return conf


def gen_scss(colours: dict[str, str]) -> str:
    scss = ""
    for name, colour in colours.items():
        scss += f"${name}: #{colour};\n"
    return scss


def gen_replace(colours: dict[str, str], template: Path, hash: bool = False) -> str:
    template = template.read_text()
    for name, colour in colours.items():
        template = template.replace(f"{{{{ ${name} }}}}", f"#{colour}" if hash else colour)
    return template


def c2s(c: str, *i: list[int]) -> str:
    """Hex to ANSI sequence (e.g. ffffff, 11 -> \x1b]11;rgb:ff/ff/ff\x1b\\)"""
    return f"\x1b]{';'.join(map(str, i))};rgb:{c[0:2]}/{c[2:4]}/{c[4:6]}\x1b\\"


def gen_sequences(colours: dict[str, str]) -> str:
    """
    10: foreground
    11: background
    12: cursor
    17: selection
    4:
        0 - 7: normal colours
        8 - 15: bright colours
        16+: 256 colours
    """
    return (
        c2s(colours["onSurface"], 10)
        + c2s(colours["surface"], 11)
        + c2s(colours["secondary"], 12)
        + c2s(colours["secondary"], 17)
        + c2s(colours["term0"], 4, 0)
        + c2s(colours["term1"], 4, 1)
        + c2s(colours["term2"], 4, 2)
        + c2s(colours["term3"], 4, 3)
        + c2s(colours["term4"], 4, 4)
        + c2s(colours["term5"], 4, 5)
        + c2s(colours["term6"], 4, 6)
        + c2s(colours["term7"], 4, 7)
        + c2s(colours["term8"], 4, 8)
        + c2s(colours["term9"], 4, 9)
        + c2s(colours["term10"], 4, 10)
        + c2s(colours["term11"], 4, 11)
        + c2s(colours["term12"], 4, 12)
        + c2s(colours["term13"], 4, 13)
        + c2s(colours["term14"], 4, 14)
        + c2s(colours["term15"], 4, 15)
        + c2s(colours["primary"], 4, 16)
        + c2s(colours["secondary"], 4, 17)
        + c2s(colours["tertiary"], 4, 18)
    )


def write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def apply_terms(sequences: str) -> None:
    state = c_state_dir / "sequences.txt"
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text(sequences)

    pts_path = Path("/dev/pts")
    for pt in pts_path.iterdir():
        if pt.name.isdigit():
            with pt.open("a") as f:
                f.write(sequences)


def apply_hypr(conf: str) -> None:
    write_file(config_dir / "hypr/scheme/current.conf", conf)


def apply_discord(scss: str) -> None:
    import tempfile

    with tempfile.TemporaryDirectory("w") as tmp_dir:
        (Path(tmp_dir) / "_colours.scss").write_text(scss)
        conf = subprocess.check_output(["sass", "-I", tmp_dir, templates_dir / "discord.scss"], text=True)

    for client in "Equicord", "Vencord", "BetterDiscord", "equibop", "vesktop", "legcord":
        write_file(config_dir / client / "themes/caelestia.theme.css", conf)


def apply_spicetify(colours: dict[str, str], mode: str) -> None:
    template = gen_replace(colours, templates_dir / f"spicetify-{mode}.ini")
    write_file(config_dir / "spicetify/Themes/caelestia/color.ini", template)


def apply_fuzzel(colours: dict[str, str]) -> None:
    template = gen_replace(colours, templates_dir / "fuzzel.ini")
    write_file(config_dir / "fuzzel/fuzzel.ini", template)


def apply_btop(colours: dict[str, str]) -> None:
    template = gen_replace(colours, templates_dir / "btop.theme", hash=True)
    write_file(config_dir / "btop/themes/caelestia.theme", template)
    subprocess.run(["killall", "-USR2", "btop"], stderr=subprocess.DEVNULL)


def apply_gtk(colours: dict[str, str], mode: str) -> None:
    template = gen_replace(colours, templates_dir / "gtk.css", hash=True)
    write_file(config_dir / "gtk-3.0/gtk.css", template)
    write_file(config_dir / "gtk-4.0/gtk.css", template)

    subprocess.run(["dconf", "write", "/org/gnome/desktop/interface/gtk-theme", "'adw-gtk3-dark'"])
    subprocess.run(["dconf", "write", "/org/gnome/desktop/interface/color-scheme", f"'prefer-{mode}'"])
    subprocess.run(["dconf", "write", "/org/gnome/desktop/interface/icon-theme", f"'Papirus-{mode.capitalize()}'"])


def apply_qt(colours: dict[str, str], mode: str) -> None:
    template = gen_replace(colours, templates_dir / "qtcolors.conf", hash=True)
    write_file(config_dir / "qt5ct/colors/caelestia.conf", template)
    write_file(config_dir / "qt6ct/colors/caelestia.conf", template)

    qtct = (templates_dir / "qtct.conf").read_text()
    qtct = qtct.replace("{{ $mode }}", mode.capitalize())

    for ver in 5, 6:
        conf = qtct.replace("{{ $config }}", str(config_dir / f"qt{ver}ct"))
        write_file(config_dir / f"qt{ver}ct/qt{ver}ct.conf", conf)


def apply_colours(colours: dict[str, str], mode: str) -> None:
    apply_terms(gen_sequences(colours))
    apply_hypr(gen_conf(colours))
    apply_discord(gen_scss(colours))
    apply_spicetify(colours, mode)
    apply_fuzzel(colours)
    apply_btop(colours)
    apply_gtk(colours, mode)
    apply_qt(colours, mode)
