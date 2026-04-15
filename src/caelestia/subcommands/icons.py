from argparse import Namespace
import json
from pathlib import Path
import subprocess


# Paths

CONFIG_FILE_PATH = Path.home() / '.config/qtengine/config.json'
ICONS_PATH = Path('/usr/share/icons')


# List icons function

def list_icons():
    return [it.name for it in ICONS_PATH.iterdir() if it.is_dir()]


# Set icon function

def set_icon(icon_theme: str):

    av_icons = list_icons()

    if icon_theme not in av_icons:
        raise ValueError('Icon theme not found.')

    if not CONFIG_FILE_PATH.exists():
        raise FileNotFoundError('QTEngine config file not found.')

    with open(CONFIG_FILE_PATH, 'r') as config:
        settings = json.load(config)
    
    settings['theme']['iconTheme'] = icon_theme

    with open(CONFIG_FILE_PATH, 'w') as config:
        json.dump(settings, config, indent=4)


    subprocess.run([
        'gsettings',
        'set',
        'org.gnome.desktop.interface',
        'icon-theme',
        icon_theme
])


# Command class setting

class Command:

    def __init__(self, args: Namespace):
        self.args = args

    def run(self):

        if self.args.list:
            for icon in list_icons():
                print(icon)

        elif self.args.set:
            set_icon(self.args.set)

        else:
            print('use --list or --set <icon-theme>')
