#!/usr/bin/env fish

. (dirname (status filename))/util.fish

install-deps git spicetify-cli spicetify-marketplace-bin

set -l dist $C_DATA/spicetify

# Clone repo
update-repo spicetify $dist

# Install systemd service
setup-systemd-monitor spicetify $dist

# Install theme files
mkdir -p $CONFIG/spicetify/Themes/caelestia
cp $dist/color.ini $CONFIG/spicetify/Themes/caelestia/color.ini
cp $dist/user.css $CONFIG/spicetify/Themes/caelestia/user.css

# Set spicetify theme
spicetify config current_theme caelestia color_scheme caelestia

# Setup marketplace
spicetify config custom_apps marketplace
