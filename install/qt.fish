#!/usr/bin/env fish

. (dirname (status filename))/util.fish

install-deps git darkly-bin
install-optional-deps 'papirus-icon-theme (icon theme)'

set -l dist $C_DATA/qt

# Update/Clone repo
update-repo qt $dist

# Install systemd service
setup-systemd-monitor qt $dist

# Change settings
confirm-copy $dist/qtct.conf $CONFIG/qt5ct/qt5ct.conf
confirm-copy $dist/qtct.conf $CONFIG/qt6ct/qt6ct.conf

log 'Done.'
