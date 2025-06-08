#!/usr/bin/env fish

. (dirname (status filename))/util.fish

if ! pacman -Q lm_sensors > /dev/null
    sudo pacman -S --noconfirm lm_sensors
    sudo sensors-detect --auto
end

install-deps git quickshell curl jq ttf-material-symbols-variable-git ttf-jetbrains-mono-nerd ttf-ibm-plex app2unit-git fd fish python-aubio python-pyaudio python-numpy cava networkmanager bluez-utils ddcutil brightnessctl
install-optional-deps 'uwsm (for systems using uwsm)'

set -l shell $C_DATA/shell

# Update/Clone repo
update-repo shell $shell

if which systemctl &> /dev/null
    log 'Installing systemd service...'

    set -l systemd $CONFIG/systemd/user
    mkdir -p $systemd
    echo -n "
[Unit]
Description=A very segsy desktop shell.
After=graphical-session.target

[Service]
Type=exec
ExecStart=$shell/run.fish
Restart=on-failure
Slice=app-graphical.slice

[Install]
WantedBy=graphical-session.target
" > $systemd/caelestia-shell.service

    systemctl --user daemon-reload
    systemctl --user enable --now caelestia-shell.service
end

log 'Done.'
