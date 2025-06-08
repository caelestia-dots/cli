#!/usr/bin/env fish

. (dirname (status filename))/util.fish

install-deps git dart-sass aylurs-gtk-shell-git alsa-utils libappindicator-gtk3

# Update/Clone repo
update-repo safeeyes $C_DATA/safeeyes

if which systemctl &> /dev/null
    log 'Installing systemd service...'

    set -l systemd $CONFIG/systemd/user
    mkdir -p $systemd
    echo -n "
[Unit]
Description=Protect your eyes from eye strain using this simple and beautiful, yet extensible break reminder.
After=graphical-session.target

[Service]
Type=exec
ExecStart=/usr/bin/ags run -d $C_DATA/safeeyes
Restart=on-failure
Slice=app-graphical.slice

[Install]
WantedBy=graphical-session.target
" > $systemd/caelestia-safeeyes.service

    systemctl --user daemon-reload
    systemctl --user enable --now caelestia-safeeyes.service
end

log 'Done.'
