#!/usr/bin/env fish

. (dirname (status filename))/util.fish

install-deps git uwsm hyprland-git hyprpaper-git hyprlock-git hypridle-git polkit-gnome gnome-keyring wl-clipboard wireplumber app2unit-git
install-optional-deps 'gammastep (night light)' 'wlogout (secondary session menu)' 'grimblast-git (screenshot freeze)' 'hyprpicker-git (colour picker)' 'foot (terminal emulator)' 'firefox (web browser)' 'vscodium-bin (IDE)' 'thunar (file manager)' 'nemo (secondary file manager)' 'fuzzel (secondary app launcher)' 'ydotool (alternate paste)' 'trash-cli (auto trash)'

set -l hypr $CONFIG/hypr

# Cause hyprland autogenerates a config file when it is removed
set -l remote https://github.com/caelestia-dots/hypr.git
if test -d $hypr
    cd $hypr || exit
    if test "$(git config --get remote.origin.url)" != $remote
        cd .. || exit
        confirm-overwrite $hypr dummy
        git clone $remote /tmp/caelestia-hypr
        rm -rf $hypr && mv /tmp/caelestia-hypr $hypr
    else
        git pull
    end
else
    git clone $remote $dir
end

# Install uwsm envs
install-link $hypr/uwsm $CONFIG/uwsm

# Enable ydotool if installed
pacman -Q ydotool &> /dev/null && systemctl --user enable --now ydotool.service

log 'Done.'
