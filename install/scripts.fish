#!/usr/bin/env fish

. (dirname (status filename))/util.fish

install-deps git hyprland-git hyprpaper-git imagemagick wl-clipboard fuzzel-git socat foot jq python xdg-user-dirs python-materialyoucolor-git app2unit-git grim wayfreeze-git wl-screenrec swappy
install-optional-deps 'discord (messaging app)' 'btop (system monitor)' 'zen-browser (web browser)'

set -l dist $C_DATA/scripts

# Update/Clone repo
update-repo scripts $dist

# Install to path
install-link $dist/main.fish ~/.local/bin/caelestia

# Install completions
test -e $CONFIG/fish/completions/caelestia.fish && rm $CONFIG/fish/completions/caelestia.fish
mkdir -p $CONFIG/fish/completions
cp $dist/completions/caelestia.fish $CONFIG/fish/completions/caelestia.fish

log 'Done.'
