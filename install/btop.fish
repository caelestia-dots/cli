#!/usr/bin/env fish

. (dirname (status filename))/util.fish

install-deps git btop

set -l dist $CONFIG/btop

# Update/Clone repo
update-repo btop $dist
sed -i 's|$SRC|'$dist'|g' $dist/btop.conf

# Install systemd service
setup-systemd-monitor btop $dist

log 'Done.'
