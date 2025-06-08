#!/usr/bin/env fish

. (dirname (status filename))/util.fish

install-deps git slurp

set -l dist $C_DATA/slurp

# Clone repo
update-repo slurp $dist

# Install systemd service
setup-systemd-monitor slurp $dist

# Install to path
install-link $dist/slurp ~/.local/bin/slurp

log 'Done.'
