#!/usr/bin/env fish

. (dirname (status filename))/util.fish

install-deps git fuzzel-git

set -l dist $CONFIG/fuzzel

# Clone repo
update-repo fuzzel $dist

# Install systemd service
setup-systemd-monitor fuzzel $dist

log 'Done.'
