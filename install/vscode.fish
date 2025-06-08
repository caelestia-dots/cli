#!/usr/bin/env fish

. (dirname (status filename))/util.fish

install-deps git

set -l dist $C_DATA/vscode

# Update/Clone repo
update-repo vscode $dist

# Install settings
for prog in 'Code' 'Code - OSS' 'VSCodium'
    set -l conf $CONFIG/$prog
    if test -d $conf
        confirm-copy $dist/settings.json $conf/User/settings.json
        confirm-copy $dist/keybindings.json $conf/User/keybindings.json
    end
end

# Install extension
for prog in code code-insiders codium
    if which $prog &> /dev/null
        log "Installing extensions for '$prog'"
        if ! contains 'catppuccin.catppuccin-vsc-icons' ($prog --list-extensions)
            read -l -p "input 'Install catppuccin icons (for light/dark integration)? [Y/n] ' -n" confirm
            test "$confirm" = 'n' -o "$confirm" = 'N' || $prog --install-extension catppuccin.catppuccin-vsc-icons
        end
        $prog --install-extension $dist/caelestia-vscode-integration/caelestia-vscode-integration-*.vsix
    end
end

log 'Done.'
