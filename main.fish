#!/usr/bin/env fish

set -l src (dirname (realpath (status filename)))

. $src/util.fish

if test "$argv[1]" = shell
    # Start shell if no args
    if test -z "$argv[2..]"
        $C_DATA/shell/run.fish
    else
        if test "$argv[2]" = help
            qs -p $C_DATA/shell ipc show
            exit
        end

        if qs list --all | grep "Config path: $C_DATA/shell/shell.qml" &> /dev/null
            qs -p $C_DATA/shell ipc call $argv[2..]
        else
            warn 'Shell unavailable'
        end
    end
    exit
end

if test "$argv[1]" = toggle
    set -l valid_toggles communication music sysmon specialws todo
    if contains -- "$argv[2]" $valid_toggles
        if test $argv[2] = specialws
            $src/toggles/specialws.fish
        else
            . $src/toggles/util.fish
            toggle-workspace $argv[2]
        end
    else
        error "Invalid toggle: $argv[2]"
    end

    exit
end

if test "$argv[1]" = workspace-action
    $src/workspace-action.sh $argv[2..]
    exit
end

if test "$argv[1]" = scheme
    if test "$argv[2]" = print
        $src/scheme/gen-print-scheme.fish $argv[3..]
    else
        $src/scheme/main.fish $argv[2..]
    end
    exit
end

if test "$argv[1]" = variant
    set -l variants vibrant tonalspot expressive fidelity fruitsalad rainbow neutral content monochrome
    if contains -- "$argv[2]" $variants
        echo -n $argv[2] > $C_STATE/scheme/current-variant.txt
        $src/scheme/gen-scheme.fish
    else
        error "Invalid variant: $argv[2]"
    end

    exit
end

if test "$argv[1]" = install
    set -l valid_modules scripts btop discord firefox fish foot fuzzel hypr safeeyes shell slurp spicetify gtk qt vscode
    if test "$argv[2]" = all
        for module in $valid_modules
            $src/install/$module.fish $argv[3..]
        end
    else
        contains -- "$argv[2]" $valid_modules && $src/install/$argv[2].fish $argv[3..] || error "Invalid module: $argv[2]"
    end
    test -f $C_STATE/scheme/current.txt || $src/scheme/main.fish onedark  # Init scheme after install or update
    exit
end

set -l valid_subcommands screenshot record clipboard clipboard-delete emoji-picker wallpaper pip

if contains -- "$argv[1]" $valid_subcommands
    $src/$argv[1].fish $argv[2..]
    exit
end

test "$argv[1]" != help && error "Unknown command: $argv[1]"

echo 'Usage: caelestia COMMAND [ ...args ]'
echo
echo 'COMMAND := help | install | shell | toggle | workspace-action | scheme | screenshot | record | clipboard | clipboard-delete | emoji-picker | wallpaper | pip'
echo
echo '  help: show this help message'
echo '  install: install a module'
echo '  shell: start the shell or message it'
echo '  toggle: toggle a special workspace'
echo '  workspace-action: execute a Hyprland workspace dispatcher in the current group'
echo '  scheme: change the current colour scheme'
echo '  variant: change the current scheme variant'
echo '  screenshot: take a screenshot'
echo '  record: take a screen recording'
echo '  clipboard: open clipboard history'
echo '  clipboard-delete: delete an item from clipboard history'
echo '  emoji-picker: open the emoji picker'
echo '  wallpaper: change the wallpaper'
echo '  pip: move the focused window into picture in picture mode or start the pip daemon'

# Set exit status
test "$argv[1]" = help
exit
