#!/usr/bin/env fish

argparse -n 'caelestia-pip' -X 0 \
    'h/help' \
    'd/daemon' \
    -- $argv
or exit

if set -q _flag_h
    echo 'Usage:'
    echo '    caelestia pip ( -h | --help )'
    echo '    caelestia pip [ -d | --daemon ]'
    echo
    echo 'Options:'
    echo '    -h, --help        Print this help message and exit'
    echo '    -d, --daemon      Run this script in daemon mode'
    echo
    echo 'Normal mode (no args):'
    echo '    Move and resize the active window to picture in picture default geometry.'
    echo
    echo 'Daemon mode:'
    echo '    Set all picture in picture window initial geometry to default.'

    exit
end

. (dirname (status filename))/util.fish

function handle-window -a address workspace
    set -l monitor_id (hyprctl workspaces -j | jq '.[] | select(.name == "'$workspace'").monitorID')
    set -l monitor_size (hyprctl monitors -j | jq -r '.[] | select(.id == '$monitor_id') | "\(.width)\n\(.height)"')
    set -l window_size (hyprctl clients -j | jq '.[] | select(.address == "'$address'").size[]')
    set -l scale_factor (math $monitor_size[2] / 4 / $window_size[2])
    set -l scaled_window_size (math -s 0 $window_size[1] x $scale_factor) (math -s 0 $window_size[2] x $scale_factor)

    hyprctl dispatch "resizewindowpixel exact $scaled_window_size,address:$address" > /dev/null
    hyprctl dispatch "movewindowpixel exact $(math -s 0 $monitor_size[1] x 0.98 - $scaled_window_size[1]) $(math -s 0 $monitor_size[2] x 0.97 - $scaled_window_size[2]),address:$address" > /dev/null
    log "Handled window at address $address"
end

if set -q _flag_d
    log 'Daemon started'
    socat -U - UNIX-CONNECT:$XDG_RUNTIME_DIR/hypr/$HYPRLAND_INSTANCE_SIGNATURE/.socket2.sock | while read line
        switch $line
            case 'openwindow*'
                set -l window (string sub -s 13 $line | string split ',')
                if string match -qr '^(Picture(-| )in(-| )[Pp]icture)$' $window[4]
                    handle-window 0x$window[1] $window[2]
                end
        end
    end

    exit
end

set -l active_window (hyprctl activewindow -j | jq -r '"\(.address)\n\(.workspace.name)\n\(.floating)"')
if test $active_window[3] = true
    handle-window $active_window
else
    warn 'Focused window is not floating, ignoring'
end
