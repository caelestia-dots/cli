#!/usr/bin/env fish

if ! hyprctl workspaces -j | jq -e 'first(.[] | select(.name == "special:special"))'
    set activews (hyprctl activewindow -j | jq -r '.workspace.name')
    string match -r -- '^special:' $activews && set togglews (string sub -s 9 $activews) || set togglews special
else
    set togglews special
end

hyprctl dispatch togglespecialworkspace $togglews
