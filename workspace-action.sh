#!/usr/bin/env bash

active_ws=$(hyprctl activeworkspace -j | jq -r '.id')

if [[ "$1" == *"group" ]]; then
    # Move to group
    hyprctl dispatch "${1::-5}" $((($2 - 1) * 10 + ${active_ws:0-1}))
else
    # Move to ws in group
    hyprctl dispatch "$1" $((((active_ws - 1) / 10) * 10 + $2))
fi
