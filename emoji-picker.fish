#!/usr/bin/env fish

set -l chosen_item (cat (dirname (status filename))/data/emojis.txt | fuzzel --dmenu --placeholder='Type to search emojis')
test -n "$chosen_item" && echo "$chosen_item" | cut -d ' ' -f 1 | tr -d '\n' | wl-copy
