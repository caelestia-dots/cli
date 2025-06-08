#!/usr/bin/env fish

set -l chosen_item (cliphist list | fuzzel --dmenu --placeholder='Type to search clipboard')
test -n "$chosen_item" && echo "$chosen_item" | cliphist decode | wl-copy
