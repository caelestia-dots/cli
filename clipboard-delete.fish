#!/usr/bin/env fish

set -l chosen_item (cliphist list | fuzzel --dmenu --prompt='del > ' --placeholder='Delete from clipboard')
test -n "$chosen_item" && echo "$chosen_item" | cliphist delete
