#!/usr/bin/env fish

. (dirname (status filename))/util.fish

mkdir -p "$C_CACHE/screenshots"
set -l tmp_file "$C_CACHE/screenshots/$(date +'%Y%m%d%H%M%S')"

if test "$argv[1]" = 'region'
    if test "$argv[2]" = 'freeze'
        wayfreeze --hide-cursor & set PID $last_pid
        sleep .1
    end

    set -l ws (hyprctl -j activeworkspace | jq -r '.id')
    set -l region (hyprctl -j clients | jq -r --argjson activeWsId $ws '.[] | select(.workspace.id == $activeWsId) | "\(.at[0]),\(.at[1]) \(.size[0])x\(.size[1])"' | slurp)
    if test -n "$region"
        grim -l 0 -g $region - | swappy -f - &
    end

    set -q PID && kill $PID

    exit
end

grim $argv $tmp_file; and wl-copy < $tmp_file; or exit 1

set -l action (notify-send -i 'image-x-generic-symbolic' -h "STRING:image-path:$tmp_file" \
    -a 'caelestia-screenshot' --action='open=Open' --action='save=Save' \
        'Screenshot taken' "Screenshot stored in $tmp_file and copied to clipboard")
switch $action
    case 'open'
        app2unit -- swappy -f $tmp_file & disown
    case 'save'
        set -l save_file (app2unit -- zenity --file-selection --save --title='Save As')
        test -z $save_file && exit 0
        if test -f $save_file
            app2unit -- yad --image='abrt' --title='Warning!' --text-align='center' --buttons-layout='center' --borders=20 \
                --text='<span size="x-large">Are you sure you want to overwrite this file?</span>' || exit 0
        end
        cp $tmp_file $save_file
end
