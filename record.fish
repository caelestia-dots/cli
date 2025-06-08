#!/usr/bin/env fish

function get-audio-source
    pactl list short sources | grep '\.monitor.*RUNNING' | cut -f 2 | head -1
end

function get-region
    slurp || exit 0
end

function get-active-monitor
    hyprctl monitors -j | jq -r '.[] | select(.focused == true) | .name'
end

argparse -n 'caelestia-record' -X 0 \
    'h/help' \
    's/sound' \
    'r/region=?' \
    'n/no-hwaccel' \
    -- $argv
or exit

if set -q _flag_h
    echo 'Usage:'
    echo '    caelestia record ( -h | --help )'
    echo '    caelestia record [ -s | --sound ] [ -r | --region ] [ -c | --compression ] [ -H | --hwaccel ]'
    echo
    echo 'Options:'
    echo '    -h, --help                    Print this help message and exit'
    echo '    -s, --sound                   Enable audio capturing'
    echo '    -r, --region [ <region> ]     The region to capture, current monitor if option not given, default region slurp'
    echo '    -N, --no-hwaccel              Do not use the GPU encoder'

    exit
end

. (dirname (status filename))/util.fish

set -l storage_dir (xdg-user-dir VIDEOS)/Recordings
set -l state_dir $C_STATE/record

mkdir -p $storage_dir
mkdir -p $state_dir

set -l file_ext 'mp4'
set -l recording_path "$state_dir/recording.$file_ext"
set -l notif_id_path "$state_dir/notifid.txt"

if pgrep wl-screenrec > /dev/null
    pkill wl-screenrec

    # Move to recordings folder
    set -l new_recording_path "$storage_dir/recording_$(date '+%Y%m%d_%H-%M-%S').$file_ext"
    mv $recording_path $new_recording_path

    # Close start notification
    if test -f $notif_id_path
        gdbus call --session \
            --dest org.freedesktop.Notifications \
            --object-path /org/freedesktop/Notifications \
            --method org.freedesktop.Notifications.CloseNotification \
            (cat $notif_id_path)
    end

    # Notification with actions
    set -l action (notify-send 'Recording stopped' "Stopped recording $new_recording_path" -i 'video-x-generic' -a 'caelestia-record' \
        --action='watch=Watch' --action='open=Open' --action='save=Save As' --action='delete=Delete')

    switch $action
        case 'watch'
            app2unit -O $new_recording_path
        case 'open'
        	dbus-send --session --dest=org.freedesktop.FileManager1 --type=method_call /org/freedesktop/FileManager1 org.freedesktop.FileManager1.ShowItems array:string:"file://$new_recording_path" string:'' \
                || app2unit -O (dirname $new_recording_path)
        case 'save'
        	set -l save_file (app2unit -- zenity --file-selection --save --title='Save As')
        	test -n "$save_file" && mv $new_recording_path $save_file || warn 'No file selected'
        case 'delete'
            rm $new_recording_path
    end
else
    # Set region if flag given otherwise active monitor
    if set -q _flag_r
        # Use given region if value otherwise slurp
        set region -g (test -n "$_flag_r" && echo $_flag_r || get-region)
    else
        set region -o (get-active-monitor)
    end

    # Sound if enabled
    set -q _flag_s && set -l audio --audio --audio-device (get-audio-source)

    # No hardware accel
    set -q _flag_n && set -l hwaccel --no-hw

    wl-screenrec $region $audio $hwaccel --codec hevc -f $recording_path & disown

    notify-send 'Recording started' 'Recording...' -i 'video-x-generic' -a 'caelestia-record' -p > $notif_id_path
end
