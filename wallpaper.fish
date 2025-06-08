#!/usr/bin/env fish

function get-valid-wallpapers
    identify -ping -format '%i\n' $wallpapers_dir/** 2> /dev/null
end

set script_name (basename (status filename))
set wallpapers_dir (xdg-user-dir PICTURES)/Wallpapers
set threshold 80

# Max 0 non-option args | h, f and d are exclusive | F and t are also exclusive
argparse -n 'caelestia-wallpaper' -X 0 -x 'h,f,d' -x 'F,t' \
    'h/help' \
    'f/file=' \
    'd/directory=' \
    'F/no-filter' \
    't/threshold=!_validate_int --min 0' \
    'T/theme=!test $_flag_value = light -o $_flag_value = dark' \
    -- $argv
or exit

. (dirname (status filename))/util.fish

if set -q _flag_h
    echo 'Usage:'
    echo '    caelestia wallpaper'
    echo '    caelestia wallpaper [ -h | --help ]'
    echo '    caelestia wallpaper [ -f | --file ] [ -T | --theme ]'
    echo '    caelestia wallpaper [ -d | --directory ] [ -F | --no-filter ] [ -T | --theme ]'
    echo '    caelestia wallpaper [ -d | --directory ] [ -t | --threshold ] [ -T | --theme ]'
    echo
    echo 'Options:'
    echo '    -h, --help                        Print this help message and exit'
    echo '    -f, --file <file>                 The file to change wallpaper to'
    echo '    -d, --directory <directory>       The folder to select a random wallpaper from (default '$wallpapers_dir')'
    echo '    -F, --no-filter                   Do not filter by size'
    echo '    -t, --threshold <threshold>       The minimum percentage of the size the image must be greater than to be selected (default '$threshold')'
    echo '    -T, --theme <"light" | "dark">    Set light/dark theme for dynamic scheme'
else
    set state_dir $C_STATE/wallpaper

    # The path to the last chosen wallpaper
    set last_wallpaper_path "$state_dir/last.txt"

    # Use wallpaper given as argument else choose random
    if set -q _flag_f
        set chosen_wallpaper (realpath $_flag_f)

        if ! identify -ping $chosen_wallpaper &> /dev/null
            error "$chosen_wallpaper is not a valid image"
            exit 1
        end
    else
        # The path to the directory containing the selection of wallpapers
        set -q _flag_d && set wallpapers_dir (realpath $_flag_d)

        if ! test -d $wallpapers_dir
            error "$wallpapers_dir does not exist"
            exit 1
        end

        # Get all files in $wallpapers_dir and exclude the last wallpaper (if it exists)
        if test -f "$last_wallpaper_path"
            set last_wallpaper (cat $last_wallpaper_path)
            test -n "$last_wallpaper" && set unfiltered_wallpapers (get-valid-wallpapers | grep -v $last_wallpaper)
        end
        set -q unfiltered_wallpapers || set unfiltered_wallpapers (get-valid-wallpapers)

        # Filter by resolution if no filter option is not given
        if set -q _flag_F
            set wallpapers $unfiltered_wallpapers
        else
            set -l screen_size (hyprctl monitors -j | jq -r 'max_by(.width * .height) | "\(.width)\n\(.height)"')
            set -l wall_sizes (identify -ping -format '%w %h\n' $unfiltered_wallpapers)

            # Apply threshold
            set -q _flag_t && set threshold $_flag_t
            set screen_size[1] (math $screen_size[1] x $threshold / 100)
            set screen_size[2] (math $screen_size[2] x $threshold / 100)

            # Add wallpapers that are larger than the screen size * threshold to list to choose from ($wallpapers)
            for i in (seq 1 (count $wall_sizes))
                set -l wall_size (string split ' ' $wall_sizes[$i])
                if test $wall_size[1] -ge $screen_size[1] -a $wall_size[2] -ge $screen_size[2]
                    set -a wallpapers $unfiltered_wallpapers[$i]
                end
            end
        end

        # Check if the $wallpapers list is unset or empty
        if ! set -q wallpapers || test -z "$wallpapers"
            error "No valid images found in $wallpapers_dir"
            exit 1
        end

        # Choose a random wallpaper from the $wallpapers list
        set chosen_wallpaper (random choice $wallpapers)
    end

    # Thumbnail wallpaper for colour gen
    mkdir -p $C_CACHE/thumbnails
    set -l thumb_path $C_CACHE/thumbnails/(sha1sum $chosen_wallpaper | cut -d ' ' -f 1).jpg
    if ! test -f $thumb_path
        magick -define jpeg:size=256x256 $chosen_wallpaper -thumbnail 128x128 $thumb_path
    end
    cp $thumb_path $state_dir/thumbnail.jpg

    # Light/dark mode detection if not specified
    if ! set -q _flag_T
        set -l lightness (magick $state_dir/thumbnail.jpg -format '%[fx:int(mean*100)]' info:)
        test $lightness -ge 60 && set _flag_T light || set _flag_T dark
    end

    # Generate colour scheme for wallpaper
    set -l src (dirname (status filename))
    MODE=$_flag_T $src/scheme/gen-scheme.fish &

    # Store the wallpaper chosen
    mkdir -p $state_dir
    echo $chosen_wallpaper > $last_wallpaper_path
    ln -sf $chosen_wallpaper "$state_dir/current"
end
