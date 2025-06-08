#!/usr/bin/env fish

set -l src (dirname (status filename))

. $src/../util.fish

test -f "$argv[1]" && set -l img (realpath "$argv[1]") || set -l img $C_STATE/wallpaper/thumbnail.jpg

set -l variants vibrant tonalspot expressive fidelity fruitsalad rainbow neutral content monochrome
contains -- "$argv[2]" $variants && set -l variant $argv[2] || set -l variant (cat $C_STATE/scheme/current-variant.txt 2> /dev/null)
contains -- "$variant" $variants || set -l variant tonalspot

set -l hash (sha1sum $img | cut -d ' ' -f 1)

# Cache scheme
if ! test -d $C_CACHE/schemes/$hash/$variant
    set -l colours ($src/score.py $img)
    $src/autoadjust.py dark $variant $colours $C_CACHE/schemes/$hash
    $src/autoadjust.py light $variant $colours $C_CACHE/schemes/$hash
end

# Copy scheme from cache
rm -rf $src/../data/schemes/dynamic
cp -r $C_CACHE/schemes/$hash/$variant $src/../data/schemes/dynamic

# Update if current
set -l variant (string match -gr 'dynamic-(.*)' (cat $C_STATE/scheme/current-name.txt 2> /dev/null))
if test -n "$variant"
    # If variant doesn't exist, use default
    test -d $src/../data/schemes/dynamic/$variant || set -l variant default
    # Apply scheme
    $src/main.fish dynamic $variant $MODE > /dev/null
end
