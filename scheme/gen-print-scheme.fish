#!/usr/bin/env fish

set -l src (dirname (status filename))

. $src/../util.fish

test -f "$argv[1]" && set -l img (realpath "$argv[1]") || set -l img $C_STATE/wallpaper/thumbnail.jpg

# Thumbnail image if not already thumbnail
if test $img != $C_STATE/wallpaper/thumbnail.jpg
    set -l thumb_path $C_CACHE/thumbnails/(sha1sum $img | cut -d ' ' -f 1).jpg
    if ! test -f $thumb_path
        magick -define jpeg:size=256x256 $img -thumbnail 128x128\> $thumb_path
    end
    set img $thumb_path
end

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

# Get mode from image
set -l lightness (magick $img -format '%[fx:int(mean*100)]' info:)
test $lightness -ge 60 && set -l mode light || set -l mode dark

# Print scheme
cat $C_CACHE/schemes/$hash/$variant/default/$mode.txt
