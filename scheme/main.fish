#!/usr/bin/env fish

# Usage:
#   caelestia scheme <scheme> <flavour> [mode]
#   caelestia scheme <scheme> [flavour]
#   caelestia scheme [scheme]

function set-scheme -a path name mode
    mkdir -p $C_STATE/scheme

    # Update scheme colours
    cp $path $C_STATE/scheme/current.txt

    # Update scheme name
    echo -n $name > $C_STATE/scheme/current-name.txt

    # Update scheme mode
    echo -n $mode > $C_STATE/scheme/current-mode.txt

    log "Changed scheme to $name ($mode)"
end

set -l src (dirname (status filename))/..
set -l schemes $src/data/schemes

. $src/util.fish

set -l scheme $argv[1]
set -l flavour $argv[2]
set -l mode $argv[3]

set -l valid_schemes (basename -a $schemes/*)

test -z "$scheme" && set -l scheme (random choice $valid_schemes)

if contains -- "$scheme" $valid_schemes
    set -l flavours (basename -a (find $schemes/$scheme/ -mindepth 1 -maxdepth 1 -type d) 2> /dev/null)
    set -l modes (basename -s .txt (find $schemes/$scheme/ -mindepth 1 -maxdepth 1 -type f) 2> /dev/null)

    if test -n "$modes"
        # Scheme only has one flavour, so second arg is mode
        set -l mode $flavour
        if test -z "$mode"
            # Try to use current mode if not provided and current mode exists for flavour, otherwise random mode
            set mode (cat $C_STATE/scheme/current-mode.txt 2> /dev/null)
            contains -- "$mode" $modes || set mode (random choice $modes)
        end

        if contains -- "$mode" $modes
            # Provided valid mode
            set-scheme $schemes/$scheme/$mode.txt $scheme $mode
        else
            error "Invalid mode for $scheme: $mode"
        end
    else
        # Scheme has multiple flavours, so second arg is flavour
        test -z "$flavour" && set -l flavour (random choice $flavours)

        if contains -- "$flavour" $flavours
            # Provided valid flavour
            set -l modes (basename -s .txt $schemes/$scheme/$flavour/*.txt)
            if test -z "$mode"
                # Try to use current mode if not provided and current mode exists for flavour, otherwise random mode
                set mode (cat $C_STATE/scheme/current-mode.txt 2> /dev/null)
                contains -- "$mode" $modes || set mode (random choice $modes)
            end

            if contains -- "$mode" $modes
                # Provided valid mode
                set-scheme $schemes/$scheme/$flavour/$mode.txt $scheme-$flavour $mode
            else
                error "Invalid mode for $scheme $flavour: $mode"
            end
        else
            # Invalid flavour
            error "Invalid flavour for $scheme: $flavour"
        end
    end
else
    error "Invalid scheme: $scheme"
end
