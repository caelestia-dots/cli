#!/usr/bin/env fish

. (dirname (status filename))/util.fish

install-deps git inotify-tools

set -l dist $C_DATA/firefox

# Update/Clone repo
update-repo firefox $dist

# Install native app manifest
for dev in mozilla zen
    if test -d $HOME/.$dev
        mkdir -p $HOME/.$dev/native-messaging-hosts
        cp $dist/native_app/manifest.json $HOME/.$dev/native-messaging-hosts/caelestiafox.json
        sed -i "s|\$SRC|$dist|g" $HOME/.$dev/native-messaging-hosts/caelestiafox.json
    end
end

# Install zen css
if test -d $HOME/.zen
    for profile in $HOME/.zen/*/chrome
        for file in userChrome userContent
            if test -f $profile/$file.css
                set -l imp "@import url('$dist/zen/$file.css');"
                grep -qFx $imp $profile/$file.css || printf '%s\n%s' $imp "$(cat $profile/$file.css)" > $profile/$file.css
            else
                echo "@import url('$dist/zen/$file.css');" > $profile/$file.css
            end
        end
    end
end

log 'Done.'
log 'Please install the extension manually from https://addons.mozilla.org/en-US/firefox/addon/caelestiafox'
