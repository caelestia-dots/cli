#!/usr/bin/env sh

# Utility script for rebuilding and running caelestia

cd $(dirname $0) || exit

sudo rm -r dist /usr/bin/caelestia /usr/lib/python3.*/site-packages/caelestia* 2> /dev/null
python -m build --wheel --no-isolation > /dev/null
sudo python -m installer --destdir=/ dist/*.whl > /dev/null

/usr/bin/caelestia "$@"
