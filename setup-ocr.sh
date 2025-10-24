#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

find_python() {
    if command -v python3 >/dev/null 2>&1; then
        echo "python3"
    elif command -v python >/dev/null 2>&1; then
        echo "python"
    else
        exit 1
    fi
}

PYTHON_BIN=$(find_python)

if ! "${PYTHON_BIN}" -m pip --version >/dev/null 2>&1; then
    exit 1
fi

missing_pkgs=$("${PYTHON_BIN}" - <<'PY'
import importlib
modules = {
    "rapidocr_onnxruntime": "rapidocr-onnxruntime",
    "onnxruntime": "onnxruntime",
    "numpy": "numpy",
    "PyQt6": "PyQt6",
    "threadpoolctl": "threadpoolctl",
}
missing = []
for module, pkg in modules.items():
    try:
        importlib.import_module(module)
    except Exception:
        missing.append(pkg)

if missing:
    print(" ".join(missing))
PY
)

if [[ -n "${missing_pkgs}" ]]; then
    "${PYTHON_BIN}" -m pip install --user ${missing_pkgs}
fi

SYSTEMD_USER_DIR="$HOME/.config/systemd/user"
mkdir -p "$SYSTEMD_USER_DIR"

SERVICE_FILE="${SCRIPT_DIR}/systemd/caelestia-ocrd.service"
if [[ -f "${SERVICE_FILE}" ]]; then
    cp "${SERVICE_FILE}" "${SYSTEMD_USER_DIR}/"
    systemctl --user daemon-reload
    systemctl --user enable caelestia-ocrd.service >/dev/null 2>&1 || true
    systemctl --user restart caelestia-ocrd.service || systemctl --user start caelestia-ocrd.service
fi

CONFIG_DIR="$HOME/.config/caelestia"
mkdir -p "$CONFIG_DIR"

OCR_CONFIG="$CONFIG_DIR/ocr.json"

"${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

config_path = Path(os.path.expanduser("~/.config/caelestia/ocr.json"))

DEFAULT = {
    "provider": "cpu-ort",
    "downscale": 0.6,
    "tiles": 1,
    "max_boxes": 300,
    "use_gpu": False,
    "warm_start": True,
    "performance": {
        "idle_threads": 1,
        "standard_threads": 0,
        "fast_threads": 0,
        "idle_cores": 1,
        "standard_cores": 0,
        "fast_cores": 0,
    },
}

if config_path.exists():
    try:
        data = json.loads(config_path.read_text())
    except Exception:
        data = {}
else:
    data = {}

def deep_fill(default, target):
    for key, value in default.items():
        if isinstance(value, dict):
            existing = target.get(key)
            if not isinstance(existing, dict):
                existing = {}
            target[key] = existing
            deep_fill(value, existing)
        else:
            target.setdefault(key, value)

deep_fill(DEFAULT, data)

config_path.parent.mkdir(parents=True, exist_ok=True)
config_path.write_text(json.dumps(data, indent=2))
PY
