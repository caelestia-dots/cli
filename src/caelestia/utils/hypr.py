import json
import os
import socket
from typing import Any

socket_base = f"{os.getenv('XDG_RUNTIME_DIR')}/hypr/{os.getenv('HYPRLAND_INSTANCE_SIGNATURE')}"
socket_path = f"{socket_base}/.socket.sock"
socket2_path = f"{socket_base}/.socket2.sock"

_lua_config_cache: bool | None = None


def message(msg: str, is_json: bool = True) -> str | dict[str, Any]:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.connect(socket_path)

        if is_json:
            msg = f"j/{msg}"
        sock.send(msg.encode())

        resp = sock.recv(8192).decode()
        while True:
            new_resp = sock.recv(8192)
            if not new_resp:
                break
            resp += new_resp.decode()

        return json.loads(resp) if is_json else resp


def is_lua_config() -> bool:
    global _lua_config_cache
    if _lua_config_cache is not None:
        return _lua_config_cache
    try:
        result = message("status")
        if isinstance(result, dict):
            _lua_config_cache = bool(result["configProvider"] == "lua")
        else:
            _lua_config_cache = False
        return _lua_config_cache
    except Exception:
        _lua_config_cache = False
        return False


DISPATCHER_MAP_LUA = {
    "togglespecialworkspace": lambda *a: (
        f'hl.dsp.workspace.toggle_special("{a[0]}")' if a else "hl.dsp.workspace.toggle_special()"
    ),
    "movetoworkspacesilent": lambda *a: (
        f'hl.dsp.window.move({{window = "address:{a[0].split(",")[1].replace("address:", "")}", workspace = "{a[0].split(",")[0]}", follow = false}})'
    ),
    "exec": lambda *a: 'hl.dsp.exec_cmd("' + " ".join(a).replace("\\", "\\\\").replace('"', '\\"') + '")',
}


def dispatch(dispatcher: str, *args: str) -> bool:
    if is_lua_config() and dispatcher in DISPATCHER_MAP_LUA:
        lua_dispatch = DISPATCHER_MAP_LUA[dispatcher](*args)
        return message(f"dispatch {lua_dispatch}", is_json=False) == "ok"
    return message(f"dispatch {dispatcher} {' '.join(map(str, args))}".rstrip(), is_json=False) == "ok"


def batch(*msgs: str, is_json: bool = False) -> str | dict[str, Any]:
    formatted_msgs = msgs

    if is_json:
        formatted_msgs = [f"j/{m.strip()}" for m in msgs]

    return message(f"[[BATCH]]{';'.join(formatted_msgs)}", is_json=False)
