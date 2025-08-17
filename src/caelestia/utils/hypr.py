import json as j
import os
import socket

socket_base = f"{os.getenv('XDG_RUNTIME_DIR')}/hypr/{os.getenv('HYPRLAND_INSTANCE_SIGNATURE')}"
socket_path = f"{socket_base}/.socket.sock"
socket2_path = f"{socket_base}/.socket2.sock"


def message(msg: str, json: bool = True) -> str | dict[str, any]:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.connect(socket_path)

        if json:
            msg = f"j/{msg}"
        sock.send(msg.encode())

        resp = sock.recv(8192).decode()
        while True:
            new_resp = sock.recv(8192)
            if not new_resp:
                break
            resp += new_resp.decode()

        return j.loads(resp) if json else resp


def dispatch(dispatcher: str, *args: list[any]) -> bool:
    return message(f"dispatch {dispatcher} {' '.join(map(str, args))}".rstrip(), json=False) == "ok"


def batch(*msgs: list[str], json: bool = False) -> str | dict[str, any]:
    if json:
        msgs = (f"j/{m.strip()}" for m in msgs)
    return message(f"[[BATCH]]{';'.join(msgs)}", json=False)
