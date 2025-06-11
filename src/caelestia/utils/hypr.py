import json as j
import os
import socket

socket_path = f"{os.getenv('XDG_RUNTIME_DIR')}/hypr/{os.getenv('HYPRLAND_INSTANCE_SIGNATURE')}/.socket.sock"


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
    return message(f"dispatch {dispatcher} {' '.join(str(a) for a in args)}".rstrip(), json=False) == "ok"
