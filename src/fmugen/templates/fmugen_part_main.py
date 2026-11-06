"""The process of one isolated part of a composite FMU ([composite] isolate = true).

Copied into resources/parts/<name>/ as main.py, so a part runs, is vendored and is compiled as a whole
FMU's backend is. It runs the part's model with its own Python and packages, and answers the composite
FMU's calls over a local connection (PartProxy in fmugen_runtime). It exits when that connection closes.
"""
import os
import sys
import threading
import traceback
from multiprocessing.connection import Client
from pathlib import Path

from fmugen_runtime import make_engine

#: This part's folder: where its interface.json is (inside the bundle, when compiled).
HERE = Path(getattr(sys, "_MEIPASS", None) or Path(__file__).resolve().parent)


def plain(result):
    """A result without the runtime's Status type, which the other process may know by another module name.

    :param result: what an engine method returned.
    :return: the same, with a status as a plain ``int``.
    """
    if isinstance(result, tuple) and result:
        return (int(result[0]), *result[1:])
    return int(result) if isinstance(result, int) and not isinstance(result, bool) else result


def main():
    """Connect to the composite FMU, make the part's engine, and answer calls until the connection closes.

    :return: the exit code.
    :rtype: int
    """
    connection = Client((sys.argv[1], int(sys.argv[2])), authkey=bytes.fromhex(os.environ["FMUGEN_PART_KEY"]))
    lock = threading.Lock()

    def send(*frame):
        with lock:
            connection.send(frame)

    try:
        engine = make_engine(HERE, lambda message, status, category: send("log", message, int(status), category),
                             os.environ.get("FMUGEN_PART_CATEGORY", "logAll"))
    except BaseException:
        send("exception", traceback.format_exc())
        return 1
    send("ready", {"discard_on": bool(engine.discard_on), "by_reference": engine.by_reference,
                   "logging_on": engine.logging_on})
    while True:
        try:
            kind, name, args = connection.recv()
        except (EOFError, OSError):
            os._exit(0)   # the composite FMU is gone; a thread the model started mustn't keep this process alive
        try:
            if kind == "get":
                result = getattr(engine, name)
            elif kind == "set":
                result = setattr(engine, name, args)
            else:
                result = plain(getattr(engine, name)(*args))
            send("result", result)
        except Exception:
            send("exception", traceback.format_exc())


if __name__ == "__main__":
    sys.exit(main())
