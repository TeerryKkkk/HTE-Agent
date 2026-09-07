"""Network denial for reproducible offline workflows (including DNS and UDP)."""
from contextlib import contextmanager, ExitStack
import os
import socket
from unittest.mock import patch


def deny_network(*args, **kwargs):
    raise AssertionError("Network access is disabled for this offline run")


@contextmanager
def offline_execution():
    with ExitStack() as stack:
        stack.enter_context(patch.dict(os.environ, {"OPENROUTER_API_KEY": ""}))
        for target in ("connect", "connect_ex", "sendto"):
            stack.enter_context(patch.object(socket.socket, target, deny_network))
        for target in ("create_connection", "getaddrinfo"):
            stack.enter_context(patch.object(socket, target, deny_network))
        yield
