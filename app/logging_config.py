"""Process-wide logging setup (like configuring ILogger providers once in Program.cs)."""

import logging


def configure_logging(level: str = "INFO") -> None:
    """Send app logs to stderr with timestamps. Safe to call more than once.

    Hosts such as Streamlit Cloud, Docker and Azure Container Apps collect stderr automatically,
    so no file handlers are needed.
    """
    root = logging.getLogger("app")
    if root.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root.addHandler(handler)
    root.setLevel(level)
    root.propagate = False
