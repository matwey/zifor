from importlib.metadata import version

try:
    __version__ = version("zifor")
except Exception:
    __version__ = "unknown"