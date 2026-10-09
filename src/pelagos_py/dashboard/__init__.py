"""The pelagos_py config dashboard: ``from pelagos_py import dashboard; dashboard.run()``."""


def run(port=8791):
    """Open the config dashboard in your browser. Needs ``pip install "pelagos_py[dashboard]"``."""
    try:
        import fastapi  # noqa: F401
        import uvicorn  # noqa: F401
    except ImportError:
        raise ImportError(
            'The dashboard needs extra packages: pip install "pelagos_py[dashboard]"'
        ) from None
    from pelagos_py.dashboard.app import serve

    serve(port)
