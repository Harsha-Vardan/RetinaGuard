# Reuse conftest bootstrap for non-pytest runs
try:
    import conftest  # noqa
except Exception:
    pass
