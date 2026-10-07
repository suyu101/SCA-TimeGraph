"""Regression test for import safety of the public package modules."""

import contextlib
import importlib
import io
import pkgutil

import sca


def test_all_package_modules_import_without_optional_runtime_dependencies():
    """Optional baselines/plotting must fail only when explicitly invoked."""
    for module in pkgutil.walk_packages(sca.__path__, sca.__name__ + "."):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            importlib.import_module(module.name)
        assert output.getvalue() == "", f"Import-time output from {module.name}"
