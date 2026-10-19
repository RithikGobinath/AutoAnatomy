"""Rebuild lib/GraphCutSupport.*.pyd from native_src/ -- needed whenever the
prebuilt extension doesn't match the running Python version/OS/architecture
(it will fail to import with a version/platform mismatch, not silently
misbehave).

Usage (from this directory, with a C++ compiler on PATH -- MinGW-w64's g++
on Windows, or your system's g++/clang on Linux/macOS):

    python build_extension.py

On Windows this requires two real workarounds for bugs this exact codebase
hit while building against setuptools' vendored distutils on Python 3.13
(both root-caused by actually reading the setuptools source, not guessed):

1. customize_compiler() (distutils/sysconfig.py) unconditionally reads
   POSIX-only Makefile config vars (CC, LDSHARED, ...) for compiler_type ==
   "mingw32", which are always None on a native Windows Python build.
   Mingw32CCompiler already sets up its own compiler executables directly in
   its own __init__, so this customization was never needed for mingw32 in
   the first place -- skipped below for that compiler_type only.

2. setuptools.__init__._install_setup_requires()'s local MinimalDistribution
   class calls self.set_defaults._disable(), but distutils.core.Distribution
   (setuptools' own vendored _distutils.dist.Distribution, since stdlib
   distutils no longer exists on Python 3.13) doesn't expose set_defaults as
   an object with a _disable() method -- a genuine internal inconsistency in
   this setuptools release. All of setup_requires' dependencies (Cython,
   numpy) must already be installed independently (this project's own
   pyproject.toml pulls them in), so skipping this step is safe.

Import ordering matters for both patches: everything below is imported
through the `distutils.*` alias namespace (never `setuptools._distutils.*`
directly). setuptools' _distutils_hack builds `distutils` as an
INDEPENDENT module tree from `setuptools._distutils` -- their Command
classes have different identities, so mixing the two makes Cython's
`build_ext` subclass fail an `issubclass(cmd_class, Command)` check deep in
distutils' own command-line parsing (confirmed by direct testing). setup.py
and Cython.Distutils only ever import through `distutils.*`, so this patch
has to target that same tree to actually take effect.

Also forces GCC-style compile flags when --compiler=mingw32 is used, since
platform.python_compiler() reports how the Python *interpreter* was built
(MSVC, on the official Windows builds) rather than the compiler actually
building this extension.
"""
import os
import subprocess
import sys

import setuptools  # import first -- installs the distutils_hack shim

if os.name == "nt":
    import distutils.sysconfig as _sysconfig_mod
    import distutils.command.build_ext as _build_ext_mod

    _original_customize_compiler = _sysconfig_mod.customize_compiler

    def _patched_customize_compiler(compiler):
        if compiler.compiler_type == "mingw32":
            return compiler
        return _original_customize_compiler(compiler)

    _sysconfig_mod.customize_compiler = _patched_customize_compiler
    _build_ext_mod.customize_compiler = _patched_customize_compiler

    setuptools._install_setup_requires = lambda attrs: None


if __name__ == "__main__":
    here = os.path.abspath(os.path.dirname(__file__))
    args = ["build_ext", "--inplace"]
    if os.name == "nt":
        args.append("--compiler=mingw32")
    sys.argv = ["setup.py"] + args
    os.chdir(here)
    import runpy
    runpy.run_path(os.path.join(here, "setup.py"), run_name="__main__")
