"""Builds lib/GraphCutSupport.*.pyd (or .so) from native_src/. Run via
build_extension.py, not directly -- that script installs the Windows/
Python-3.13 workarounds this build needs (see its docstring).

Adapted from upstream RiccardoBiondi/FemurSegmentation's setup.py, trimmed
to only the single extension this vendored subset needs.
"""
import os
import platform

import numpy as np
from setuptools import Extension, setup
from Cython.Distutils import build_ext

here = os.path.abspath(os.path.dirname(__file__))
native_src = os.path.join(here, "native_src")

CPP_COMPILER = platform.python_compiler()
if any("mingw32" in arg for arg in os.sys.argv):
    # platform.python_compiler() reports how the Python interpreter itself
    # was built (MSVC on official Windows builds), not the compiler this
    # extension is actually being built with.
    CPP_COMPILER = "GCC (forced: --compiler=mingw32 was requested)"

extra_link_args = []
if "GCC" in CPP_COMPILER or "Clang" in CPP_COMPILER:
    compiler_args = [
        "-std=c++11", "-std=gnu++11", "-g0",
        "-Wno-unused-function", "-Wno-narrowing", "-Wall", "-Wextra",
        "-Wno-unused-result", "-Wno-unknown-pragmas", "-Wfatal-errors",
        "-Wpedantic", "-Wno-write-strings", "-Wno-overflow", "-Wno-parentheses",
        "-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION",
    ]
    if os.name == "nt":
        # Statically link the MinGW runtime so the built .pyd doesn't need
        # libstdc++-6.dll/libgcc_s_seh-1.dll/libwinpthread-1.dll present (or
        # PATH-discoverable -- Python's DLL loader no longer honors PATH by
        # default since 3.8) on whichever machine actually runs it.
        extra_link_args = ["-static-libgcc", "-static-libstdc++", "-static"]
elif "MSC" in CPP_COMPILER:
    compiler_args = ["/std:c++11", "-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION"]
else:
    raise ValueError(f"Unknown C++ compiler: {CPP_COMPILER!r}")


class _build_ext(build_ext):
    def build_extensions(self):
        try:
            self.compiler.compiler_so.remove("-Wstrict-prototypes")
            self.compiler.compiler_so.remove("-Wdate-time")
        except (AttributeError, ValueError):
            pass
        build_ext.build_extensions(self)


ext_modules = [
    Extension(
        name="lib.GraphCutSupport",
        sources=[
            os.path.join(native_src, "maxflow-v3.01", "graph.cpp"),
            os.path.join(native_src, "maxflow-v3.01", "maxflow.cpp"),
            os.path.join(native_src, "src", "graphcut.pyx"),
        ],
        include_dirs=[
            np.get_include(),
            os.path.join(native_src, "include"),
            os.path.join(native_src, "maxflow-v3.01"),
        ],
        extra_compile_args=compiler_args,
        extra_link_args=extra_link_args,
        language="c++",
    )
]

setup(
    name="femur-graphcut-extension",
    version="0.0.0",
    ext_modules=ext_modules,
    cmdclass={"build_ext": _build_ext},
)
