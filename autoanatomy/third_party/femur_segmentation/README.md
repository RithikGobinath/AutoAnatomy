# Vendored: RiccardoBiondi/FemurSegmentation

Source: https://github.com/RiccardoBiondi/FemurSegmentation (MIT license, see
LICENSE in this directory).

Only the pure-Python modules needed for the semi-automatic graph-cut path are
vendored (`IOManager.py`, `utils.py`, `filters.py`, `boneness.py`,
`links.py`) -- not `automatic_segmentation.py`, `metrics.py`,
`peak_detection.py`, or `image_splitter.py`, which pull in extra
dependencies (pandas, matplotlib, scikit-learn) unused by
`autoanatomy.engine.femur_semiauto`.

`lib/GraphCutSupport.cp313-win_amd64.pyd` is a prebuilt Cython/C++ extension
(wraps the BK maxflow/min-cut solver, maxflow-v3.01, GPLv3-compatible use per
upstream) built for this exact machine: Windows x64, CPython 3.13, MinGW-w64
g++, statically linked against the MinGW runtime (no external DLLs needed).
It will NOT load under a different Python version, OS, or architecture --
rebuild it from upstream's `setup.py` in that case. See `build_extension.py`
in this directory for the exact build invocation and the Windows/Python-3.13
workarounds it required.
