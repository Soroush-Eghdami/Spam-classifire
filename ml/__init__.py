"""Sift ML package: a framework-agnostic spam-classification pipeline.

The package is intentionally free of any web-framework imports so that the
model can be trained, evaluated, and used from plain Python, Django, or a
notebook.  The public entry points are:

- :mod:`ml.data`     -- loading, cleaning, deduplication, splitting
- :mod:`ml.features` -- feature extraction (word TF-IDF, char TF-IDF, stats)
- :mod:`ml.train`    -- model search, threshold tuning, artifact writing
- :mod:`ml.evaluate` -- metrics and report helpers
- :mod:`ml.predict`  -- loading a saved pipeline and explaining a prediction
"""

__version__ = "2.0.0"
