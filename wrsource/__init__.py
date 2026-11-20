"""Revisionable weighted record source and reproducible weighted sampler.

Two independently runnable modules live in this package:

* :mod:`wrsource.source`  -- SQLite backed, versioned source service.
* :mod:`wrsource.dataset` -- reproducible weighted sample / manifest producer.

The implementation intentionally uses only the Python 3.11 standard library.
"""

__all__ = ["source", "sampling", "dataset"]
__version__ = "1.0.0"
