"""Shared infrastructure: configuration, dataset I/O, result schema, and runtime helpers.

Nothing in this package imports Torch or Transformers at module level, so the
offline analysis stages (split construction, table regeneration) can run on a
machine without a GPU or a model installed.
"""
