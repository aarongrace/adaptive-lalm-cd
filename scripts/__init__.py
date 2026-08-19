"""Command-line stages that run the experiment pipeline end to end.

Each module is a thin entry point over the packages above, invoked as
``python -m scripts.<name>`` from the project root. In dependency order:
``build_oracle`` -> ``prepare_splits`` -> ``cache_hidden_states`` ->
``train_selector``, with ``summarize_results`` runnable any time after
decoding.
"""
