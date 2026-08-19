"""Contrastive audio decoding: the correction rule, the two model runners, and dispatch.

``contrastive`` holds Eq. (3) and the branch-distance metrics; ``engine`` holds
the model-independent evaluation loop; ``run_qwen`` and ``run_af3`` contribute
only the model-specific loading and input encoding; ``run_parallel`` spreads a
branch sweep across GPUs.
"""
