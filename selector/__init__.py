"""The adaptive perturbation selector: oracle labels, cached features, head, and training.

The pipeline runs in one direction. ``oracle`` turns decoded branch results
into per-example multi-hot correctness targets; ``cache`` stores the clean
forward pass those targets will be predicted from; ``data`` pairs the two into
tensors; ``model`` and ``train`` fit the head; ``protocol`` pins the specific
candidate pools and reported values the paper names.
"""
