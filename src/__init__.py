"""Top-level package for the DFU thermogram research pipeline.

The package intentionally keeps imports lightweight so utility modules can be
used without immediately requiring heavy optional dependencies such as
TensorFlow.
"""

__all__ = ["data", "models", "utils"]
