from __future__ import annotations

import math


def perplexity(loss: float) -> float:
    """
    Convert causal language-model loss into perplexity.
    """
    return math.exp(min(loss, 20.0))
