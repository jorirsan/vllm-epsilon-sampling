"""
Epsilon Sampling for vLLM (v1)
==============================
Implements epsilon-sampling as described in:

  Freitag et al. (2023). "Epsilon Sampling Rocks: Investigating Sampling
  Strategies for Minimum Bayes Risk Decoding for Machine Translation."
  Findings of EMNLP 2023. https://aclanthology.org/2023.findings-emnlp.617

  Hewitt et al. (2022). "Truncation Sampling as Language Model Desmoothing."
  Findings of EMNLP 2022.

Algorithm
---------
At each decoding step, after the model produces a raw logits vector:

  1. Compute the softmax probability distribution p over the vocabulary.
  2. Find the epsilon threshold (default ε = 0.02, as used in the paper).
  3. Mask to -inf every token whose probability is strictly less than ε,
     so it receives zero probability mass after the subsequent softmax
     inside the sampler.
  4. Guard: if ALL tokens would be masked (i.e. even the top-1 token is
     below ε — which should never happen in practice but is theoretically
     possible for very small ε), fall back to keeping only the argmax token
     so sampling never degenerates to an empty distribution.

Note on temperature
-------------------
Temperature scaling (τ) is applied *before* this processor is called, inside
vLLM's own sampler, so this processor operates on already-temperature-scaled
logits.  If you want to pair epsilon-sampling with a custom temperature, set
`temperature` in `SamplingParams` as usual.

Configuration
-------------
The epsilon threshold is read from the environment variable
``VLLM_EPSILON_SAMPLING_EPS`` (float, default ``0.02``).
This is the recommended way to configure processors that are auto-loaded via
entrypoints, since vLLM instantiates them without constructor arguments.

Usage – auto-loaded entrypoint (install the package, then serve normally)
--------------------------------------------------------------------------
    pip install -e .
    vllm serve <model>   # EpsilonSamplingLogitsProcessor loads automatically

Usage – explicit FQCN
----------------------
    vllm serve <model> \\
        --logits_processors vllm_epsilon_sampling.processor:EpsilonSamplingLogitsProcessor

Usage – Python API
------------------
    from vllm import LLM, SamplingParams
    from vllm_epsilon_sampling.processor import EpsilonSamplingLogitsProcessor

    llm = LLM(model="...", logits_processors=[EpsilonSamplingLogitsProcessor])
    outputs = llm.generate("Hello", SamplingParams(temperature=1.0))
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch

from vllm.v1.sample.logits_processor import BatchUpdate, LogitsProcessor

if TYPE_CHECKING:
    from vllm.config import VllmConfig

# ---------------------------------------------------------------------------
# Default hyper-parameters (match paper's recommended setting)
# ---------------------------------------------------------------------------
_DEFAULT_EPSILON: float = 0.02
_ENV_KEY: str = "VLLM_EPSILON_SAMPLING_EPS"

from vllm.v1.sample.logits_processor import (
    AdapterLogitsProcessor, # Wrapper base-class
    RequestLogitsProcessor, # Request-level logitsproc type annotation
)
from typing import Any, Optional
from vllm import SamplingParams


# Stand-in for your request-level logits processor:
class EpsilonPerReqLogitsProcessor:
    """The request-level logits processor masks out all logits except the
    token id identified by `target_token`"""

    def __init__(self, epsilon: float) -> None:
        """Specify `target_token`"""
        self.epsilon = epsilon

    def __call__(
        self,
        output_ids: list[int],
        logits: torch.Tensor,
    ) -> torch.Tensor:
        #val_to_keep = logits[self.target_token].item()
        #logits[:] = float("-inf")
        #logits[self.target_token] = val_to_keep
        #return logits
        probs = torch.softmax(logits, dim=-1)  # (num_requests, vocab_size)

        # --- 2. Build boolean mask of tokens to keep (p >= epsilon) --------
        keep_mask = probs >= self.epsilon  # True where token survives

        # --- 3. Safety fallback: ensure at least one token survives per row -
        # If every token in a row is below epsilon (pathological edge case),
        # keep only the argmax token so we never produce an all-−inf row.
        any_kept = keep_mask.any(dim=-1, keepdim=True)  # (num_requests, 1)
        if not any_kept.all():
            # For rows where nothing passes, force the argmax token through.
            argmax_indices = logits.argmax(dim=-1, keepdim=True)
            fallback_mask = torch.zeros_like(keep_mask, dtype=torch.bool)
            fallback_mask.scatter_(1, argmax_indices, True)
            # Apply fallback only to rows that had nothing kept
            no_token_kept = ~any_kept.expand_as(keep_mask)
            keep_mask = keep_mask | (fallback_mask & no_token_kept)

        # --- 4. Mask out tokens below epsilon --------------------------------
        logits = logits.masked_fill(~keep_mask, float("-inf"))

        return logits


# Example of wrapping the request-level logits processor:
class EpsilonSamplingLogitsProcessor(AdapterLogitsProcessor):
    """Example of wrapping a fake request-level logit processor to create a
    batch-level logits processor"""

    @classmethod
    def validate_params(cls, params: SamplingParams):
        epsilon: Any | None = params.extra_args and params.extra_args.get(
            "epsilon"
        )
        if epsilon is not None and not isinstance(epsilon, float):
            raise ValueError(
                f"{epsilon=} is not float"
            )

    def is_argmax_invariant(self) -> bool:
        return False

    def new_req_logits_processor(
        self,
        params: SamplingParams,
    ) -> Optional[RequestLogitsProcessor]:
        """This method returns a new request-level logits processor, customized
        to the `target_token` value associated with a particular request.

        Returns None if the logits processor should not be applied to the
        particular request. To use the logits processor the request must have
        a "target_token" custom argument with an integer value.

        Args:
        params: per-request sampling params

        Returns:
        `Callable` request logits processor, or None
        """
        epsilon: Any | None = params.extra_args and params.extra_args.get(
            "epsilon"
        )
        if epsilon is None:
            return None
        return EpsilonPerReqLogitsProcessor(epsilon)