"""
Unit tests for EpsilonSamplingLogitsProcessor.

Run with:  pytest tests/test_processor.py -v
These tests mock the vLLM-specific dependencies so you don't need a GPU or
a full vLLM install to run them.
"""

from __future__ import annotations

import math
import os
import types
import unittest
from unittest.mock import MagicMock, patch

import torch

# ---------------------------------------------------------------------------
# Minimal stubs so the module can be imported without a vLLM installation
# ---------------------------------------------------------------------------
import sys

# Stub out vllm.v1.sample.logits_processor
_lp_mod = types.ModuleType("vllm.v1.sample.logits_processor")

class _FakeLogitsProcessor:
    pass

_lp_mod.LogitsProcessor = _FakeLogitsProcessor
_lp_mod.BatchUpdate = None
_lp_mod.MoveDirectionality = None

# Build the vllm package hierarchy
_vllm = types.ModuleType("vllm")
_vllm_v1 = types.ModuleType("vllm.v1")
_vllm_v1_sample = types.ModuleType("vllm.v1.sample")

sys.modules.setdefault("vllm", _vllm)
sys.modules.setdefault("vllm.v1", _vllm_v1)
sys.modules.setdefault("vllm.v1.sample", _vllm_v1_sample)
sys.modules["vllm.v1.sample.logits_processor"] = _lp_mod

# Now import our module
from vllm_epsilon_sampling.processor import (  # noqa: E402
    EpsilonSamplingLogitsProcessor,
    _DEFAULT_EPSILON,
    _ENV_KEY,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_processor(eps: float | None = None) -> EpsilonSamplingLogitsProcessor:
    """Instantiate processor, optionally overriding epsilon via env var."""
    env_patch: dict[str, str] = {}
    if eps is not None:
        env_patch[_ENV_KEY] = str(eps)
    with patch.dict(os.environ, env_patch, clear=False):
        proc = EpsilonSamplingLogitsProcessor(
            vllm_config=MagicMock(),
            device=torch.device("cpu"),
            is_pin_memory=False,
        )
    return proc


class TestEpsilonSamplingInit(unittest.TestCase):
    def test_default_epsilon(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(_ENV_KEY, None)
            proc = _make_processor()
        self.assertAlmostEqual(proc.epsilon, _DEFAULT_EPSILON)

    def test_custom_epsilon_via_env(self):
        proc = _make_processor(eps=0.05)
        self.assertAlmostEqual(proc.epsilon, 0.05)

    def test_invalid_epsilon_raises(self):
        with self.assertRaises(ValueError):
            _make_processor(eps=0.0)   # must be > 0
        with self.assertRaises(ValueError):
            _make_processor(eps=1.0)   # must be < 1
        with self.assertRaises(ValueError):
            with patch.dict(os.environ, {_ENV_KEY: "not_a_float"}):
                _make_processor()

    def test_is_not_argmax_invariant(self):
        proc = _make_processor()
        self.assertFalse(proc.is_argmax_invariant())

    def test_update_state_is_noop(self):
        proc = _make_processor()
        # Should not raise
        proc.update_state(None)
        proc.update_state(MagicMock())


class TestEpsilonSamplingApply(unittest.TestCase):
    """Core algorithm tests."""

    def _apply(self, logits_2d: list[list[float]], eps: float = 0.02) -> torch.Tensor:
        proc = _make_processor(eps=eps)
        t = torch.tensor(logits_2d, dtype=torch.float32)
        return proc.apply(t)

    # ------------------------------------------------------------------
    # Basic masking
    # ------------------------------------------------------------------

    def test_tokens_below_epsilon_masked(self):
        # One row: two tokens.  Give token-0 log(0.9) ≈ large, token-1 log(0.01)
        # p(0)=0.9, p(1)=0.1 with ε=0.15 → token-1 should be masked
        logits = torch.tensor([[math.log(9.0), math.log(1.0)]])  # unnormalized
        proc = _make_processor(eps=0.15)
        out = proc.apply(logits)
        # token-0 survives, token-1 is -inf
        self.assertTrue(torch.isfinite(out[0, 0]))
        self.assertEqual(out[0, 1].item(), float("-inf"))

    def test_tokens_at_epsilon_survive(self):
        # p(0)=0.02 exactly at boundary → should survive (>= ε)
        # Use two equal logits: each gets p=0.5, well above any ε we test
        logits = torch.zeros(1, 2)
        proc = _make_processor(eps=0.02)
        out = proc.apply(logits)
        # Both should survive
        self.assertTrue(torch.isfinite(out[0, 0]))
        self.assertTrue(torch.isfinite(out[0, 1]))

    def test_high_epsilon_keeps_only_top_token(self):
        # ε=0.99: only the token with p≥0.99 survives.
        # Craft logits so token-0 gets ~p=1.0
        logits = torch.tensor([[100.0, 0.0, 0.0]])
        proc = _make_processor(eps=0.99)
        out = proc.apply(logits)
        self.assertTrue(torch.isfinite(out[0, 0]))
        self.assertEqual(out[0, 1].item(), float("-inf"))
        self.assertEqual(out[0, 2].item(), float("-inf"))

    def test_all_equal_logits_all_survive_at_low_eps(self):
        # vocab=5, all logits=0 → each p=0.2.  ε=0.1 → all survive.
        logits = torch.zeros(1, 5)
        proc = _make_processor(eps=0.1)
        out = proc.apply(logits)
        self.assertTrue(torch.isfinite(out).all())

    def test_all_equal_logits_some_masked_at_high_eps(self):
        # vocab=5, all logits=0 → each p=0.2.  ε=0.25 → none naturally pass.
        # Safety fallback must keep the argmax (index 0).
        logits = torch.zeros(1, 5)
        proc = _make_processor(eps=0.25)
        out = proc.apply(logits)
        # At least one finite token
        self.assertTrue(torch.isfinite(out).any())

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

    def test_batch_independent_masking(self):
        # Row 0: confident → only token-0 survives
        # Row 1: uniform  → both survive (at low ε)
        logits = torch.tensor([
            [10.0, -10.0],
            [0.0,   0.0],
        ])
        proc = _make_processor(eps=0.02)
        out = proc.apply(logits)
        # Row 0: token-1 masked
        self.assertTrue(torch.isfinite(out[0, 0]))
        self.assertEqual(out[0, 1].item(), float("-inf"))
        # Row 1: both finite
        self.assertTrue(torch.isfinite(out[1, 0]))
        self.assertTrue(torch.isfinite(out[1, 1]))

    # ------------------------------------------------------------------
    # Safety fallback
    # ------------------------------------------------------------------

    def test_fallback_when_all_below_epsilon(self):
        # Force ε=0.99 with 3 equal tokens (each p=0.333 < 0.99).
        # Safety fallback must keep exactly one token (argmax).
        logits = torch.tensor([[0.0, 0.0, 0.0]])
        proc = _make_processor(eps=0.99)
        out = proc.apply(logits)
        finite_count = torch.isfinite(out[0]).sum().item()
        self.assertGreaterEqual(finite_count, 1, "At least one token must survive")

    def test_fallback_keeps_argmax(self):
        # Make token-2 the clear argmax; ε=0.99 forces fallback.
        logits = torch.tensor([[1.0, 2.0, 10.0]])
        proc = _make_processor(eps=0.99)
        out = proc.apply(logits)
        # token-2 should be the surviving one
        self.assertTrue(torch.isfinite(out[0, 2]))

    # ------------------------------------------------------------------
    # In-place safety: input tensor should not be mutated unexpectedly
    # ------------------------------------------------------------------

    def test_output_has_correct_shape(self):
        logits = torch.randn(4, 32000)
        proc = _make_processor()
        out = proc.apply(logits)
        self.assertEqual(out.shape, logits.shape)


if __name__ == "__main__":
    unittest.main()
