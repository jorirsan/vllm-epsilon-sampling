# vllm-epsilon-sampling

Epsilon sampling logits processor for [vLLM v1](https://docs.vllm.ai/).

Implements the **epsilon-sampling** strategy from:

> Freitag et al. (2023). *Epsilon Sampling Rocks: Investigating Sampling Strategies for Minimum Bayes Risk Decoding for Machine Translation.* Findings of EMNLP 2023.  
> https://aclanthology.org/2023.findings-emnlp.617

and originally proposed by:

> Hewitt et al. (2022). *Truncation Sampling as Language Model Desmoothing.* Findings of EMNLP 2022.

---

## What is epsilon sampling?

At each decoding step, standard ancestral sampling draws from the entire vocabulary, including tokens with negligibly small probability. Nucleus (top-p) and top-k sampling prune the tail, but each has drawbacks:

- **Top-k** fixes the number of kept tokens regardless of model confidence.
- **Top-p / nucleus** can still include hundreds of very low-probability tokens when the distribution is flat.

**Epsilon sampling** prunes the vocabulary by an absolute probability threshold:

```
keep token y  iff  P_model(y | context) >= ε
```

This gives a variable-size candidate set that adapts to model confidence, with the guarantee that every sampled token has at least probability mass `ε`. Freitag et al. (2023) found that `ε = 0.02` works well across machine translation language pairs, and that MBR decoding on top of epsilon samples significantly outperforms all other tested sampling strategies.

---

## Installation

```bash
pip install -e /path/to/vllm-epsilon-sampling
# or after publishing to PyPI:
pip install vllm-epsilon-sampling
```

Because the package registers a `vllm.logits_processors` entry point, **vLLM loads it automatically on startup** — no extra flags needed.

---
## Usage

### Auto-loaded via entry point (recommended)

Just install the package. vLLM picks it up automatically:

```bash
pip install vllm-epsilon-sampling
```

### Explicit FQCN via CLI

```bash
vllm serve <model> \
    --logits_processors vllm_epsilon_sampling.processor:EpsilonSamplingLogitsProcessor
```

### Python offline inference

```python
from vllm import LLM, SamplingParams
from vllm_epsilon_sampling.processor import EpsilonSamplingLogitsProcessor

llm = LLM(
    model="meta-llama/Llama-3-8B",
    #logits_processors=[EpsilonSamplingLogitsProcessor],
    #If installed via pip, the epsilon sampler should already be automatically loaded.
)

#generates 16 samples with epsilon sampling
outputs = llm.generate(
    ["Translate to German: The weather is nice today."],
    SamplingParams(n=16, top_k=-1, temperature=1.0, max_tokens=128, extra_args={"epsilon":0.02}),
)
```

---

## Algorithm detail

```
logits  (num_requests × vocab_size)
   │
   ▼
probs = softmax(logits, dim=-1)
   │
   ▼
keep_mask = (probs >= ε)          ← True for tokens that survive
   │
   ├─ safety: if keep_mask[i].any() == False for some row i
   │           → force-keep argmax(logits[i]) to avoid −inf everywhere
   │
   ▼
logits = masked_fill(logits, ~keep_mask, −inf)
   │
   ▼
returned to vLLM sampler → softmax → multinomial sample
```

Temperature is applied inside vLLM's sampler *before* this processor sees the logits, so set `SamplingParams(temperature=τ)` as usual. The paper's best single-language-pair setting was `(ε=0.02, τ=1.0)`.

---

## License

Apache 2.0
