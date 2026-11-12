"""One-time fetch of the response-lens weights (lm_head + final norm).

The vLLM V1 engine keeps the model in the EngineCore process, so the driver
cannot read the weights in-process. Instead the worker returns them via
``collective_rpc``. Tensors do not survive the RPC serialization intact
(they are re-encoded to ``(dtype, shape, aux_index)`` triples whose aux
buffers are not delivered to the caller), so the worker serializes the two
weights with ``torch.save`` (``.pt`` format) into a byte buffer and ships it
as a base64 string — plain strings survive the RPC intact.

Usage (driver side, sync or async engine). The worker exposes a
``fetch_lens_weights`` method (see ``hs_capture_worker``) that delegates to
the function below; call it by name — vLLM's RPC uses restricted
serialization that rejects callables unless
``VLLM_ALLOW_INSECURE_SERIALIZATION=1`` is set::

    from mia.lens_weights import decode_lens_weights
    results = llm.collective_rpc("fetch_lens_weights")
    weights = next((decode_lens_weights(r) for r in results
                    if r is not None), None)
    # weights == {"unembed": Tensor, "norm": Tensor}  (CPU, original dtype)
"""
from __future__ import annotations

import base64
import io

# Candidate state-dict keys, in priority order (tied-embedding models expose
# the unembedding only as the embedding weight).
_UNEMBED_KEYS = ("lm_head.weight", "output.weight", "model.embed_tokens.weight",
                 "transformer.wte.weight", "embed_tokens.weight")
_NORM_KEYS = ("model.norm.weight", "model.final_layernorm.weight",
              "transformer.ln_f.weight", "norm.weight", "model.norm.bias")


def fetch_lens_weights(self) -> dict | None:
    """Worker-side implementation of the ``fetch_lens_weights`` RPC method.

    Returns the unembedding + final-norm as a base64-encoded ``torch.save``
    (``.pt``) byte buffer, or ``None`` if the model/weights are not on this
    rank. The tensors are moved to CPU in their original dtype (the projector
    upcasts to float32 on its device). Invoked via the worker's
    ``fetch_lens_weights`` method (string-form ``collective_rpc``).
    """
    import torch
    model = getattr(getattr(self, "model_runner", None), "model", None)
    if model is None:
        return None
    sd = model.state_dict()
    keys = list(sd.keys())
    unembed_k = next((c for c in _UNEMBED_KEYS if c in keys), None)
    norm_k = next((c for c in _NORM_KEYS if c in keys), None)
    if not unembed_k or not norm_k:
        return None
    u = sd[unembed_k].detach().to("cpu")
    n = sd[norm_k].detach().to("cpu")
    buf = io.BytesIO()
    torch.save({"unembed": u, "norm": n}, buf)
    return {"pt_b64": base64.b64encode(buf.getvalue()).decode("ascii")}


def decode_lens_weights(result: dict | None) -> dict | None:
    """Driver-side: decode a ``fetch_lens_weights`` payload into
    ``{"unembed": Tensor, "norm": Tensor}`` (CPU, original dtype)."""
    if result is None:
        return None
    import torch
    buf = io.BytesIO(base64.b64decode(result["pt_b64"]))
    return torch.load(buf, map_location="cpu", weights_only=True)
