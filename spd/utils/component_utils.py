# TYPE_CHECKING import to avoid circular dependency at runtime
from typing import TYPE_CHECKING

import torch
from jaxtyping import Float
from torch import Tensor

from spd.configs import GIVariant, SamplingType
from spd.models.components import ComponentsMaskInfo, WeightDeltaAndMask, make_mask_infos
from spd.routing import Router

if TYPE_CHECKING:
    from spd.models.component_model import ComponentModel


def _power_iteration(G: Float[Tensor, "B C"], n_iter: int) -> Float[Tensor, " C"]:
    C = G.shape[-1]
    scale = G.norm() + 1e-30
    Gn = G / scale  # now O(1) entries
    v = torch.randn(C, device=G.device, dtype=G.dtype)
    v /= v.norm() + 1e-10
    for _ in range(n_iter):
        v = Gn.T @ (Gn @ v)
        v /= v.norm() + 1e-10
    return v


def gradient_informed_source(
    grad: Float[Tensor, "... C"],
    ci: Float[Tensor, "... C"],
    variant: GIVariant,
    coeff: float,
    power_iters: int,
    importance_temperature: float = 1.0,
) -> tuple[Float[Tensor, "... C"], float | None]:
    """Stochastic mask source biased toward high-reconstruction-error directions.

    `grad` is ∂L_recon/∂g for a full stochastic mask, so each per-datapoint row
    encodes that datapoint's joint sensitivity direction. The directional variants
    push a fresh uniform source down toward ablation (g_c) along a sensitivity
    direction while leaving the orthogonal randomness intact, so the source stays a
    *distribution* of masks rather than a deterministic worst case.

    Returns the source [..., C] and, for "power_iter", the fraction of batch
    gradient energy captured by the top direction (else None).
    """
    base_random = torch.rand_like(ci)
    match variant:
        case "per_component":
            # Scaling before exponentiation cancels during normalization while
            # preventing fp32 underflow for small gradients at high temperatures.
            grad_mag = grad.abs()
            grad_mag = grad_mag / (grad_mag.amax(dim=-1, keepdim=True) + 1e-10)
            importance = grad_mag.pow(importance_temperature)
            importance_normalized = importance / (importance.sum(dim=-1, keepdim=True) + 1e-10)
            return (1.0 - importance_normalized) * base_random, None
        case "per_example":
            direction = grad / (grad.norm(dim=-1, keepdim=True) + 1e-10)
            source = (base_random - coeff * direction.clamp(min=0) * base_random).clamp(0.0, 1.0)
            return source, None
        case "mean":
            grad_matrix = grad.reshape(-1, grad.shape[-1])
            direction = grad_matrix.mean(dim=0)
            direction = direction / (direction.norm() + 1e-10)
            source = (base_random - coeff * direction.clamp(min=0) * base_random).clamp(0.0, 1.0)
            return source, None
        case "power_iter":
            grad_matrix = grad.reshape(-1, grad.shape[-1])
            direction = _power_iteration(grad_matrix, power_iters)
            captured = (grad_matrix @ direction).pow(2).sum() / (grad_matrix.pow(2).sum() + 1e-10)
            # The singular vector's sign is arbitrary; orient it so the batch projects
            # positively onto it (the gradient points toward increasing loss), so the
            # ablation push below consistently targets the sensitive direction.
            if (grad_matrix @ direction).sum() < 0:
                direction = -direction
            source = (base_random - coeff * direction.clamp(min=0) * base_random).clamp(0.0, 1.0)
            print(
                f"captured={captured.item():.6e}  Gnorm={grad_matrix.norm().item():.3e}  projnorm={(grad_matrix @ direction).norm().item():.3e}"
            )
            return source, captured.item()


def calc_stochastic_component_mask_info(
    causal_importances: dict[str, Float[Tensor, "... C"]],
    component_mask_sampling: SamplingType,
    weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] | None,
    router: Router,
    component_model: "ComponentModel | None" = None,
    use_gradient_informed: bool = True,
    importance_temperature: float = 1.0,
) -> dict[str, ComponentsMaskInfo]:
    """Draw stochastic component masks.

    For per-component ``gradient_informed`` sampling the stochastic source is
    scaled down for high-attribution components:

        w_c = |grad_c|^k / sum_c |grad_c|^k
        stochastic_source = (1 - w_c) * Uniform[0, 1]

    ``importance_temperature`` is the exponent ``k``. k=1 recovers plain
    magnitude normalisation (mass spread ~1/C over all components, so the
    sampler stays close to uniform). Larger k concentrates the weight onto the
    top-attribution components, moving those away from uniform while the bulk
    becomes more uniform.
    """
    ci_sample = next(iter(causal_importances.values()))
    leading_dims = ci_sample.shape[:-1]
    device = ci_sample.device
    dtype = ci_sample.dtype

    component_masks: dict[str, Float[Tensor, "... C"]] = {}
    for layer, ci in causal_importances.items():
        match component_mask_sampling:
            case "binomial":
                stochastic_source = torch.randint(0, 2, ci.shape, device=device).float()
            case "continuous":
                stochastic_source = torch.rand_like(ci)
            case "gradient_informed":
                grad_ci_dict = (
                    getattr(component_model, "_importance_sampling_gradients", None)
                    if (component_model is not None and use_gradient_informed)
                    else None
                )
                if grad_ci_dict is None or layer not in grad_ci_dict:
                    # No gradients stored yet (e.g. step 0): fall back to random sampling
                    stochastic_source = torch.rand_like(ci)
                else:
                    assert component_model is not None
                    stochastic_source, _ = gradient_informed_source(
                        grad=grad_ci_dict[layer],
                        ci=ci,
                        variant=component_model.gi_variant,
                        coeff=component_model.gi_coeff,
                        power_iters=component_model.gi_power_iters,
                        importance_temperature=importance_temperature,
                    )

        component_masks[layer] = ci + (1 - ci) * stochastic_source

    weight_deltas_and_masks: dict[str, WeightDeltaAndMask] | None = None
    if weight_deltas is not None:
        weight_deltas_and_masks = {}
        for layer in causal_importances:
            weight_deltas_and_masks[layer] = (
                weight_deltas[layer],
                torch.rand(leading_dims, device=device, dtype=dtype),
            )

    routing_masks = router.get_masks(
        module_names=list(causal_importances.keys()),
        mask_shape=leading_dims,
    )

    return make_mask_infos(
        component_masks=component_masks,
        weight_deltas_and_masks=weight_deltas_and_masks,
        routing_masks=routing_masks,
    )


def calc_ci_l_zero(ci: Float[Tensor, "... C"], threshold: float) -> float:
    return (ci > threshold).float().sum(-1).mean().item()
