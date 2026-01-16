# TYPE_CHECKING import to avoid circular dependency at runtime
from multiprocessing import Value
from typing import TYPE_CHECKING

import torch
from jaxtyping import Float
from torch import Tensor

from spd.configs import SamplingType
from spd.models.components import ComponentsMaskInfo, WeightDeltaAndMask, make_mask_infos
from spd.routing import Router

if TYPE_CHECKING:
    from spd.models.component_model import ComponentModel


def calc_stochastic_component_mask_info(
    causal_importances: dict[str, Float[Tensor, "... C"]],
    component_mask_sampling: SamplingType,
    weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] | None,
    router: Router,
    component_model: "ComponentModel | None" = None,
    use_gradient_informed: bool = True,
) -> dict[str, ComponentsMaskInfo]:
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
                    if component_model is not None
                    else None
                )

                if use_gradient_informed:
                    grad_ci_dict = (
                        getattr(component_model, "_importance_sampling_gradients", None)
                        if component_model is not None
                        else None
                    )
                    if grad_ci_dict is not None:
                        grad = grad_ci_dict[layer]
                        print(
                            f"[Sampling] Layer {layer}: Using gradient-informed sampling! "
                            f"grad shape={grad.shape}, min={grad.min():.6f}, max={grad.max():.6f}"
                        )

                        importance = torch.abs(grad)
                        importance_normalized = importance / (
                            importance.sum(dim=-1, keepdim=True) + 1e-10
                        )
                        stochastic_source = 1.0 - importance_normalized

                        print(
                            f"[Sampling] Layer {layer}: stochastic_source min={stochastic_source.min():.6f}, "
                            f"max={stochastic_source.max():.6f}, mean={stochastic_source.mean():.6f}"
                        )
                    else:
                        raise ValueError("grad_ci_dict is None")
                else:
                    print(f"[Sampling] Layer {layer}: Using Uniform sampling! ")
                    stochastic_source = torch.rand_like(ci)

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
