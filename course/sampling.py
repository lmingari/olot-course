import torch


@torch.no_grad()
def sample(model, shape, steps=50, return_trajectory=False):
    """Generate samples by integrating dx/dt = model(x, t) from t=0 to t=1 (Euler).

    Args:
        model: trained FlowUNet (already on its device).
        shape: shape of the samples, (B, C, H, W).
        steps: number of Euler steps. More steps = more accurate, but slower.
        return_trajectory: if True, also return every intermediate state.

    Returns:
        If return_trajectory is False:
            Tensor (B, C, H, W) on the model's device.
        If True:
            Tensor (steps + 1, B, C, H, W) on the CPU; entry k is the state at
            t = k / steps (entry 0 is the initial noise, the last is the sample).
        Both are in the *transformed* space (apply ``transform.invert`` to get
        physical units).
    """
    model.eval()
    device = next(model.parameters()).device

    x = torch.randn(shape, device=device)             # x_0 ~ N(0, I)
    dt = 1.0 / steps
    trajectory = [x.cpu()] if return_trajectory else None

    for k in range(steps):
        t = torch.full((shape[0],), k * dt, device=device)   # (B,) same time for all samples
        x = x + dt * model(x, t)                              # Euler step
        if return_trajectory:
            trajectory.append(x.cpu())

    return torch.stack(trajectory) if return_trajectory else x
