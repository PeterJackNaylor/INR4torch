from __future__ import annotations

from typing import List, Union, Optional, Tuple
import torch
from torch.autograd import grad
import torch._dynamo


def hessian(y: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
    """Compute the Hessian matrix of y with respect to x.

    Parameters
    ----------
    y : torch.Tensor
        Output tensor of shape (meta_batch_size, num_observations, channels).
    x : torch.Tensor
        Input tensor of shape (meta_batch_size, num_observations, dims).

    Returns
    -------
    h : torch.Tensor
        Hessian tensor of shape (meta_batch_size, num_observations, channels, dims, dims).
    status : int
        0 if OK, -1 if NaN values detected.
    """
    meta_batch_size, num_observations = y.shape[:2]
    grad_y = torch.ones_like(y[..., 0]).to(y.device)
    h = torch.zeros(
        meta_batch_size, num_observations, y.shape[-1], x.shape[-1], x.shape[-1]
    ).to(y.device)
    for i in range(y.shape[-1]):
        # calculate dydx over batches for each feature value of y
        dydx = grad(y[..., i], x, grad_y, create_graph=True)[0]

        # calculate hessian on y for each x value
        for j in range(x.shape[-1]):
            h[..., i, j, :] = grad(dydx[..., j], x, grad_y, create_graph=True)[0][
                ..., :
            ]

    status = 0
    if torch.any(torch.isnan(h)):
        status = -1
    return h, status


def laplace(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Compute the Laplacian of y with respect to x.

    Equivalent to the trace of the Hessian, computed as
    divergence(gradient(y, x), x).

    Parameters
    ----------
    y : torch.Tensor
        Output tensor.
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        Laplacian tensor.
    """
    grad = gradient(y, x)
    return divergence(grad, x)


def divergence(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Compute the divergence of a vector field y with respect to x.

    Parameters
    ----------
    y : torch.Tensor
        Vector field tensor with last dimension being the vector components.
    x : torch.Tensor
        Input coordinates.

    Returns
    -------
    torch.Tensor
        Divergence scalar field.
    """
    div = 0.0
    for i in range(y.shape[-1]):
        div += grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][
            ..., i : i + 1
        ]
    return div


@torch.jit.script
def gradient(
    dy: torch.Tensor,
    dx: Union[List[torch.Tensor], torch.Tensor],
    ones_like_tensor: Optional[List[Optional[torch.Tensor]]] = None,
    create_graph: bool = True,
) -> Optional[torch.Tensor]:
    """Compute the gradient of dy with respect to dx using autograd.

    Parameters
    ----------
    dy : torch.Tensor
        The output tensor to differentiate.
    dx : torch.Tensor or list of torch.Tensor
        The input tensor(s) with respect to which the gradient is computed.
    ones_like_tensor : list of torch.Tensor, optional
        Pre-allocated grad_outputs tensor. If None, uses torch.ones_like(dy).
    create_graph : bool, optional
        Whether to create computation graph for higher-order gradients. Default: True.

    Returns
    -------
    torch.Tensor
        Gradient tensor with the same shape as dx.
    """
    if ones_like_tensor is None:
        grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(dy)]
    else:
        grad_outputs = ones_like_tensor

    if isinstance(dx, torch.Tensor):
        dx = [dx]

    dy_dx = torch.autograd.grad(
        [dy],
        dx,
        grad_outputs=grad_outputs,
        create_graph=create_graph,
        retain_graph=True,
        allow_unused=False,
    )[0]

    return dy_dx


def jacobian(y: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
    """Compute the Jacobian matrix of y with respect to x.

    Parameters
    ----------
    y : torch.Tensor
        Output tensor of shape (meta_batch_size, num_observations, out_dims).
    x : torch.Tensor
        Input tensor of shape (meta_batch_size, num_observations, in_dims).

    Returns
    -------
    jac : torch.Tensor
        Jacobian of shape (meta_batch_size, num_observations, out_dims, in_dims).
    status : int
        0 if OK, -1 if NaN values detected.
    """
    meta_batch_size, num_observations = y.shape[:2]
    jac = torch.zeros(meta_batch_size, num_observations, y.shape[-1], x.shape[-1]).to(
        y.device
    )  # (meta_batch_size*num_points, 2, 2)
    for i in range(y.shape[-1]):
        # calculate dydx over batches for each feature value of y
        y_flat = y[..., i].view(-1, 1)
        jac[:, :, i, :] = grad(y_flat, x, torch.ones_like(y_flat), create_graph=True)[0]

    status = 0
    if torch.any(torch.isnan(jac)):
        status = -1

    return jac, status
