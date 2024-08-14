from typing import List, Union, Optional
import torch
from torch.autograd import grad
import torch._dynamo

def hessian(y, x):
    """hessian of y wrt x
    y: shape (meta_batch_size, num_observations, channels)
    x: shape (meta_batch_size, num_observations, 2)
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


def laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)


def divergence(y, x):
    div = 0.0
    for i in range(y.shape[-1]):
        div += grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][
            ..., i : i + 1
        ]
    return div


@torch.jit.script
def gradient(dy: torch.Tensor,
             dx: Union[List[torch.Tensor], torch.Tensor],
             ones_like_tensor: Optional[List[Optional[torch.Tensor]]] = None,
             create_graph: bool = True) -> Optional[torch.Tensor]:
    """Compute the gradient of a tensor `dy` with respect to another tensor `dx`.

    :param dy: The tensor to compute the gradient for.
    :param dx: The tensor with respect to which the gradient is computed.
    :param ones_like_tensor: A tensor with the same shape as `dy`, used for creating the gradient (default is None).
    :param create_graph: Whether to create a computational graph for higher-order gradients (default is True).
    :return: The gradient of `dy` with respect to `dx`.
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



def jacobian(y, x):
    """jacobian of y wrt x"""
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
