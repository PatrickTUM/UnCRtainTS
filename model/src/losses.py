import math
import torch
Tensor = torch.Tensor
import torch.nn as nn
import torch.nn.modules.loss
from torch.nn.modules.loss import _Loss
from torch.overrides import has_torch_function_variadic, handle_torch_function

from torch import vmap

S2_BANDS = 13


def get_loss(config):
    if config.loss == "GNLL":
        criterion1 = GaussianNLLLoss(reduction='mean', eps=1e-8, full=True)
        criterion = lambda pred, targ, var: criterion1(pred, targ, var)
    elif config.loss == "MGNLL":
        criterion1 = MultiGaussianNLLLoss(reduction='mean', eps=1e-8, full=True, mode=config.covmode, chunk=config.chunk_size)
        criterion = lambda pred, targ, var: criterion1(pred, targ, var)
    elif config.loss=="l1":
        criterion1 = nn.L1Loss()
        criterion = lambda pred, targ: criterion1(pred, targ)
    elif config.loss=="l2":
        criterion1 = nn.MSELoss()
        criterion = lambda pred, targ: criterion1(pred, targ)
    else: raise NotImplementedError

    # wrap losses
    loss_wrap = lambda *args: args
    loss = loss_wrap(criterion) 
    return loss if not isinstance(loss, tuple) else loss[0]


def calc_loss(criterion, config, out, y, var=None):
    
    if config.loss in ['GNLL']:
        loss, variance = criterion(out, y, var)
    elif config.loss in ['MGNLL']:
        loss, variance = criterion(out, y, var)
    else: 
        loss, variance = criterion(out, y), None
    return loss, variance


def gaussian_nll_loss(
    input: Tensor,
    target: Tensor,
    var: Tensor,
    full: bool = False,
    eps: float = 1e-8,
    reduction: str = "mean",
) -> Tensor:
    r"""Gaussian negative log likelihood loss.

    based on :class:`~torch.nn.GaussianNLLLoss` for details.

    Args:
        input: expectation of the Gaussian distribution.
        target: sample from the Gaussian distribution.
        var: tensor of positive variance(s), one for each of the expectations
            in the input (heteroscedastic), or a single one (homoscedastic).
        full (bool, optional): include the constant term in the loss calculation. Default: ``False``.
        eps (float, optional): value added to var, for stability. Default: 1e-6.
        reduction (string, optional): specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the output is the average of all batch member losses,
            ``'sum'``: the output is the sum of all batch member losses.
            Default: ``'mean'``.
    """
    if has_torch_function_variadic(input, target, var):
        return handle_torch_function(
            gaussian_nll_loss,
            (input, target, var),
            input,
            target,
            var,
            full=full,
            eps=eps,
            reduction=reduction,
        )

    # Check var size
    # If var.size == input.size, the case is heteroscedastic and no further checks are needed.
    # Otherwise:
    if var.size() != input.size():

        # If var is one dimension short of input, but the sizes match otherwise, then this is a homoscedastic case.
        # e.g. input.size = (10, 2, 3), var.size = (10, 2)
        # -> unsqueeze var so that var.shape = (10, 2, 1)
        # this is done so that broadcasting can happen in the loss calculation
        if input.size()[:-1] == var.size():
            var = torch.unsqueeze(var, dim=-1)

        # This checks if the sizes match up to the final dimension, and the final dimension of var is of size 1.
        # This is also a homoscedastic case.
        # e.g. input.size = (10, 2, 3), var.size = (10, 2, 1)
        elif input.size()[:-1] == var.size()[:-1] and var.size(-1) == 1:  # Heteroscedastic case
            pass

        # If none of the above pass, then the size of var is incorrect.
        else:
            raise ValueError("var is of incorrect size")

    # Check validity of reduction mode
    if reduction != 'none' and reduction != 'mean' and reduction != 'sum':
        raise ValueError(reduction + " is not valid")

    # Entries of var must be non-negative
    if torch.any(var < 0):
        raise ValueError("var has negative entry/entries")

    # Clamp for stability
    var = var.clone()
    with torch.no_grad():
        var.clamp_(min=eps)

    # Calculate the loss
    loss = 0.5 * (torch.log(var) + (input - target)**2 / var)
    if full:
        loss += 0.5 * math.log(2 * math.pi)

    if reduction == 'mean':
        return loss.mean(), var
    elif reduction == 'sum':
        return loss.sum(), var
    else:
        return loss, var


def multi_diag_gaussian_nll(pred, target, var):
    # maps var from [B x 1 x C] to [B x 1 x C x C]
    pred, target, var = pred.squeeze(dim=1), target.squeeze(dim=1), var.squeeze(dim=1)

    k  = pred.shape[-1]
    prec = torch.diag_embed(1/var, offset=0, dim1=-2, dim2=-1)  
    # the log-determinant of a diagonal matrix is simply the trace of the log of the diagonal matrix
    logdetv = var.log().sum() # this may be more numerically stable a general calculation
    err  = (pred - target).unsqueeze(dim=1)
    # for the Mahalanobis distance xTCx to be defined and >= 0, the precision matrix must be positive definite
    xTCx = torch.bmm(torch.bmm(err, prec), err.permute(0, 2, 1)).squeeze().nan_to_num().clamp(min=1e-9) # note: equals torch.bmm(torch.bmm(-err, prec), -err)
    # define the NLL loss
    loss = -(-k/2 * torch.log(2*torch.tensor(torch.pi)) - 1/2 * logdetv - 1/2 * xTCx)

    return loss, torch.diag_embed(var, offset=0, dim1=-2, dim2=-1).cpu()



def multi_gaussian_nll_loss(
    input: Tensor,
    target: Tensor,
    var: Tensor,
    full: bool = False,
    eps: float = 1e-8,
    reduction: str = "mean",
    mode: str = "diag",
    chunk = None
) -> Tensor:
    r"""Multivariate Gaussian negative log likelihood loss.

    based on :class:`~torch.nn.GaussianNLLLoss` for details.

    Args:
        input: expectation of the Gaussian distribution.
        target: sample from the Gaussian distribution.
        var: tensor of positive variance(s), one for each of the expectations
            in the input (heteroscedastic), or a single one (homoscedastic).
        full (bool, optional): include the constant term in the loss calculation. Default: ``False``.
        eps (float, optional): value added to var, for stability. Default: 1e-6.
        reduction (string, optional): specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the output is the average of all batch member losses,
            ``'sum'``: the output is the sum of all batch member losses.
            Default: ``'mean'``.
    """
    if has_torch_function_variadic(input, target, var):
        return handle_torch_function(
            multi_gaussian_nll_loss,
            (input, target, var),
            input,
            target,
            var,
            full=full,
            eps=eps,
            reduction=reduction,
            mode=mode,
            chunk=None
        )

    if mode=='iso':
        # duplicate the scalar variance across all spectral dimensions
        var = var.expand(-1,-1,S2_BANDS,-1,-1)

    # Check validity of reduction mode
    if reduction != 'none' and reduction != 'mean' and reduction != 'sum':
        raise ValueError(reduction + " is not valid")

    # Entries of var must be non-negative
    if torch.any(var < 0):
        raise ValueError("var has negative entry/entries")

    # Clamp for stability
    var = var.clone()
    with torch.no_grad():
        var[:,:,:S2_BANDS].clamp_(min=eps)

    if mode in ['iso', 'diag']:
        mapdims = (-1,-1,-1)
        loss, variance = vmap(vmap(multi_diag_gaussian_nll, in_dims=mapdims, chunk_size=chunk), in_dims=mapdims, chunk_size=chunk)(input, target, var)
    
    variance = variance.moveaxis(1,-1).moveaxis(0,-1).unsqueeze(1)

    if reduction == 'mean':
        return loss.mean(), variance
    elif reduction == 'sum':
        return loss.sum(), variance
    else:
        return loss, variance



class GaussianNLLLoss(_Loss):
    r"""Gaussian negative log likelihood loss.

    The targets are treated as samples from Gaussian distributions with
    expectations and variances predicted by the neural network. For a
    ``target`` tensor modelled as having Gaussian distribution with a tensor
    of expectations ``input`` and a tensor of positive variances ``var`` the loss is:

    .. math::
        \text{loss} = \frac{1}{2}\left(\log\left(\text{max}\left(\text{var},
        \ \text{eps}\right)\right) + \frac{\left(\text{input} - \text{target}\right)^2}
        {\text{max}\left(\text{var}, \ \text{eps}\right)}\right) + \text{const.}

    where :attr:`eps` is used for stability. By default, the constant term of
    the loss function is omitted unless :attr:`full` is ``True``. If ``var`` is not the same
    size as ``input`` (due to a homoscedastic assumption), it must either have a final dimension
    of 1 or have one fewer dimension (with all other sizes being the same) for correct broadcasting.

    Args:
        full (bool, optional): include the constant term in the loss
            calculation. Default: ``False``.
        eps (float, optional): value used to clamp ``var`` (see note below), for
            stability. Default: 1e-6.
        reduction (string, optional): specifies the reduction to apply to the
            output:``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
            will be applied, ``'mean'``: the output is the average of all batch
            member losses, ``'sum'``: the output is the sum of all batch member
            losses. Default: ``'mean'``.

    Shape:
        - Input: :math:`(N, *)` or :math:`(*)` where :math:`*` means any number of additional
          dimensions
        - Target: :math:`(N, *)` or :math:`(*)`, same shape as the input, or same shape as the input
          but with one dimension equal to 1 (to allow for broadcasting)
        - Var: :math:`(N, *)` or :math:`(*)`, same shape as the input, or same shape as the input but
          with one dimension equal to 1, or same shape as the input but with one fewer
          dimension (to allow for broadcasting)
        - Output: scalar if :attr:`reduction` is ``'mean'`` (default) or
          ``'sum'``. If :attr:`reduction` is ``'none'``, then :math:`(N, *)`, same
          shape as the input

    Note:
        The clamping of ``var`` is ignored with respect to autograd, and so the
        gradients are unaffected by it.

    Reference:
        Nix, D. A. and Weigend, A. S., "Estimating the mean and variance of the
        target probability distribution", Proceedings of 1994 IEEE International
        Conference on Neural Networks (ICNN'94), Orlando, FL, USA, 1994, pp. 55-60
        vol.1, doi: 10.1109/ICNN.1994.374138.
    """
    __constants__ = ['full', 'eps', 'reduction']
    full: bool
    eps: float

    def __init__(self, *, full: bool = False, eps: float = 1e-8, reduction: str = 'mean') -> None:
        super(GaussianNLLLoss, self).__init__(None, None, reduction)
        self.full = full
        self.eps  = eps

    def forward(self, input: Tensor, target: Tensor, var: Tensor) -> Tensor:
        return gaussian_nll_loss(input, target, var, full=self.full, eps=self.eps, reduction=self.reduction)




class MultiGaussianNLLLoss(_Loss):
    r"""Multivariate Gaussian negative log likelihood loss.

    The targets are treated as samples from Gaussian distributions with
    expectations and variances predicted by the neural network. For a
    ``target`` tensor modelled as having Gaussian distribution with a tensor
    of expectations ``input`` and a tensor of positive variances ``var`` the loss is:

    .. math::
        \text{loss} = \frac{1}{2}\left(\log\left(\text{max}\left(\text{var},
        \ \text{eps}\right)\right) + \frac{\left(\text{input} - \text{target}\right)^2}
        {\text{max}\left(\text{var}, \ \text{eps}\right)}\right) + \text{const.}

    where :attr:`eps` is used for stability. By default, the constant term of
    the loss function is omitted unless :attr:`full` is ``True``. If ``var`` is not the same
    size as ``input`` (due to a homoscedastic assumption), it must either have a final dimension
    of 1 or have one fewer dimension (with all other sizes being the same) for correct broadcasting.

    Args:
        full (bool, optional): include the constant term in the loss
            calculation. Default: ``False``.
        eps (float, optional): value used to clamp ``var`` (see note below), for
            stability. Default: 1e-6.
        reduction (string, optional): specifies the reduction to apply to the
            output:``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
            will be applied, ``'mean'``: the output is the average of all batch
            member losses, ``'sum'``: the output is the sum of all batch member
            losses. Default: ``'mean'``.

    Shape:
        - Input: :math:`(N, *)` or :math:`(*)` where :math:`*` means any number of additional
          dimensions
        - Target: :math:`(N, *)` or :math:`(*)`, same shape as the input, or same shape as the input
          but with one dimension equal to 1 (to allow for broadcasting)
        - Var: :math:`(N, *)` or :math:`(*)`, same shape as the input, or same shape as the input but
          with one dimension equal to 1, or same shape as the input but with one fewer
          dimension (to allow for broadcasting)
        - Latent: :math:`(N, *)` or :math:`(*)`, same shape as the input, or same shape as the input but
          with one dimension equal to 1, or same shape as the input but with one fewer
          dimension (to allow for broadcasting)
        - Output: scalar if :attr:`reduction` is ``'mean'`` (default) or
          ``'sum'``. If :attr:`reduction` is ``'none'``, then :math:`(N, *)`, same
          shape as the input

    Note:
        The clamping of ``var`` is ignored with respect to autograd, and so the
        gradients are unaffected by it.

    Reference:
        Nix, D. A. and Weigend, A. S., "Estimating the mean and variance of the
        target probability distribution", Proceedings of 1994 IEEE International
        Conference on Neural Networks (ICNN'94), Orlando, FL, USA, 1994, pp. 55-60
        vol.1, doi: 10.1109/ICNN.1994.374138.
    """
    __constants__ = ['full', 'eps', 'reduction']
    full: bool
    eps: float

    def __init__(self, *, full: bool = False, eps: float = 1e-8, reduction: str = 'mean', mode: str = 'diag', chunk: None) -> None:
        super(MultiGaussianNLLLoss, self).__init__(None, None, reduction)
        self.full = full
        self.eps  = eps
        self.mode = mode
        self.chunk = chunk
    
    def forward(self, input: Tensor, target: Tensor, var: Tensor) -> Tensor:
        return multi_gaussian_nll_loss(input, target, var, full=self.full, eps=self.eps, reduction=self.reduction, mode=self.mode, chunk=self.chunk)
