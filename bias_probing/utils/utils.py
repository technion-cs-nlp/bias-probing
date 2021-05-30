import torch


class GradMulConst(torch.autograd.Function):
    """A layer the multiplies gradients by a constant value.
    This layer can be used to create an adversarial loss, by setting `const=-lambda`.
    """

    @staticmethod
    def forward(ctx, x, const):
        ctx.const = const
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.const, None


def grad_mul_const(x, const):
    """Apply a `GradMulConst` layer to a Tensor x with a constant value `const`
    """
    return GradMulConst.apply(x, const)
