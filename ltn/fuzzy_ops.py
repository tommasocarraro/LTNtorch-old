import torch
# TODO commentare tutti gli operatori
"""
Element-wise fuzzy logic operators for PyTorch.
Supports traditional NumPy/PyTorch broadcasting.

To use in LTN formulas (broadcasting w.r.t. ltn variables appearing in a formula), 
wrap the operator with `ltn.WrapperConnective` or `ltn.WrapperQuantifier`. 
"""

# these are the projection functions to make the Product Real Logic stable. These functions help to change the input
# of particular fuzzy operators in such a way they do not lead to gradient problems (vanishing, exploding).
eps = 1e-4  # epsilon is set to small value


def pi_0(x):
    return (1-eps)*x + eps


def pi_1(x):
    return (1-eps)*x


# here, it begins the implementation of fuzzy operators in PyTorch
class NotStandard:
    def __call__(self, x):
        return 1. - x


class NotGodel:
    def __call__(self, x):
        zeros = torch.zeros_like(x)
        return torch.equal(x, zeros)


class AndMin:
    def __call__(self, x, y):
        return torch.minimum(x, y)


class AndProd:
    def __init__(self, stable=True):
        self.stable = stable

    def __call__(self, x, y, stable=None):
        stable = self.stable if stable is None else stable
        if stable:
            x, y = pi_0(x), pi_1(y)
        return torch.mul(x, y)


class AndLuk:
    def __call__(self, x, y):
        zeros = torch.zeros_like(x)
        return torch.maximum(x + y - 1., zeros)


class OrMax:
    def __call__(self, x, y):
        return torch.maximum(x, y)


class OrProbSum:
    def __init__(self, stable=True):
        self.stable = stable

    def __call__(self, x, y, stable=None):
        stable = self.stable if stable is None else stable
        if stable:
            x, y = pi_0(x), pi_1(y)
        return x + y - torch.mul(x, y)


class OrLuk:
    def __call__(self, x, y):
        ones = torch.ones_like(x)
        return torch.minimum(x + y, ones)


class ImpliesKleeneDienes:
    def __call__(self, x, y):
        return torch.maximum(1. - x, y)


class ImpliesGodel:
    def __call__(self, x, y):
        return torch.where(torch.le(x, y), torch.ones_like(x), y)


class ImpliesReichenbach:
    def __init__(self, stable=True):
        self.stable = stable

    def __call__(self, x, y, stable=None):
        stable = self.stable if stable is None else stable
        if stable:
            x, y = pi_0(x), pi_1(y)
        return 1. - x + torch.mul(x, y)


class ImpliesGoguen:
    def __init__(self,stable=True):
        self.stable = stable

    def __call__(self, x, y, stable=None):
        stable = self.stable if stable is None else stable
        if stable:
            x = pi_0(x)
        return torch.where(torch.le(x, y), torch.ones_like(x), torch.div(y, x))


class ImpliesLuk:
    def __call__(self, x, y):
        ones = torch.ones_like(x)
        return torch.minimum(1. - x + y, ones)


class Equiv:
    """Returns an operator that computes: And(Implies(x,y),Implies(y,x))"""
    def __init__(self, and_op, implies_op):
        self.and_op = and_op
        self.implies_op = implies_op

    def __call__(self, x, y):
        return self.and_op(self.implies_op(x, y), self.implies_op(y, x))


class AggregMin:
    def __call__(self, xs, dim=None, keepdim=False):
        return torch.min(xs, dim=dim, keepdim=keepdim)


class AggregMax:
    def __call__(self, xs, dim=None, keepdim=False):
        return torch.max(xs, dim=dim, keepdim=keepdim)


class AggregMean:
    def __call__(self, xs, dim=None, keepdim=False):
        numerator = torch.nansum(xs, dim=dim, keepdim=keepdim)
        denominator = torch.sum(~torch.isnan(xs), dim=dim, keepdim=keepdim)
        return torch.div(numerator / denominator)


class AggregPMean:
    def __init__(self, p=2, stable=True):
        self.p = p
        self.stable = stable

    def __call__(self, xs, dim=None, keepdim=False, p=None, stable=None):
        p = self.p if p is None else p
        stable = self.stable if stable is None else stable
        if stable:
            xs = pi_0(xs)
        xs = torch.pow(xs, p)
        numerator = torch.nansum(xs, dim=dim, keepdim=keepdim)
        denominator = torch.sum(~torch.isnan(xs), dim=dim, keepdim=keepdim)
        return torch.pow(torch.div(numerator, denominator), 1 / p)


class AggregPMeanError:
    def __init__(self, p=2, stable=True):
        self.p = p
        self.stable = stable

    def __call__(self, xs, dim=None, keepdim=False, p=None, stable=None):
        p = self.p if p is None else p
        stable = self.stable if stable is None else stable
        if stable:
            xs = pi_0(xs)
        xs = torch.pow(1. - xs, p)
        numerator = torch.nansum(xs, dim=dim, keepdim=keepdim)
        denominator = torch.sum(~torch.isnan(xs), dim=dim, keepdim=keepdim)
        return 1. - torch.pow(torch.div(numerator, denominator), 1 / p)