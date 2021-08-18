import torch
import numpy as np
from ltn import Grounding
"""
This module of the LTN project contains the PyTorch implementation of some common fuzzy logic operators. Refer to the
LTN paper for a detailed description of these operators (see the Appendix).
The operators support the traditional NumPy/PyTorch broadcasting.

In order to use these fuzzy operators with LTN formulas (broadcasting w.r.t. LTN variables appearing in a formula), 
it is necessary to wrap the operators with `ltn.WrapperConnective` or `ltn.WrapperQuantifier`. 
"""

# these are the projection functions to make the Product Real Logic stable. These functions help to change the input
# of particular fuzzy operators in such a way they do not lead to gradient problems (vanishing, exploding).
eps = 1e-4  # epsilon is set to small value in such a way to not change the input too much


def pi_0(x):
    """
    Function that has to be used when we need to assure that the truth value in input to a fuzzy operator is never equal
    to zero, in such a way to avoid gradient problems. It maps the interval [0, 1] in the interval ]0, 1], where the 0
    is excluded.
    :param x: a truth value
    :return: the input truth value changed in such a way to prevent gradient problems (0 is changed with a small number
    near 0).
    """
    return (1-eps)*x + eps


def pi_1(x):
    """
    Function that has to be used when we need to assure that the truth value in input to a fuzzy operator is never equal
    to one, in such a way to avoid gradient problems. It maps the interval [0, 1] in the interval [0, 1[, where the 1
    is excluded.
    :param x: a truth value
    :return: the input truth value changed in such a way to prevent gradient problems (1 is changed with a small number
    near 1).
    """
    return (1-eps)*x


# here, it begins the implementation of fuzzy operators in PyTorch

# NEGATION

class NotStandard:
    """
    Implementation of the standard fuzzy negation, namely the standard strict and strong negation.
    """
    def __call__(self, x):
        """
        Method __call__ for the standard fuzzy negation operator.
        :param x: the input truth value;
        :return: the standard negation of the input.
        """
        if isinstance(x, Grounding):
            x = x.tensor
        return 1. - x


class NotGodel:
    """
    Implementation of the Goedel fuzzy negation.
    """
    def __call__(self, x):
        """
        Method __call__ for the Godel fuzzy negation operator.
        :param x: the input truth value;
        :return: the Godel negation of the input.
        """
        if isinstance(x, Grounding):
            x = x.tensor
        zeros = torch.zeros_like(x)
        return torch.equal(x, zeros)

# CONJUNCTION


class AndMin:
    """
    Implementation of the Goedel fuzzy conjunction (min operator).
    """
    def __call__(self, x, y):
        """
        Method __call__ for the Godel fuzzy conjunction operator.
        :param x: the truth value of the first input;
        :param y: the truth value of the second input;
        :return: the Godel conjunction of the two inputs.
        """
        if isinstance(x, Grounding):
            x = x.tensor
        if isinstance(y, Grounding):
            y = y.tensor
        return torch.minimum(x, y)


class AndProd:
    """
    Implementation of the Goguen fuzzy conjunction (product operator).
    """
    def __init__(self, stable=True):
        """
        This constructor has to be used to set whether it has to be used the stable version (it avoids gradient
        problems) of the Goguen fuzzy conjunction or not.
        :param stable: a boolean flag indicating whether it has to be used the stable version of the operator or not.
        """
        self.stable = stable

    def __call__(self, x, y, stable=None):
        """
        Method __call__ for the Goguen fuzzy conjunction operator.
        :param x: the truth value of the first input;
        :param y: the truth value of the second input;
        :param stable: a boolean flag indicating whether it has to be used the stable version of the operator or not.
        :return: the Goguen conjunction of the two inputs.
        """
        stable = self.stable if stable is None else stable
        if isinstance(x, Grounding):
            x = x.tensor
        if isinstance(y, Grounding):
            y = y.tensor
        if stable:
            x, y = pi_0(x), pi_1(y)
        return torch.mul(x, y)


class AndLuk:
    """
    Implementation of the Lukasiewicz fuzzy conjunction.
    """
    def __call__(self, x, y):
        """
        Method __call__ for the Lukasiewicz fuzzy conjunction operator.
        :param x: the truth value of the first input;
        :param y: the truth value of the second input.
        :return: the Lukasiewicz conjunction of the two inputs.
        """
        if isinstance(x, Grounding):
            x = x.tensor
        if isinstance(y, Grounding):
            y = y.tensor
        zeros = torch.zeros_like(x)
        return torch.maximum(x + y - 1., zeros)

# DISJUNCTION


class OrMax:
    """
    Implementation of the Goedel fuzzy disjunction (max operator).
    """
    def __call__(self, x, y):
        """
        Method __call__ for the Goedel fuzzy disjunction operator.
        :param x: the truth value of the first input;
        :param y: the truth value of the second input.
        :return: the Goedel disjunction of the two inputs.
        """
        if isinstance(x, Grounding):
            x = x.tensor
        if isinstance(y, Grounding):
            y = y.tensor
        return torch.maximum(x, y)


class OrProbSum:
    """
    Implementation of the Goguen fuzzy disjunction (probabilistic sum operator).
    """
    def __init__(self, stable=True):
        """
        This constructor has to be used to set whether it has to be used the stable version (it avoids gradient
        problems) of the Goguen fuzzy disjunction or not.
        :param stable: a boolean flag indicating whether it has to be used the stable version of the operator or not.
        """
        self.stable = stable

    def __call__(self, x, y, stable=None):
        """
        Method __call__ for the Goguen fuzzy disjunction operator.
        :param x: the truth value of the first input;
        :param y: the truth value of the second input;
        :param stable: a boolean flag indicating whether it has to be used the stable version of the operator or not.
        :return: the Goguen disjunction of the two inputs.
        """
        stable = self.stable if stable is None else stable
        if isinstance(x, Grounding):
            x = x.tensor
        if isinstance(y, Grounding):
            y = y.tensor
        if stable:
            x, y = pi_0(x), pi_1(y)
        return x + y - torch.mul(x, y)


class OrLuk:
    """
    Implementation of the Lukasiewicz fuzzy disjunction.
    """
    def __call__(self, x, y):
        """
        Method __call__ for the Lukasiewicz fuzzy disjunction operator.
        :param x: the truth value of the first input;
        :param y: the truth value of the second input.
        :return: the Lukasiewicz disjunction of the two inputs.
        """
        if isinstance(x, Grounding):
            x = x.tensor
        if isinstance(y, Grounding):
            y = y.tensor
        ones = torch.ones_like(x)
        return torch.minimum(x + y, ones)

# IMPLICATION (differences between strong and residuated implications can be found in the Appendix of the LTN paper)


class ImpliesGoedelStrong:
    """
    Implementation of the Goedel strong fuzzy implication.
    """
    def __call__(self, x, y):
        """
        Method __call__ for the Goedel strong implication operator.
        :param x: the truth value of the first input;
        :param y: the truth value of the second input.
        :return: the Goedel strong implication of the two inputs.
        """
        if isinstance(x, Grounding):
            x = x.tensor
        if isinstance(y, Grounding):
            y = y.tensor
        return torch.maximum(1. - x, y)


class ImpliesGoedelResiduated:
    """
    Implementation of the Goedel residuated fuzzy implication.
    """
    def __call__(self, x, y):
        """
        Method __call__ for the Goedel residuated implication operator.
        :param x: the truth value of the first input;
        :param y: the truth value of the second input.
        :return: the Goedel residuated implication of the two inputs.
        """
        if isinstance(x, Grounding):
            x = x.tensor
        if isinstance(y, Grounding):
            y = y.tensor
        return torch.where(torch.le(x, y), torch.ones_like(x), y)


class ImpliesGoguenStrong:
    """
    Implementation of the Goguen strong fuzzy implication.
    """
    def __init__(self, stable=True):
        """
        This constructor has to be used to set whether it has to be used the stable version (it avoids gradient
        problems) of the Goguen strong fuzzy implication or not.
        :param stable: a boolean flag indicating whether it has to be used the stable version of the operator or not.
        """
        self.stable = stable

    def __call__(self, x, y, stable=None):
        """
        Method __call__ for the Goguen strong implication operator.
        :param x: the truth value of the first input;
        :param y: the truth value of the second input.
        :return: the Goguen strong implication of the two inputs.
        """
        stable = self.stable if stable is None else stable
        if isinstance(x, Grounding):
            x = x.tensor
        if isinstance(y, Grounding):
            y = y.tensor
        if stable:
            x, y = pi_0(x), pi_1(y)
        return 1. - x + torch.mul(x, y)


class ImpliesGoguenResiduated:
    """
    Implementation of the Goguen residuated fuzzy implication.
    """
    def __init__(self, stable=True):
        """
        This constructor has to be used to set whether it has to be used the stable version (it avoids gradient
        problems) of the Goguen residuated fuzzy implication or not.
        :param stable: a boolean flag indicating whether it has to be used the stable version of the operator or not.
        """
        self.stable = stable

    def __call__(self, x, y, stable=None):
        """
        Method __call__ for the Goguen residuated implication operator.
        :param x: the truth value of the first input;
        :param y: the truth value of the second input.
        :return: the Goguen residuated implication of the two inputs.
        """
        stable = self.stable if stable is None else stable
        if isinstance(x, Grounding):
            x = x.tensor
        if isinstance(y, Grounding):
            y = y.tensor
        if stable:
            x = pi_0(x)
        return torch.where(torch.le(x, y), torch.ones_like(x), torch.div(y, x))


class ImpliesLuk:
    """
    Implementation of the Lukasiewicz fuzzy implication.
    """
    def __call__(self, x, y):
        """
        Method __call__ for the Lukasiewicz implication operator.
        :param x: the truth value of the first input;
        :param y: the truth value of the second input.
        :return: the Lukasiewicz implication of the two inputs.
        """
        if isinstance(x, Grounding):
            x = x.tensor
        if isinstance(y, Grounding):
            y = y.tensor
        ones = torch.ones_like(x)
        return torch.minimum(1. - x + y, ones)

# EQUIVALENCE


class Equiv:
    """
    Returns an operator that computes: And(Implies(x,y),Implies(y,x)). In other words, it computes: x -> y AND y -> x.
    """
    def __init__(self, and_op, implies_op):
        """
        This constructor has to be used to set the operator for the conjunction and for the implication of the
        equivalence operator.
        :param and_op: fuzzy operator for the conjunction;
        :param implies_op; fuzzy operator for the implication.
        """
        self.and_op = and_op
        self.implies_op = implies_op

    def __call__(self, x, y):
        """
        Method __call__ for the equivalence operator.
        :param x: the truth value of the first input;
        :param y: the truth value of the second input.
        :return: the fuzzy equivalence of the two inputs.
        """
        if isinstance(x, Grounding):
            x = x.tensor
        if isinstance(y, Grounding):
            y = y.tensor
        return self.and_op(self.implies_op(x, y), self.implies_op(y, x))

# AGGREGATORS FOR QUANTIFIERS - only the aggregators introduced in the LTN paper are implemented


class AggregMin:
    """
    Implementation of the min aggregator operator.
    """
    def __call__(self, xs, dim=None, keepdim=False):
        """
        Method __call__ for the min aggregator operator. Notice the use of torch.where(). This has to be used
        because the guarded quantification is implemented in PyTorch by putting NaN values where the grounding of the
        formula does not satisfy the guarded condition. Therefore, if we aggregate on a tensor with NaN values, it is
        highly probable that we will obtain NaN as the output of the aggregation. For this reason, the aggregation do
        not have to consider the NaN values contained in the input tensor.
        :param xs: the truth values (grounding of formula) for which the aggregation has to be computed;
        :param dim: the axes on which the aggregation has to be performed;
        :param keepdim: whether the output has to keep the same dimensions as the input after the aggregation.
        :return: the result of the mean aggregation. The shape of the result depends on the variables that are used
        in the quantification (namely, the dimensions across which the aggregation has been computed).
        """
        if isinstance(xs, Grounding):
            xs = xs.tensor
        xs = torch.where(torch.eq(xs, np.nan), 1., xs.double())
        out = torch.min(xs.float(), dim=dim, keepdim=keepdim)
        if isinstance(out, tuple):
            out = out[0]
        return out


class AggregMean:
    """
    Implementation of the mean aggregator operator.
    """
    def __call__(self, xs, dim=None, keepdim=False):
        """
        Method __call__ for the mean aggregator operator. Notice the use of torch.nansum(). This has to be used
        because the guarded quantification is implemented in PyTorch by putting NaN values where the grounding of the
        formula does not satisfy the guarded condition. Therefore, if we aggregate on a tensor with NaN values, it is
        highly probable that we will obtain NaN as the output of the aggregation. For this reason, the aggregation do
        not have to consider the NaN values contained in the input tensor.
        :param xs: the truth values (grounding of formula) for which the aggregation has to be computed;
        :param dim: the axes on which the aggregation has to be performed;
        :param keepdim: whether the output has to keep the same dimensions as the input after the aggregation.
        :return: the result of the mean aggregation. The shape of the result depends on the variables that are used
        in the quantification (namely, the dimensions across which the aggregation has been computed).
        """
        if isinstance(xs, Grounding):
            xs = xs.tensor
        numerator = torch.nansum(xs, dim=dim, keepdim=keepdim)
        denominator = torch.sum(~torch.isnan(xs), dim=dim, keepdim=keepdim)
        return torch.div(numerator, denominator)


class AggregPMean:
    """
    Implementation of the p-mean aggregator operator. This has been selected as an approximation of the existential
    quantifier with parameter p equal to or greater than 1. If p tends to infinity, the p-mean aggregator tends to the
    maximum of the input values (approximation of fuzzy existential quantification).
    """
    def __init__(self, p=2, stable=True):
        """
        This constructor has to be used to set whether it has to be used the stable version (it avoids gradient
        problems) of the p-mean aggregator or not. Also, it is possible to set the value of the parameter p.
        :param p: value of the parameter p;
        :param stable: a boolean flag indicating whether it has to be used the stable version of the aggregator or not.
        """
        self.p = p
        self.stable = stable

    def __call__(self, xs, dim=None, keepdim=False, p=None, stable=None):
        """
        Method __call__ for the p-mean aggregator operator. Notice the use of torch.nansum(). This has to be used
        because the guarded quantification is implemented in PyTorch by putting NaN values where the grounding of the
        formula does not satisfy the guarded condition. Therefore, if we aggregate on a tensor with NaN values, it is
        highly probable that we will obtain NaN as the output of the aggregation. For this reason, the aggregation do
        not have to consider the NaN values contained in the input tensor.
        :param xs: the truth values (grounding of formula) for which the aggregation has to be computed;
        :param dim: the axes on which the aggregation has to be performed;
        :param keepdim: whether the output has to keep the same dimensions as the input after the aggregation.
        :param: p: the value of the parameter p;
        :param stable: a boolean flag indicating whether it has to be used the stable version of the aggregator or not.
        :return: the result of the p-mean aggregation. The shape of the result depends on the variables that are used
        in the quantification (namely, the dimensions across which the aggregation has been computed).
        """
        p = self.p if p is None else p
        stable = self.stable if stable is None else stable
        if isinstance(xs, Grounding):
            xs = xs.tensor
        if stable:
            xs = pi_0(xs)
        xs = torch.pow(xs, p)
        numerator = torch.nansum(xs, dim=dim, keepdim=keepdim)
        denominator = torch.sum(~torch.isnan(xs), dim=dim, keepdim=keepdim)
        return torch.pow(torch.div(numerator, denominator), 1 / p)


class AggregPMeanError:
    """
    Implementation of the p-mean error aggregator operator. This has been selected as an approximation of the universal
    quantifier with parameter p equal to or greater than 1. If p tends to infinity, the p-mean error aggregator tends
    to the minimum of the input values (approximation of fuzzy universal quantification).
    """
    def __init__(self, p=2, stable=True):
        """
        This constructor has to be used to set whether it has to be used the stable version (it avoids gradient
        problems) of the p-mean error aggregator or not. Also, it is possible to set the value of the parameter p.
        :param p: value of the parameter p;
        :param stable: a boolean flag indicating whether it has to be used the stable version of the aggregator or not.
        """
        self.p = p
        self.stable = stable

    def __call__(self, xs, dim=None, keepdim=False, p=None, stable=None):
        """
        Method __call__ for the p-mean error aggregator operator. Notice the use of torch.nansum(). This has to be used
        because the guarded quantification is implemented in PyTorch by putting NaN values where the grounding of the
        formula does not satisfy the guarded condition. Therefore, if we aggregate on a tensor with NaN values, it is
        highly probable that we will obtain NaN as the output of the aggregation. For this reason, the aggregation do
        not have to consider the NaN values contained in the input tensor.
        :param xs: the truth values (grounding of formula) for which the aggregation has to be computed;
        :param dim: the axes on which the aggregation has to be performed;
        :param keepdim: whether the output has to keep the same dimensions as the input after the aggregation.
        :param: p: the value of the parameter p;
        :param stable: a boolean flag indicating whether it has to be used the stable version of the aggregator or not.
        :return: the result of the p-mean error aggregation. The shape of the result depends on the variables that are
        used in the quantification (namely, the dimensions across which the aggregation has been computed).
        """
        p = self.p if p is None else p
        stable = self.stable if stable is None else stable
        if isinstance(xs, Grounding):  # inside an object of type Grounding there is the tensor we need
            # for the computation
            xs = xs.tensor
        if stable:
            xs = pi_1(xs)
        xs = torch.pow(1. - xs, p)
        numerator = torch.nansum(xs, dim=dim, keepdim=keepdim)
        denominator = torch.sum(~torch.isnan(xs), dim=dim, keepdim=keepdim)
        return 1. - torch.pow(torch.div(numerator, denominator), 1 / p)