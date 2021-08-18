import ltn
import torch
import numpy as np


def main():
    c = ltn.constant([1., 2., 3.], trainable=True)
    v1 = ltn.variable('v1', [6., 5., 3.])
    v2 = ltn.variable('v2', [2., 7., 1.])
    And = ltn.WrapperConnective(ltn.fuzzy_ops.AndProd())
    Implies = ltn.WrapperConnective(ltn.fuzzy_ops.ImpliesGoguenStrong())
    Or = ltn.WrapperConnective(ltn.fuzzy_ops.OrProbSum())
    Not = ltn.WrapperConnective(ltn.fuzzy_ops.NotStandard())
    Forall = ltn.WrapperQuantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantification_type="forall")
    Exists = ltn.WrapperQuantifier(ltn.fuzzy_ops.AggregPMean(p=2), quantification_type="exists")
    p = ltn.Predicate(lambda_func=lambda arg: arg[0] > 5)
    _and = And(p(v1), p(v2))
    _implies = Implies(p(v1), p(v2))
    _not = Not(p(v1))
    _or = Or(p(v1), p(v2))
    _forall = Forall(v1, p(v1))
    _exists = Exists(v1, p(v1))
    # torch.stack wants tensors, so we need to convert the groundings to tensors
    axioms = torch.stack(ltn.Grounding.convert_groundings_to_tensors([_forall, _exists]))

    print(ltn.fuzzy_ops.AggregPMeanError(p=2)(axioms, dim=0))


if __name__ == "__main__":
    main()