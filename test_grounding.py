import ltn
import torch


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
    print(And(p(v1), p(v2)))
    print(Implies(p(v1), p(v2)))
    print(p(v1))
    print(Not(p(v1)))
    print(Or(p(v1), p(v2)))
    print(Forall(v1, p(v1)))
    print(Exists(v1, p(v1)))


if __name__ == "__main__":
    main()