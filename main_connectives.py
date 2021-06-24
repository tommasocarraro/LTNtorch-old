import ltn
import numpy as np
import torch

def main():
    Not = ltn.WrapperConnective(ltn.fuzzy_ops.NotStandard())
    And = ltn.WrapperConnective(ltn.fuzzy_ops.AndProd())
    Or = ltn.WrapperConnective(ltn.fuzzy_ops.OrProbSum())
    Implies = ltn.WrapperConnective(ltn.fuzzy_ops.ImpliesReichenbach())
    dom = ltn.Domain([2], 'R^2')
    x = ltn.Variable('x', dom, np.random.normal(0., 1., (10, 2)))  # 10 values in R²
    y = ltn.Variable('y', dom, np.random.normal(0., 2., (5, 2)))  # 5 values in R²

    c1 = ltn.Constant('c1', dom, [0.5, 0.0])
    c2 = ltn.Constant('c2', dom, [4.0, 2.0])

    Eq = ltn.Predicate('sim', [dom, dom], lambda_func=lambda args: torch.exp(-torch.norm(args[0] - args[1], dim=1)))  # predicate measuring similarity

    print(Eq([c1.get_grounding(), c2.get_grounding()]))
    print(Not(Eq([c1.get_grounding(), c2.get_grounding()])))
    print(Implies(Eq([c1.get_grounding(), c2.get_grounding()]), Eq([c2.get_grounding(), c1.get_grounding()])))

    print(And(Eq([x.get_grounding(), c1.get_grounding()]), Eq([x.get_grounding(), c2.get_grounding()])).shape)

    print(Or(Eq([x.get_grounding(), c1.get_grounding()]), Eq([x.get_grounding(), y.get_grounding()])).shape)

    Forall = ltn.WrapperQuantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="forall")
    Exists = ltn.WrapperQuantifier(ltn.fuzzy_ops.AggregPMean(p=5), quantifier="exists")

    print(Forall(x.get_grounding(), Eq([x.get_grounding(), y.get_grounding()])).shape)

    #print(Eq([x.get_grounding(), y.get_grounding()]))

    print(Forall([x.get_grounding(), y.get_grounding()], Eq([x.get_grounding(), y.get_grounding()])))

    print(Exists([x.get_grounding(), y.get_grounding()], Eq([x.get_grounding(), y.get_grounding()])))

    print(Forall(x.get_grounding(), Exists(y.get_grounding(), Eq([x.get_grounding(), y.get_grounding()]))))

    x = ltn.Variable('x', dom, np.random.normal(0., 1., (5, 2)))

    print(Eq([x.get_grounding(), y.get_grounding()]))

    x, y = ltn.diag([x.get_grounding(), y.get_grounding()])

    print(Eq([x, y]))




if __name__ == "__main__":
    main()