"""
This example shows how propositional variables have to be used in LTNtorch. In particular, it shows how to manage
trainable propositional variables by using torch.clamp().
"""
import torch
import numpy as np
import ltn
import logging
logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)


def main():
    # we define three trainable propositional variables
    a = ltn.propositional_variable(0.2, trainable=True)
    b = ltn.propositional_variable(0.5, trainable=True)
    c = ltn.propositional_variable(0.5, trainable=True)
    # we define two not-trainable propositional variables
    w1 = ltn.propositional_variable(0.3)
    w2 = ltn.propositional_variable(0.9)

    # we define a variable and a predicate that we need to define our knowledge base for this example
    x = ltn.variable("x", np.array([[1, 2], [3, 4], [5, 6]]))
    P = ltn.Predicate(layers_size=(2, 16, 16, 1))

    # connective and quantifiers
    Not = ltn.WrapperConnective(ltn.fuzzy_ops.NotStandard())
    And = ltn.WrapperConnective(ltn.fuzzy_ops.AndProd())
    Implies = ltn.WrapperConnective(ltn.fuzzy_ops.ImpliesGoguenStrong())
    Forall = ltn.WrapperQuantifier(ltn.fuzzy_ops.AggregPMeanError(p=5), quantification_type="forall")
    Exists = ltn.WrapperQuantifier(ltn.fuzzy_ops.AggregPMean(p=10), quantification_type="exists")

    # here, we define our knowledge base
    def axioms(a, b, c):
        # after the optimizer step, the value of the propositional variables could be changed in a value out of the
        # range [0, 1] since PyTorch does not allow to automatically constraint the value of a tensor in a range.
        # for this reason, we need to clamp the value of the three propositional variables in [0, 1]. To do so, we use
        # torch.clamp().
        a = torch.clamp(a, 0., 1.)
        a.free_variables = []  # since torch.clamp() returns a new tensor, the dynamic attribute free_variables will be
        # removed by this operation. For this reason, we need to reassign it to the LTN propositional variables.
        b = torch.clamp(b, 0., 1.)
        b.free_variables = []
        c = torch.clamp(c, 0., 1.)
        c.free_variables = []
        axioms = [
            # [ (A and B and (forall x: P(x))) -> Not C ] and C
            And(
                Implies(
                        And(And(a, b), Forall(x, P(x))),
                        Not(c)
                ),
                c
            ),
            # w1 -> (forall x: P(x))
            Implies(w1, Forall(x, P(x))),
            # w2 -> (Exists x: P(x))
            Implies(w2, Exists(x, P(x)))
        ]
        weights = [
            1.,
            1.,
            1.
        ]
        axioms = torch.stack([torch.squeeze(ax) for ax in axioms])
        weights = torch.tensor(weights)
        sat_level = torch.sum(weights * axioms) / torch.sum(weights)

        return sat_level, axioms

    # training of the model
    params = [a, b, c]
    optimizer = torch.optim.SGD(params, lr=0.01)

    for epoch in range(1000):
        optimizer.zero_grad()
        sat_agg = axioms(a, b, c)
        loss = 1. - sat_agg[0]
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print("Epoch %d: Sat Level %.3f" % (epoch, sat_agg[0]))
    print("Training finished at Epoch %d with Sat Level %.3f " % (epoch, sat_agg[0]))

    # notice that the torch.clamp() is performed before passing the propositional variables through the knowledge base.
    # this choice has been done because PyTorch does not allow to perform the clamp after the backward phase.
    # for this reason, after the last epoch, the propositional variables could have a value out of [0., 1.] since a
    # backward phase has been performed. Therefore, in order to print the final output we need to clamp the propositional
    # variables again.
    a = torch.clamp(a, 0., 1.)
    a.free_variables = []
    b = torch.clamp(b, 0., 1.)
    b.free_variables = []
    c = torch.clamp(c, 0., 1.)
    c.free_variables = []

    # print the value of the propositional variables after the training
    print("a:", a.item())
    print("b:", b.item())
    print("c:", c.item())
    # print the value of the axioms after the training
    print(axioms(a, b, c)[1].detach().numpy())


if __name__ == "__main__":
    main()