import torch
import numpy as np
import ltn
import logging
logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)


def main():
    # LTN constants which represent the individuals in the dataset, namely the individuals (a, ..., n)
    embedding_size = 10  # size of the embedding, each individual is an LTN constant which can be interpreted as
    # an embedding

    # first group of people (from a to h)
    g1 = {person: ltn.constant(np.random.uniform(low=0.0, high=1.0, size=embedding_size), trainable=True) for person
          in 'abcdefgh'}
    # second group of people (from i to n)
    g2 = {person: ltn.constant(np.random.uniform(low=0.0, high=1.0, size=embedding_size), trainable=True) for person
          in 'ijklmn'}
    # dictionary which contains all the people
    g = {**g1, **g2}

    # Predicates of the Smokes-Friends-Cancer problem

    Smokes = ltn.Predicate(layers_size=(10, 16, 16, 1))
    Friends = ltn.Predicate(layers_size=(20, 16, 16, 1))  # the input size is 20 since Friends takes as input
    # two people
    Cancer = ltn.Predicate(layers_size=(10, 16, 16, 1))

    # these list represent the facts of the knowledge base
    friends = [('a', 'b'), ('a', 'e'), ('a', 'f'), ('a', 'g'), ('b', 'c'), ('c', 'd'), ('e', 'f'), ('g', 'h'),
               ('i', 'j'), ('j', 'm'), ('k', 'l'), ('m', 'n')]
    smokes = ['a', 'e', 'f', 'g', 'j', 'n']
    cancer = ['a', 'e']

    # Connectives, quantifiers and aggregators

    Not = ltn.WrapperConnective(ltn.fuzzy_ops.NotStandard())
    And = ltn.WrapperConnective(ltn.fuzzy_ops.AndProd())
    Or = ltn.WrapperConnective(ltn.fuzzy_ops.OrProbSum())
    Implies = ltn.WrapperConnective(ltn.fuzzy_ops.ImpliesGoguenStrong())
    Forall = ltn.WrapperQuantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantification_type="forall")
    Exists = ltn.WrapperQuantifier(ltn.fuzzy_ops.AggregPMean(p=6), quantification_type="exists")

    formula_aggregator = ltn.fuzzy_ops.AggregPMeanError(p=2)

    # defining the theory which codifies the knowledge base
    def axioms(p_exists):
        # variables which represents all the people
        # since the embeddings change during the training, we need to re-istantiate these two variables at every step
        p = ltn.variable("p", torch.stack(list(g.values())))
        q = ltn.variable("q", torch.stack(list(g.values())))
        axioms = []

        # Friends: knowledge incomplete in that
        #     Friend(x,y) with x<y may be known
        #     but Friend(y,x) may not be known

        axioms.append(formula_aggregator(torch.stack([Friends([g[x], g[y]]) for (x, y) in friends]), dim=0))

        axioms.append(formula_aggregator(torch.stack(
                [Not(Friends([g[x], g[y]])) for x in g1 for y in g1 if (x, y) not in friends and x < y] +
                [Not(Friends([g[x], g[y]])) for x in g2 for y in g2 if (x, y) not in friends and x < y]), dim=0))

        # Smokes: knowledge complete
        axioms.append(formula_aggregator(torch.stack([Smokes(g[x]) for x in smokes]), dim=0))
        axioms.append(formula_aggregator(torch.stack([Not(Smokes(g[x])) for x in g if x not in smokes]), dim=0))

        # Cancer: knowledge complete in g1 only
        axioms.append(formula_aggregator(torch.stack([Cancer(g[x]) for x in cancer]), dim=0))
        axioms.append(formula_aggregator(torch.stack([Not(Cancer(g[x])) for x in g1 if x not in cancer]), dim=0))

        # friendship is anti-reflexive
        axioms.append(Forall(p, Not(Friends([p, p])), p=5))
        # friendship is symmetric
        axioms.append(Forall([p, q], Implies(Friends([p, q]), Friends([q, p])), p=5))
        # everyone has a friend
        axioms.append(Forall(p, Exists(q, Friends([p, q]), p=p_exists)))
        # smoking propagates among friends
        axioms.append(Forall([p, q], Implies(And(Friends([p, q]), Smokes(p)), Smokes(q))))
        # smoking causes cancer + not smoking causes not cancer
        axioms.append(Forall(p, Implies(Smokes(p), Cancer(p))))
        axioms.append(Forall(p, Implies(Not(Smokes(p)), Not(Cancer(p)))))
        # computing sat_level
        axioms = torch.stack([torch.squeeze(ax) for ax in axioms])
        sat_level = formula_aggregator(axioms, dim=0)
        return sat_level

    def phi1():
        p = ltn.variable("p", torch.stack(list(g.values())))
        return Forall(p, Implies(Cancer(p), Smokes(p)), p=5)

    def phi2():
        p = ltn.variable("p", torch.stack(list(g.values())))
        q = ltn.variable("q", torch.stack(list(g.values())))
        return Forall([p, q], Implies(Or(Cancer(p), Cancer(q)), Friends([p, q])), p=5)

    # # Training

    print(g.values())
    # TODO sistemare questo problema dovuto a PyTorch
    params = list(Smokes.parameters()) + list(Friends.parameters()) + list(Cancer.parameters()) + list(g.values())
    optimizer = torch.optim.Adam(params, lr=0.001)

    for epoch in range(1000):
        print(g['a'])
        if epoch <= 200:
            p_exists = 1
        else:
            p_exists = 6
        optimizer.zero_grad()
        sat_agg = axioms(p_exists)
        loss = 1. - sat_agg
        loss.backward()
        optimizer.step()

        # we print metrics every 20 epochs of training
        if epoch % 20 == 0:
            logger.info(" epoch %d | loss %.4f | Train Sat %.3f | Phi1 Sat %.3f | Phi2 Sat %.3f", epoch, loss,
                        sat_agg, phi1(), phi2())


if __name__ == "__main__":
    main()