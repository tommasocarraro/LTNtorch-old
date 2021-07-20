# Logic Tensor Networks (LTN)

Logic Tensor Network (LTN) is a neuro-symbolic framework that supports querying, learning and reasoning with both rich 
data and rich abstract knowledge about the world.
LTN uses a differentiable first-order logic language, called Real Logic, to incorporate data and logic.
Real Logic defines a `grounding`, which is a mapping from the logical domain (non-logical and logical symbols) to 
tensors in the Real field or operations (mathematical functions, neural networks, etc.) on tensors. Constants 
are mapped into tensors, variables are mapped into sequences of tensors, functions are mapped into mathematical
functions which take as input some tensors and return a tensor, formulas (atomic or not) are mapped into a value in
[0., 1.]. Examples of possible groundings are showed in the following figure. In the figure, `friend(Mary, John)` is an
atomic formula (predicate), while `∀x(friend(John, x) → friend(Mary, x))` is a closed formula (all the variables are
quantified). The letter `G` is the grounding, a function which maps the logical domain into the Real domain.

![Grounding_illustration](./docs/img/framework_grounding.png)

LTN converts Real Logic formulas (e.g. `∀x(cat(x) → ∃y(partOf(x,y)∧tail(y)))`) into [PyTorch](https://www.pytorch.org/) 
computational graphs. Such formulas can express complex queries about the data, prior knowledge to satisfy during 
learning, statements to prove, etc. An example on how LTN converts such formulas into PyTorch graphs is showed in the
following figure.

![Computational_graph_illustration](./docs/img/framework_computational_graph.png)

Using LTN, one can represent and effectively compute some of the most important tasks of deep learning. Examples of such 
tasks are classification, regression, clustering, or link prediction.
The ["Getting Started"](#getting-started) section of the README links to tutorials and examples of LTN code.

[[Paper]](https://arxiv.org/pdf/2012.13635.pdf)
```
@misc{badreddine2021logic,
      title={Logic Tensor Networks}, 
      author={Samy Badreddine and Artur d'Avila Garcez and Luciano Serafini and Michael Spranger},
      year={2021},
      eprint={2012.13635},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}
```


## Installation

Clone the LTN repository and install it using `pip install -e <local project path>`.

Following are the dependencies used for development (similar versions should run fine):
- python 3.9
- torch >= 1.9.0
- numpy >= 1.21.1
- matplotlib >= 3.4.2
- pandas >= 1.3.0
- scikit-learn >= 0.24.2
- torchvision >= 0.10.0

## Repository structure

- `logictensornetworks/core.py` -- core system for defining constants, variables, predicates, functions and formulas;
- `logictensornetworks/fuzzy_ops.py` -- a collection of fuzzy logic operators defined using PyTorch primitives;
- `logictensornetworks/utils.py` -- a collection of useful functions used in the examples;
- `tutorials/` -- tutorials to start with LTN;
- `examples/` -- various problems approached using LTN.

## Getting Started

### Tutorials

`tutorials/` contains a walk-through of LTN. In order, the tutorials cover the following topics:
1. [Grounding in LTN part 1](https://nbviewer.jupyter.org/github/bmxitalia/LTNtorch/blob/original-implementation/tutorials/1-grounding_non_logical_symbols.ipynb): Real Logic, constants, predicates, functions, variables,
2. [Grounding in LTN part 2](https://nbviewer.jupyter.org/github/bmxitalia/LTNtorch/blob/original-implementation/tutorials/2-grounding_connectives.ipynb): connectives and quantifiers (+ [complement](https://nbviewer.jupyter.org/github/bmxitalia/LTNtorch/blob/original-implementation/tutorials/2b-operators_and_gradients.ipynb): choosing appropriate operators for learning),
3. [Learning in LTN](https://nbviewer.jupyter.org/github/bmxitalia/LTNtorch/blob/original-implementation/tutorials/3-knowledgebase_and_learning.ipynb): using satisfiability of LTN formulas as a training objective,
4. [Reasoning in LTN](https://nbviewer.jupyter.org/github/bmxitalia/LTNtorch/blob/original-implementation/tutorials/4-reasoning.ipynb): measuring if a formula is the logical consequence of a knowledgebase.

The tutorials are implemented using jupyter notebooks.

### Examples

`examples/` contains a series of experiments. Their objective is to show how the language of Real Logic can be used to specify a number of tasks that involve learning from data and reasoning about logical knowledge. Examples of such tasks are: classification, regression, clustering, link prediction.

- The `binary classification` example illustrates in the simplest setting how to ground a binary classifier as a predicate in LTN, and how to feed batches of data during training;
- The `multiclass classification` examples (single-label, multi-label) illustrate how to ground predicates that can classify samples in several classes;
- The `MNIST digit addition` example showcases the power of a neurosymbolic approach in a classification task that only provides groundtruth for some final labels (result of the addition), where LTN is used to provide prior knowledge about intermediate labels (possible digits used in the addition);
- The `regression` example illustrates how to ground a regressor as a function symbol in LTN;
- The `clustering` example illustrates how LTN can solve a task using first-order constraints only, without any label being given through supervision;
- The `Smokes Friends Cancer` example is a classical link prediction problem of Statistical Relational Learning where LTN learns embeddings for individuals based on fuzzy groundtruths and first-order constraints.

The examples are presented using Python scripts.

## License

This project is licensed under the MIT License - see the [LICENSE](https://nbviewer.jupyter.org/github/bmxitalia/LTNtorch/blob/original-implementation/LICENSE) file for details.

## Acknowledgements

LTN has been developed thanks to active contributions and discussions with the following people (in alphabetical order):
- Alessandro Daniele (FBK)
- Artur d’Avila Garcez (City)
- Benedikt Wagner (City)
- Emile van Krieken (VU Amsterdam)
- Francesco Giannini (UniSiena)
- Giuseppe Marra (UniSiena)
- Ivan Donadello (FBK)
- Lucas Bechberger (UniOsnabruck)
- Luciano Serafini (FBK)
- Marco Gori (UniSiena)
- Michael Spranger (Sony AI)
- Michelangelo Diligenti (UniSiena)
- Samy Badreddine (Sony AI)
- Tommaso Carraro (FBK)
