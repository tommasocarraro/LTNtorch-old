import torch
import torch.nn as nn


def constant(value, trainable=False):
    """Returns a torch.tensor that represents the grounding of an LTN constant (grounded with the value given in input).

    An ltn constant denotes an individual grounded as a tensor in the Real field.
    The individual can be pre-defined (fixed data point) or learnable (embedding).

    Args:
        value: a value to feed in the tensor. The value becomes the grounding of the individual.
        trainable: whether the LTN constant is trainable or not. If False, the subgraph containing the constant
        will be excluded from the gradient computation. Defaults to False.
    """
    const = torch.tensor(value)
    if trainable:
        const.requires_grad = True

    const.active_doms = []  # this adds a new attribute to the torch.tensor. This attribute is an empty list because
    # a constant does not have variables in it, so there is no need for a label.

    return const


def variable(label, individuals_seq):
    """Returns a torch.tensor that represents the grounding of an LTN variable (grounded with the sequence of
    individuals given in input).

    A ltn variable denotes a sequence of individuals.
    Axis 0 is the batch dimension: if `x` is an `ltn.variable`, `x[0]` gives the first individual,
    `x[1]` gives the second individual, and so forth, the usual way.

    Args:
        label: string. In ltn, variables need to be labelled.
        individuals_seq: A sequence of individuals to feed in a tensor.
            Alternatively, a tensor to use as is (with some dynamically added attributes, like active_doms).
    """
    if label.startswith("diag"):
        raise ValueError("Labels starting with diag are reserved.")
    if isinstance(individuals_seq, torch.FloatTensor):
        var = individuals_seq
    else:
        var = torch.tensor(individuals_seq)

    if len(var.shape) == 1:
        # add a dimension if there is only one individual in the sequence, since axis 0 represents the batch dimension
        var = var.view(1, var.shape[0])
    var.latent_dom = label
    var.active_doms = [label]

    return var


def get_dim0_of_dom(grounding, dom):
    """Returns the number of values that the domain takes in the input grounding (tensor).
    """
    return grounding.size(grounding.active_doms.index(dom))


def cross_grounding_values(groundings, flatten_dim0=False):
    """
    This function creates the combination of all the possible values of the groundings given in input.

    It returns a list of tensors containing the combinations of values of the input groundings. Each one
     of these tensors is a component of the combination. If these tensors are concatenated along axis 0, the combinations
     are generated. The output list contains one tensor per input grounding.

    Moreover, it returns a list of variable labels and a list containing the number of individuals for each variable.
    The variable labels correspond to the variables of which the groundings have been passed in input.

    Args:
        groundings: list of groundings of potentially different sizes for which the combination of values have to
        be generated. These groundings can be ltn variables, constants, functions, predicates, or any expression built
        on those.
        flatten_dim0: if True, it removes the first dimension from the output tensors and flat it. For example, if one
        output tensor has size [3, 2, 2], if flatten_dim0 is set to True, its size becomes [6, 2].
    """
    doms_to_dim0 = {}
    for grounding in groundings:
        for dom in grounding.active_doms:
            doms_to_dim0[dom] = get_dim0_of_dom(grounding, dom)
    doms = list(doms_to_dim0.keys())
    dims0 = list(doms_to_dim0.values())
    crossed_groundings = []
    for grounding in groundings:
        doms_in_grounding = list(grounding.active_doms)
        doms_not_in_grounding = list(set(doms).difference(doms_in_grounding))
        for new_dom in doms_not_in_grounding:
            new_idx = len(doms_in_grounding)
            grounding = torch.unsqueeze(grounding, dim=new_idx)
            grounding = torch.repeat_interleave(grounding, repeats=doms_to_dim0[new_dom], dim=new_idx)
            doms_in_grounding.append(new_dom)
        perm = [doms_in_grounding.index(dom) for dom in doms] + list(range(len(doms_in_grounding), len(grounding.shape)))
        grounding = grounding.permute(perm)
        grounding.active_doms = doms
        if flatten_dim0:
            shape_list = [-1] + list(grounding.shape[len(doms_in_grounding)::])
            grounding = torch.reshape(grounding, shape=tuple(shape_list))
        crossed_groundings.append(grounding)
    return crossed_groundings, doms, dims0


class Predicate(nn.Module):
    """Predicate class for ltn.

    An ltn predicate is a mathematical function (either pre-defined or learnable) that maps
    from some n-ary domain of individuals to a real from [0,1] that can be interpreted as a truth value.
    Examples of predicates can be similarity measures, classifiers, etc.

    Predicates can be defined using any operations in Pytorch. They can be linear functions, Deep Neural Networks,
    and so forth.

    An ltn predicate implements a `nn.Module` instance that can "broadcast" ltn terms as follows:
    1. Evaluating a predicate with one variable of n individuals yields n output values,
    where the i-th output value corresponds to the term calculated with the i-th individual.
    2. Evaluating a predicate with k variables (x1,...,xk) with respectively n1,...,nk
    individuals each, yields a result with n1*...*nk values. The result is organized in a tensor
    where the first k dimensions can be indexed to retrieve the outcome(s) that correspond to each variable.
    The tensor output of a predicate has a dynamically added attribute `active_doms`
    that tells which axis corresponds to which variable (using the label of the variable).

    Attributes:
        model: The wrapped PyTorch model, without the ltn-specific broadcasting.
    """

    def __init__(self, model):
        """Initializes the ltn predicate with the given nn.Module instance,
        wrapping it with the ltn-broadcasting mechanism."""
        super(Predicate, self).__init__()
        self.model = model

    def forward(self, inputs, *args, **kwargs):
        """Encapsulates the "self.model.forward()" to handle the ltn-broadcasting.

        Args:
            inputs: tensor or list of tensors that are ltn terms (ltn variable, ltn constant or
                    output of a ltn functions).
        Returns:
            outputs: tensor of truth values, with dimensions s.t. each variable corresponds to one axis.
        """
        if not isinstance(inputs, (list, tuple)):
            # if inputs is not a list of groundings (in the case we have only one input for the predicate),
            # cross_grounding_values is used only to compute doms and dims_0
            inputs, doms, dims_0 = cross_grounding_values([inputs], flatten_dim0=True)
            inputs = inputs[0]
        else:
            inputs, doms, dims_0 = cross_grounding_values(inputs, flatten_dim0=True)
        print(inputs.shape)
        # TODO chiedere a Luciano cosa succede se passo una matrice e vorrei avere il valore del predicato
        outputs = self.model(inputs, *args, **kwargs)
        print(outputs)
        # TODO capire questa cosa
        print(dims_0)
        if dims_0:
            outputs = torch.reshape(outputs, tuple(dims_0)) # qua ho visto che non un transpose ottengo lo stesso risultato

        outputs.active_doms = doms
        return outputs

    @classmethod
    def Lambda(cls, lambda_operator):
        """Constructor that takes in argument a lambda function. It is appropriate for small
        non-trainable mathematical operations that return a value in [0,1]."""
        model = LambdaModel(lambda_operator)
        return cls(model)

    @classmethod
    def MLP(cls, layer_dims=(16, 16, 1)):
        layers = []
        for i in range(1, len(layer_dims)):
            layers.append(nn.Linear(layer_dims[i - 1], layer_dims[i]))
            if i != (len(layer_dims) - 1):
                layers.append(nn.ELU())
            else:
                layers.append(nn.Sigmoid())
        model = nn.Sequential(*layers)
        return cls(model)
