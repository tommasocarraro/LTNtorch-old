import torch
from torch import nn

# TODO definire dominio con shape e sample
# TODO fare in modo che le costanti, variabili, funzioni e predicati prendano in input i propri domini, in questo
# TODO modo si riesce ad evitare di tenere taccia delle dimensioni quando si creano le reti neurali


class Domain(object):
    """Domain class for ltn.

    An ltn domain defines the type of a constant, variable, function, or predicate. Intuitively, a domain could define
    the possible values that a constant or variable can assume, the possible values that a function can take as input and
    produce as output, and the possible values that a predicate can take as input.

    Args:
        shape: it is the shape of the domain. It must be a tuple of integers. For example, shape (3,4,2) defines the
        domain of tensors of dimension (3,4,2). Notice that the shape defined the grounding of the domain. In
        fact, a domain symbol is grounded as a set of tensor of size shape.
        domain_name: it is a string containing the name of the domain, for example, 'people'.
    Attributes:
        shape: see shape argument.
        domain_name: see domain_name argument.
    """
    def __init__(self, shape, domain_name):
        if not isinstance(shape, tuple) and all(isinstance(v, int) for v in shape):
            raise ValueError("The shape attribute must be a tuple of integers.")
        self.shape = shape
        self.domain_name = domain_name

    def __repr__(self):
        return "Domain(domain_name='" + self.domain_name + "', grounding=R^" + str(self.shape) + ")"

    # TODO sample method to sample from the domain with a given distribution
    '''
    def sample(self, distribution, n=100):
        """
        It samples n samples from the distribution given in input
        Args:
            distribution: the distribution from which the samples have to be sampled
            n: the number of samples to be sampled from the distribution
        """
    '''


class Constant(object):
    # TODO capire se aggiungere la batch dimension anche per la costante
    """Constant class for ltn.

    An ltn constant denotes an individual grounded as a tensor in the Real field.
    The individual can be pre-defined (fixed data point) or learnable (embedding).

    Args:
        constant_name: string containing the name of the constant.
        domain: it is the domain of the LTN constant.
        value: the value that becomes the grounding of the LTN constant. The value becomes the grounding of the
        individual represented by the constant.
        trainable: whether the LTN constant is trainable or not. If False, the subgraph containing the constant
        will be excluded from the gradient computation. Defaults to False. If True, the constant is initialized using the
        value parameter.
    Attributes:
        constant_name: see constant_name argument.
        grounding: it is the grounding of the LTN constant. Specifically, it is a torch.tensor with shape depending on
        the domain of the constant.
        domain: see the domain argument.
        free_variables: it is a list of string containing the labels of the free variables contained in the expression.
        In the case of a constant, free_variables is empty since a constant does not contain variables.
    """
    def __init__(self, constant_name, domain, value, trainable=False):
        value = torch.tensor(value, requires_grad=trainable)
        if value.shape != domain.shape:
            raise ValueError("The value given for the constant does not match the constant's domain. The shape of the "
                             " value must match the shape of the constant's domain.")
        self.constant_name = constant_name
        self.grounding = value
        self.domain = domain
        self.free_variables = []

    def __repr__(self):
        return "Constant(constant_name='" + self.constant_name + "', domain=" + repr(self.domain) + ", grounding=" \
               + str(self.grounding) + ", free_variables=" + str(self.free_variables) + ")"


class Variable(object):
    # TODO capire a cosa serve latent_dom
    """Variable class for ltn.

    An ltn variable denotes a sequence of individuals. It is grounded as a sequence of tensors (groundings of
    individuals) in the real field.
    Axis 0 is the batch dimension: if `x` is an `ltn.Variable`, `x[0]` gives the first individual,
    `x[1]` gives the second individual, and so forth, i.e., the usual way.

    Args:
        variable_name: it is a string containing the name of the variable, for example 'x'.
        domain: it is the domain of the LTN variable.
        individual_seq: it is a sequence of individuals (sequence of tensors) to ground the ltn variable.
            Alternatively, a tensor to use as is.
    Attributes:
        grounding: it is the grounding of the LTN variable. Specifically, it is a torch.tensor with shape depending on
        the domain of the variable.
        domain: see the domain argument.
        free_variables: it is a list of string containing the labels of the free variables contained in the expression.
        In this case, since we have just a variable, free_variables will contain the variable itself.
    """
    def __init__(self, variable_name, domain, individuals_seq):
        if isinstance(individuals_seq, torch.FloatTensor):
            grounding = individuals_seq
        else:
            grounding = torch.tensor(individuals_seq)
        if grounding[0].shape != domain.shape:
            raise ValueError("The shape of the given individuals does not match the shape of the variable's domain. "
                             " The shape of the individuals must match the shape of the variable's domain.")
        if len(grounding.shape) == 1:
            # add a dimension if there is only one individual in the sequence, since axis 0 represents the batch dimension
            grounding = grounding.view(1, grounding.shape[0])

        self.grounding = grounding
        self.domain = domain
        if variable_name.startswith("diag"):
            raise ValueError("Labels starting with diag are reserved.")
        self.variable_name = variable_name
        self.free_variables = [variable_name]

    def __repr__(self):
        return "Variable(variable_name='" + self.variable_name + "', domain=" + repr(self.domain) + \
               ", individuals_number=" + str(self.grounding.shape[0]) + ", grounding=" + str(self.grounding) + \
               ", free_variables=" + str(self.free_variables) + ")"


def get_n_individuals_of_var(symbol, var):
    """Returns the number of individuals of the variable var contained in the grounding of the symbol given in input.
    Here, var is needed to specify the axis of the variable in the grounding (tensor).
    """
    return symbol.grounding.size(symbol.free_variables.index(var))


def cross_grounding_values_of_symbols(symbols, flatten_dim0=False):
    """
    This function creates the combination of all the possible values of the groundings of the symbols given in input.
    These symbols can be ltn variables, constants, functions, predicates, or any expression built on those.

    It returns a list of tensors containing the combinations of values of the groundings of the input symbols. Each one
    of these tensors is a component of the combination. If these tensors are concatenated along axis 0, the combinations
    are generated. The output list contains one tensor per input symbol.

    Moreover, it returns a list of variable labels and a list containing the number of individuals for each variable.
    The variable labels correspond to the variables contained in the groundings of the symbols that have been passed
    in input.

    Args:
        symbols: list of symbols of potentially different sizes for which the combination of values of the groundings
        have to be generated. These symbols can be ltn variables, constants, functions, predicates, or any expression
        built on those.
        flatten_dim0: if True, it removes the first dimension from the output tensors and flat it. For example, if one
        output tensor has size [3, 2, 2], if flatten_dim0 is set to True, its size becomes [6, 2].
    """
    vars_to_n_individuals = {}
    for symbol in symbols:
        for var in symbol.free_variables:
            vars_to_n_individuals[var] = get_n_individuals_of_var(symbol, var)
    vars = list(vars_to_n_individuals.keys())
    n_individuals_per_var = list(vars_to_n_individuals.values())
    crossed_symbol_groundings = []
    for symbol in symbols:
        vars_in_symbol = list(symbol.free_variables)
        vars_not_in_symbol = list(set(vars).difference(vars_in_symbol))
        symbol_grounding = symbol.grounding
        for new_var in vars_not_in_symbol:
            new_idx = len(vars_in_symbol)
            symbol_grounding = torch.unsqueeze(symbol_grounding, dim=new_idx)
            symbol_grounding = torch.repeat_interleave(symbol_grounding, repeats=vars_to_n_individuals[new_var],
                                                       dim=new_idx)
            vars_in_symbol.append(new_var)
        perm = [vars_in_symbol.index(var) for var in vars] + list(range(len(vars_in_symbol),
                                                                        len(symbol_grounding.shape)))
        symbol_grounding = symbol_grounding.permute(perm)
        symbol.free_variables = vars
        if flatten_dim0:
            shape_list = [-1] + list(symbol_grounding.shape[len(vars_in_symbol)::])
            symbol_grounding = torch.reshape(symbol_grounding, shape=tuple(shape_list))
        crossed_symbol_groundings.append(symbol_grounding)
    return crossed_symbol_groundings, vars, n_individuals_per_var


class Predicate(nn.Module):
    """Predicate class for ltn.

    An ltn predicate is a mathematical function (either pre-defined or learnable) that maps
    from some n-ary domain of individuals to a real number in [0,1] (fuzzy) that can be interpreted as a truth value.
    Examples of predicates can be similarity measures, classifiers, etc.

    Predicates can be defined using any operations in Pytorch. They can be linear functions, Deep Neural Networks,
    and so forth.

    An ltn predicate implements a `nn.Module` instance that can "broadcast" ltn terms as follows:
    1. Evaluating a predicate with one variable of n individuals yields n output values,
    where the i-th output value corresponds to the term calculated with the i-th individual.
    2. Evaluating a predicate with k variables (x1,...,xk) with respectively n1,...,nk
    individuals each, yields a result with n1*...*nk values. The result is organized in a tensor
    where the first k dimensions can be indexed to retrieve the outcome(s) that correspond to each variable.
    The attribute free_variables tells which axis corresponds to which variable in the tensor output by
    the predicate (using the name of the variable).

    Args:
        predicate_name: string containing the name of the predicate;
        input_domain: the domain of the input of the predicate;
        model: model that becomes the grounding of the predicate;

    Attributes:
        predicate_name: see predicate_name argument;
        input_domain: see input_domain argument;
        grounding: the grounding of the ltn predicate.
    """

    def __init__(self, predicate_name, input_domain, model):
        """Initializes the ltn predicate with the given nn.Module instance,
        wrapping it with the ltn-broadcasting mechanism."""
        super(Predicate, self).__init__()
        self.predicate_name = predicate_name
        self.input_domain = input_domain

        self.grounding = model

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
            inputs, doms, dims_0 = cross_grounding_values_of_symbols([inputs], flatten_dim0=True)
            inputs = inputs[0]
        else:
            inputs, doms, dims_0 = cross_grounding_values_of_symbols(inputs, flatten_dim0=True)
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