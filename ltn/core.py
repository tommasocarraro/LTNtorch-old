import torch
from torch import nn
import math
import copy

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
        if not isinstance(shape, list) and all(isinstance(v, int) for v in shape):
            raise ValueError("The shape attribute must be a list of integers.")
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
        the domain of the constant. The grounding has a dynamically added attribute called free_variables, which
        contains a list of strings of the labels of the free variables contained in the expression. In the case of a
        constant, free_variables is empty since a constant does not contain variables.
        domain: see the domain argument.
    """
    def __init__(self, constant_name, domain, value, trainable=False):
        value = torch.tensor(value, requires_grad=trainable)
        if value.shape != torch.Size(domain.shape):
            raise ValueError("The value given for the constant does not match the constant's domain. The shape of the "
                             "value must match the shape of the constant's domain.")
        self.constant_name = constant_name
        self.grounding = value
        self.domain = domain
        self.grounding.free_variables = []

    def __repr__(self):
        return "Constant(constant_name='" + self.constant_name + "', domain=" + repr(self.domain) + ", grounding=" \
               + str(self.grounding) + ", grounding_free_variables=" + str(self.grounding.free_variables) + ")"

    def get_grounding(self):
        return self.grounding


class Variable(object):
    # TODO capire a cosa serve latent_dom nel codice orginale
    """Variable class for ltn.

    An ltn variable denotes a sequence of individuals. It is grounded as a sequence of tensors (groundings of
    individuals) in the real field.
    Axis 0 is the batch dimension: if `x` is an `ltn.Variable`, `x[0]` gives the first individual,
    `x[1]` gives the second individual, and so forth, i.e., the usual way.

    Args:
        variable_name: it is a string containing the name of the variable, for example 'x'.
        domain: it is the domain of the LTN variable.
        individuals_seq: it is a sequence of individuals (sequence of tensors) to ground the ltn variable.
            Alternatively, a tensor to use as is.
    Attributes:
        grounding: it is the grounding of the LTN variable. Specifically, it is a torch.tensor with shape depending on
        the domain of the variable. The grounding has a dynamically added attribute called free_variables, which
        contains a list of strings of the labels of the free variables contained in the expression. In this case, since
        we have just a variable, free_variables will contain the variable itself.
        domain: see the domain argument.
    """
    def __init__(self, variable_name, domain, individuals_seq):
        if isinstance(individuals_seq, torch.FloatTensor):
            grounding = individuals_seq
        else:
            grounding = torch.tensor(individuals_seq)
        if grounding[0].shape != torch.Size(domain.shape):
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
        self.grounding.free_variables = [variable_name]

    def __repr__(self):
        return "Variable(variable_name='" + self.variable_name + "', domain=" + repr(self.domain) + \
               ", individuals_number=" + str(self.grounding.shape[0]) + ", grounding=" + str(self.grounding) + \
               ", grounding_free_variables=" + str(self.grounding.free_variables) + ")"

    def get_grounding(self):
        return self.grounding


def get_n_individuals_of_var(grounding, var):
    """Returns the number of individuals of the variable var contained in the grounding given in input.
    Here, var is needed to specify the axis of the variable in the grounding (tensor).
    """
    return grounding.size(grounding.free_variables.index(var))


def cross_grounding_values_of_symbols(symbol_groundings, flatten_dim0=False):
    """
    This function creates the combination of all the possible values of the groundings given in input. These are
    the groundings of logical symbols or any expression built on them. These symbols can be ltn variables, constants,
    functions, predicates, or any expression built on those.

    It returns a list of tensors containing the combinations of values of the groundings given in input. Each one
    of these tensors is a component of the combination. If these tensors are concatenated along axis 1, the combinations
    are generated. The output list contains one tensor per input symbol.

    Moreover, it returns a list of variable labels and a list containing the number of individuals for each variable.
    The variable labels correspond to the variables contained in the groundings of the symbols that have been passed
    in input.

    Args:
        symbol_groundings: list of groundings of symbols of potentially different sizes for which the combination of
        values have to be generated. These groundings are related to symbols that can be ltn variables, constants,
        functions, predicates, or any expression built on those.
        flatten_dim0: if True, it removes the first dimension from the output tensors and flat it. For example, if one
        output tensor has size [3, 2, 2], if flatten_dim0 is set to True, its size becomes [6, 2].
    """
    vars_to_n_individuals = {}
    for grounding in symbol_groundings:
        for var in grounding.free_variables:
            vars_to_n_individuals[var] = get_n_individuals_of_var(grounding, var)
    vars = list(vars_to_n_individuals.keys())
    n_individuals_per_var = list(vars_to_n_individuals.values())
    crossed_symbol_groundings = []
    for grounding in symbol_groundings:
        vars_in_symbol = list(grounding.free_variables)
        vars_not_in_symbol = list(set(vars).difference(vars_in_symbol))
        symbol_grounding = grounding
        for new_var in vars_not_in_symbol:
            new_idx = len(vars_in_symbol)
            symbol_grounding = torch.unsqueeze(symbol_grounding, dim=new_idx)
            symbol_grounding = torch.repeat_interleave(symbol_grounding, repeats=vars_to_n_individuals[new_var],
                                                       dim=new_idx)
            vars_in_symbol.append(new_var)
        perm = [vars_in_symbol.index(var) for var in vars] + list(range(len(vars_in_symbol),
                                                                        len(symbol_grounding.shape)))
        symbol_grounding = symbol_grounding.permute(perm)
        grounding.free_variables = vars
        if flatten_dim0:
            shape_list = [-1] + list(symbol_grounding.shape[len(vars_in_symbol)::])
            symbol_grounding = torch.reshape(symbol_grounding, shape=tuple(shape_list))
        crossed_symbol_groundings.append(symbol_grounding)

    return crossed_symbol_groundings, vars, n_individuals_per_var


class LambdaModel(nn.Module):
    """ Simple `nn.Module` that implements a non-trainable model based on a lambda function.
    Used in `ltn.Predicate.lambda_operation` and `ltn.Function.lambda_operation`.
    """
    def __init__(self, lambda_func):
        super(LambdaModel, self).__init__()
        self.lambda_func = lambda_func

    def forward(self, x):
        return self.lambda_func(x)


class Predicate(nn.Module):
    # TODO descrivere bene cosa fa il metodo init e come usa i parametri
    """Predicate class for ltn.

    An ltn predicate is a mathematical function (either pre-defined or learnable) that maps
    from some n-ary domain of individuals to a real number in [0,1] (fuzzy) that can be interpreted as a truth value.
    Examples of predicates can be similarity measures, classifiers, etc.

    Predicates can be defined using any operations in PyTorch. They can be linear functions, Deep Neural Networks,
    and so forth.

    An ltn predicate implements a `nn.Module` instance that can "broadcast" ltn terms as follows:
    1. Evaluating a predicate with one variable of n individuals yields n output values,
    where the i-th output value corresponds to the predicate calculated with the i-th individual.
    2. Evaluating a predicate with k variables (x1,...,xk) with respectively n1,...,nk
    individuals each, yields a result with n1*...*nk values. The result is organized in a tensor
    where the first k dimensions can be indexed to retrieve the outcome(s) that correspond to each variable.
    The attribute free_variables tells which axis corresponds to which variable in the tensor output by
    the predicate (using the name of the variable).

    Args:
        predicate_name: string containing the name of the predicate;
        input_domain: list of domains of the inputs of the predicate;
        model: model that becomes the grounding of the predicate;
        layers_size: if a model is not given, it is possible to give layers_size and an MLP with that layers will be
        used as model;
        lambda_func: if a model is not given and layers_size is not given, it is possible to give a lambda function.
        In this case the lambda function will be used to define a non-trainable model for the ltn predicate.
    Attributes:
        predicate_name: see predicate_name argument;
        input_domain: see input_domain argument;
        grounding: the grounding of the ltn predicate. The grounding of a predicate is a non-trainable model implemented
        using a lambda function or a learnable model. When the groundings of the inputs are given to the predicate model,
        the model returns a real value in [0, 1] for each combination of the values of the groundings given in input.
        Then, at the output is attached a dynamic attribute called free_variables, which contains the list of free
        variables contained in the output tensor;
        model_type: it is a string containing the type of the model (linear, lambda or conv).
    """
    # TODO descrivere la lambda sui model type
    def __init__(self, predicate_name, input_domain, model=None, layers_size=None, lambda_func=None):
        """Initializes the ltn predicate with the given nn.Module instance,
        wrapping it with the ltn-broadcasting mechanism."""
        super(Predicate, self).__init__()
        if model is None and layers_size is None and lambda_func is None:
            raise ValueError("A model, or dimension of layers for constructing a model, or a lambda function to be used"
                             " as a non-trainable model should be given in input.")
        if model is not None and (layers_size is not None or lambda_func is not None):
            raise ValueError("A model has been given, so layers_size and lambda_func can't be given.")
        if model is None and layers_size is not None and lambda_func is not None:
            raise ValueError("Only one of layers_size and lambda_func can be given.")
        if model is None and layers_size is not None:
            model = self.MLP(layers_size)
            self.model_type = "linear"
        if model is None and lambda_func is not None:
            model = self.lambda_operation(lambda_func)
            self.model_type = "lambda"
        assert isinstance(input_domain, list), "The input_domain should be a list of domains."
        self.predicate_name = predicate_name
        self.input_domain = input_domain
        if isinstance(model, (nn.Sequential, nn.Module)) and self.model_type == "linear":
            model_layers = [layer for layer in model.modules()]
            first_layer = model_layers[1]  # in position 0 there is the copy of the model
            if isinstance(first_layer, nn.Linear):
                self.model_type = "linear"
                first_layer_size = first_layer.in_features
                flat_input_domain_size = sum([math.prod(list(domain.shape)) for domain in input_domain])
                if first_layer_size != flat_input_domain_size:
                    raise ValueError("The input layer size of the given model does not match the size of the input domain. "
                                     "The size of the input layer must match the size of the input domain (sum of sizes"
                                     " of input domains flattened).")

        # TODO fare meglio questa parte della convolution perche' e' complicata, per ora i controlli sulle conv non
        # sono implementati
        '''
        if isinstance(first_layer, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)):
            if len(input_domain.shape) != 3:
                raise ValueError("The given model is a CNN model, but the input domain does not correspond to an image."
                                 " An image should have three dimensions (width, height, depth). One or more dimensions "
                                 "are missed.")
        '''
        self.grounding = model

    def forward(self, inputs, *args, **kwargs):
        """Encapsulates the "self.model.forward()" to handle the ltn-broadcasting.

        Args:
            inputs: list of tensors that are ltn terms (ltn variable, ltn constant or
                    output of a ltn functions).
        Returns:
            outputs: tensor of truth values, with dimensions s.t. each variable corresponds to one axis.
        """
        assert isinstance(inputs, list), "The inputs parameter should be a list of tensors."
        inputs, vars, n_individuals_per_var = cross_grounding_values_of_symbols(inputs, flatten_dim0=True)
        if self.model_type == "linear":
            # qui devo fare il flat e la concatenazione degli input
            flat_inputs = [torch.flatten(x, start_dim=1) for x in inputs]
            inputs = torch.cat(flat_inputs, dim=1) if len(flat_inputs) > 1 else flat_inputs[0]
        if self.model_type == 'lambda':
            inputs = torch.cat(inputs, dim=0)
        outputs = self.grounding(inputs, *args, **kwargs)
        if n_individuals_per_var:
            # se ci sono delle variabili nella espressione di input, l'output diventa un tensore dove gli assi
            # corrispondono alle variabili
            outputs = torch.reshape(outputs, tuple(n_individuals_per_var))

        # TODO capire bene a cosa serve active doms perche' qui ho un active doms per predicato, invece forse ne serve uno per output
        outputs.free_variables = vars
        return outputs

    @staticmethod
    def lambda_operation(lambda_function):
        """It construct a simple and non-trainable mathematical operation using the lambda function given in input.
        It is appropriate for small non-trainable mathematical operations that return a value in [0,1]."""
        model = LambdaModel(lambda_function)
        return model

    @staticmethod
    def MLP(layer_dims=(16, 16, 1)):
        """
        It constructs a fully-connected MLP with the layers given in input.
        :param layer_dims: dimensions of the layers of the MLP.
        :return: an MLP with architecture defined by layers_dim parameter.
        """
        layers = []
        for i in range(1, len(layer_dims)):
            layers.append(nn.Linear(layer_dims[i - 1], layer_dims[i]))
            if i != (len(layer_dims) - 1):
                layers.append(nn.ELU())
            else:
                layers.append(nn.Sigmoid())
        model = nn.Sequential(*layers)
        return model

    def __repr__(self):
        return "Predicate(predicate_name='" + self.predicate_name + "', input_domain=" + repr(self.input_domain) + \
               ", grounding=" + str(self.grounding) + ")"


class Function(nn.Module):
    # TODO pensare a estensione in cui una funzione puo' avere piu' di un output
    """Function class for LTN.

    An ltn function is a mathematical function (pre-defined or learnable) that maps
    n individuals to one individual in the tensor domain.
    Examples of functions can be distance functions, regressors, etc.

    Functions can be defined using any operations in PyTorch.
    They can be linear functions, Deep Neural Networks, and so forth.

    An ltn function implements a `torch.nn.Module` instance that can "broadcast" ltn terms as follows:
    1. Evaluating a term with one variable of n individuals yields n output values,
    where the i-th output value corresponds to the term calculated with the i-th individual.
    2. Evaluating a term with k variables (x1,...,xk) with respectively n1,...,nk
    individuals each, yields a result with n1*...*nk values. The result is organized in a tensor
    where the first k dimensions can be indexed to retrieve the outcome(s) that correspond to each variable.
    The attribute free_variables tells which axis corresponds to which variable in the tensor output by
    the function (using the name of the variable).

    Args:
        function_name: string containing the name of the function;
        input_domain: list of domains of the inputs of the function;
        output_domain: domain of the output of the function;
        model: model that becomes the grounding of the predicate;
        layers_size: if a model is not given, it is possible to give layers_size and an MLP with that layers will be
        used as model;
        lambda_func: if a model is not given and layers_size is not given, it is possible to give a lambda function.
        In this case the lambda function will be used to define a non-trainable model for the ltn function.
    Attributes:
        function_name: see predicate_name argument;
        input_domain: see input_domain argument;
        output_domain: see output_domain argument;
        grounding: the grounding of the ltn function. The grounding of a function is a non-trainable model implemented
        using a lambda function or a learnable model. When the groundings of the inputs are given to the function model,
        the model returns a tensor in the output domain for each combination of the values of the groundings given in
        input. Then, at the output is attached a dynamic attribute called free_variables, which contains the list of
        free variables contained in the output tensor;
        model_type: it is a string containing the type of the model (linear, lambda or conv).
    """

    def __init__(self, function_name, input_domain, output_domain, model=None, layers_size=None, lambda_func=None):
        """Initializes the ltn function with the given nn.Module instance,
        wrapping it with the ltn-broadcasting mechanism."""
        super(Function, self).__init__()
        if model is None and layers_size is None and lambda_func is None:
            raise ValueError("A model, or dimension of layers for constructing a model, or a lambda function to be used"
                             " as a non-trainable model should be given in input.")
        if model is not None and (layers_size is not None or lambda_func is not None):
            raise ValueError("A model has been given, so layers_size and lambda_func can't be given.")
        if model is None and layers_size is not None and lambda_func is not None:
            raise ValueError("Only one of layers_size and lambda_func can be given.")
        if model is None and layers_size is not None:
            model = self.MLP(layers_size)
            self.model_type = "linear"
        if model is None and lambda_func is not None:
            model = self.lambda_operation(lambda_func)
            self.model_type = "lambda"
        assert isinstance(input_domain, list), "The input_domain should be a list of domains."
        self.function_name = function_name
        self.input_domain = input_domain
        self.output_domain = output_domain
        if isinstance(model, (nn.Sequential, nn.Module)) and self.model_type == "linear":
            model_layers = [layer for layer in model.modules()]
            first_layer = model_layers[1]  # in position 0 there is the copy of the model
            last_layer = model_layers[-1]
            if isinstance(first_layer, nn.Linear):
                self.model_type = "linear"
                first_layer_size = first_layer.in_features
                flat_input_domain_size = sum([math.prod(list(domain.shape)) for domain in input_domain])
                if first_layer_size != flat_input_domain_size:
                    raise ValueError(
                        "The input layer size of the given model does not match the size of the input domain. "
                        "The size of the input layer must match the size of the input domain (sum of sizes"
                        " of input domains flattened).")
            last_layer_size = last_layer.out_features
            flat_output_domain_size = math.prod(list(output_domain.shape))
            if last_layer_size != flat_output_domain_size:
                raise ValueError(
                    "The output layer size of the given model does not match the size of the output domain. "
                    "The size of the output layer must match the size of the output domain (sum of sizes"
                    " of output domains flattened)."
                )

        # TODO fare meglio questa parte della convolution perche' e' complicata, per ora i controlli sulle conv non
        # sono implementati
        '''
        if isinstance(first_layer, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)):
            if len(input_domain.shape) != 3:
                raise ValueError("The given model is a CNN model, but the input domain does not correspond to an image."
                                 " An image should have three dimensions (width, height, depth). One or more dimensions "
                                 "are missed.")
        '''
        self.grounding = model

    def forward(self, inputs, *args, **kwargs):
        """Encapsulates the "self.model.forward()" to handle the ltn-broadcasting.

        Args:
            inputs: list of tensors that are ltn terms (ltn variable, ltn constant or
                    output of a ltn functions).
        Returns:
            outputs: tensor of output values (each output is a tensor too), with dimensions s.t. each variable
            corresponds to one axis.
        """
        assert isinstance(inputs, list), "The inputs parameter should be a list of tensors."
        inputs, vars, n_individuals_per_var = cross_grounding_values_of_symbols(inputs, flatten_dim0=True)
        if self.model_type == "linear":
            # qui devo fare il flat e la concatenazione degli input
            flat_inputs = [torch.flatten(x, start_dim=1) for x in inputs]
            inputs = torch.cat(flat_inputs, dim=1) if len(flat_inputs) > 1 else flat_inputs[0]
        if self.model_type == 'lambda':
            inputs = torch.cat(inputs, dim=0)
        outputs = self.grounding(inputs, *args, **kwargs)
        # qui mi escono gli output flat, ora devo fare una reshape
        outputs = torch.reshape(outputs, [outputs.shape[0]] + self.output_domain.shape)
        if n_individuals_per_var:
            # se ci sono delle variabili nella espressione di input, l'output diventa un tensore dove gli assi
            # corrispondono alle variabili
            outputs = torch.reshape(outputs, tuple(n_individuals_per_var + list(outputs.shape[1::])))

        outputs.free_variables = vars
        return outputs

    @staticmethod
    def lambda_operation(lambda_function):
        """It construct a simple and non-trainable mathematical operation using the lambda function given in input.
        It is appropriate for small non-trainable mathematical operations that return a value in [0,1]."""
        model = LambdaModel(lambda_function)
        return model

    @staticmethod
    def MLP(layer_dims=(16, 16, 1)):
        """
        It constructs a fully-connected MLP with the layers given in input.
        :param layer_dims: dimensions of the layers of the MLP.
        :return: an MLP with architecture defined by layers_dim parameter.
        """
        layers = []
        for i in range(1, len(layer_dims)):
            layers.append(nn.Linear(layer_dims[i - 1], layer_dims[i]))
            if i != (len(layer_dims) - 1):
                layers.append(nn.ELU())
        model = nn.Sequential(*layers)
        return model

    def __repr__(self):
        return "Function(function_name='" + self.function_name + "', input_domain=" + repr(self.input_domain) + \
               ", output_domain=" + repr(self.output_domain) + ", grounding=" + str(self.grounding) + ")"