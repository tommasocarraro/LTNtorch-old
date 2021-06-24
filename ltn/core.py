import torch
from torch import nn
import math
import copy
import numpy as np

# TODO ricordarsi di mettere il seed per le cose random

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
        """
        This function returns a deep copy of the grounding of the LTN constant.
        :return: deep copy of the LTN constant grounding.
        """
        # here, a deep copy is needed because if it is not used cross_groundings_values() will modify the object instance
        ret_grounding = copy.deepcopy(self.grounding)
        ret_grounding.free_variables = self.grounding.free_variables
        return ret_grounding


class PropositionalVariable(object):
    # TODO capire come restringere il valore della variabile proposizionale tra 0 e 1, sembra che in PyTorch non si possa
    """PropositionalVariable class for ltn.

    An ltn propositional variable denotes a propositional logic variable grounded as a scalar in the Real field.

    Args:
        propositional_var_name: string containing the name of the propositional variable.
        truth_value: the value that becomes the grounding of the LTN propositional variable.
        trainable: whether the LTN propositional variable is trainable or not. If False, the subgraph containing the
        propositional variable will be excluded from the gradient computation. Defaults to False. If True, the
        propositional variable is initialized using the truth_value parameter.
    Attributes:
        propositional_var_name: see propositional_var_name argument.
        grounding: it is the grounding of the LTN propositional variable. Specifically, it is a torch.tensor containing
        the truth value of the propositional variable. The grounding has a dynamically added attribute called
        free_variables, which contains a list of strings of the labels of the free variables contained in the expression.
        In the case of a propositional variable, free_variables is empty since a propositional variable does not
        contain variables.
    """
    def __init__(self, propositional_var_name, truth_value, trainable=False):
        truth_value = torch.tensor(truth_value, requires_grad=trainable)
        assert len(truth_value.shape) == 1, "The truth value must be a scalar, not a vector, matrix or tensor."
        assert (truth_value <= 1 and truth_value >= 0), "The truth value must be in [0., 1.]"

        self.propositional_var_name = propositional_var_name
        self.grounding = truth_value
        self.grounding.free_variables = []

    def __repr__(self):
        return "PropositionalVariable(propositional_variable_name='" + self.propositional_var_name + \
               "', grounding=" + repr(self.grounding) + ", grounding_free_variables=" + \
               str(self.grounding.free_variables) + ")"

    def get_grounding(self):
        """
        This function returns a deep copy of the grounding of the LTN propositional variable.
        :return: deep copy of the LTN propositional variable grounding.
        """
        # here, a deep copy is needed because if it is not used cross_groundings_values() will modify the object instance
        ret_grounding = copy.deepcopy(self.grounding)
        ret_grounding.free_variables = self.grounding.free_variables
        return ret_grounding


class Variable(object):
    # TODO descrivere latent_variable una volta capito il significato
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
        self.grounding.latent_variable = variable_name

    def __repr__(self):
        return "Variable(variable_name='" + self.variable_name + "', domain=" + repr(self.domain) + \
               ", individuals_number=" + str(self.grounding.shape[0]) + ", grounding=" + str(self.grounding) + \
               ", grounding_free_variables=" + str(self.grounding.free_variables) + ")"

    def get_grounding(self):
        """
        This function returns a deep copy of the grounding of the LTN variable.
        :return: deep copy of the LTN variable grounding.
        """
        # here, a deep copy is needed because if it is not used cross_groundings_values() will modify the object instance
        ret_grounding = copy.deepcopy(self.grounding)
        ret_grounding.free_variables = self.grounding.free_variables
        ret_grounding.latent_variable = self.variable_name
        return ret_grounding


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
        symbol_grounding.free_variables = vars
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
            # TODO capire cosa fare con gli input nel caso della lambda come comportarsi, perche' ci sono dei problemi
            # TODO capire se va bene lasciare lista di tensori, in tal caso definire la lambda in maniera corretta
            # forse non bisogna fare nessuna trasformazione
            print()
            #inputs = torch.cat(inputs, dim=0)
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


def diag(variables_groundings):
    # TODO descrivere bene cosa fanno questi metodi, con tanto di input e output
    """Sets the given ltn variables for diagonal quantification (no broadcasting between these variables).

    Given 2 (or more) ltn variables, there are scenarios where one wants to express statements about
    specific pairs (or tuples) only, such that the i-th tuple contains the i-th instances of the variables.
    We allow this using `ltn.diag`.
    Note: diagonal quantification assumes that the variables have the same number of individuals.
    Given a predicate `P(x,y)` with two variables `x` and `y`,
    the usual broadcasting followed by an aggregation would compute (in Python pseudo-code):
        ```
        for i,x_i in enumerate(x):
            for j,y_j in enumerate(y):
                results[i,j]=P(x_i,y_i)
        aggregate(results)
        ```
    In contrast, diagonal quantification would compute:
        ```
        for i,(x_i, y_i) in enumerate(zip(x,y)):
            results[i].append(P(x_i,y_i))
        aggregate(results)
        ```
    Ltn computes only the "zipped" results.
    """
    # check if variables have the same number of individuals
    n_individuals = [var.shape[0] for var in variables_groundings]
    assert len(set(n_individuals)) == 1, "The given variables have a different number of individuals between each other." \
                                         " It is not possible to perform diagonal quantification between variables that" \
                                         " have a different number of individuals."
    diag_vars = "diag_" + "_".join([var.latent_variable for var in variables_groundings])
    for var in variables_groundings:
        var.free_variables = [diag_vars]
    return variables_groundings


def undiag(variables_groundings):
    """Resets the usual broadcasting strategy for the given ltn variables.

    In practice, `ltn.diag` is designed to be used with quantifiers.
    Every quantifier automatically calls `ltn.undiag` after the aggregation is performed,
    so that the variables keep their normal behavior outside of the formula.
    Therefore, it is recommended to use `ltn.diag` only in quantified formulas as follows:
        ```
        Forall(ltn.diag(x,l), C([x,l]))
        ```
    """
    for var in variables_groundings:
        var.free_variables = [var.latent_variable]
    return variables_groundings


class WrapperConnective:
    # TODO scrivere meglio la documentazione
    """Class to wrap binary connective operators to use them within ltn formulas.

    LTN supports various logical connectives. They are grounded using fuzzy semantics.
    The implementation of some common fuzzy logic operators using PyTorch primitives is in `ltn.fuzzy_ops`.
    The wrapper ltn.WrapperConnective allows to use the operators with LTN formulas.
    It takes care of combining sub-formulas that have different variables appearing in them
    (the sub-formulas may have different dimensions that need to be "broadcasted").
    Attributes:
        connective_operator: the original binary connective operator (without broadcasting).
    """

    def __init__(self, connective_operator):
        self.connective_operator = connective_operator

    def __call__(self, *symbol_groundings, **kwargs):
        # TODO capire a cosa serviva l'eccezione qui
        # TODO cambiare symbol_groundings in groundings perche' sono groundings
        symbol_groundings, vars, _ = cross_grounding_values_of_symbols(symbol_groundings)
        result = self.connective_operator(*symbol_groundings)
        result.free_variables = vars
        return result


class WrapperQuantifier:
    # TODO scrivere meglio la documentazione, soprattutto per la maschera
    # TODO vedere se cambiare symbol_grounding in grounding e basta, perche' la formula non e' un simbolo ma ha comunque un grounding
    """Class to wrap quantification operators to use them within ltn formulas.

    LTN supports universal and existential quantification. They are grounded using aggregation operators.
    The implementation of some common aggregators using PyTorch primitives is in `ltn.fuzzy_ops`.
    The wrapper allows to use the quantifiers with LTN formulas.
    It takes care of selecting the tensor dimensions to aggregate, given some variables in arguments.
    Additionally, boolean conditions (`mask_fn`,`mask_vars`) can be used for guarded quantification.
    Attributes:
        aggregation_operator: The original aggregation operator. It is a wrapper for the aggregation operator;
        quantifier: it is a string indicating the quantification that has to be performed (exists or forall)
    """

    def __init__(self, aggregation_operator, quantifier):
        self.aggregation_operator = aggregation_operator
        if quantifier not in ["forall", "exists"]:
            raise ValueError("The keyword for the quantifier should be \"forall\" or \"exists\".")
        self.quantifier = quantifier

    def __call__(self, variables_groundings, symbol_grounding, mask_vars=None, mask_fn=None, **kwargs):
        # TODO descrivere bene la documentazione del metodo, tipo variables_groundings sono i grounding delle variabili
        # TODO forse la quantificazione si applica solo ai predicati e anche i connettivi si applicano solo ai predicati
        # TODO quindi, correggere la documentazione di conseguenza
        # da quantificare, symbol_grounding e' il grounding del termine o predicato su cui fare quantificazione, mask_vars
        # e mask_fn servono per costruire la maschera per la guarded quantification. Mask_vars sono le variabili (sono groundings) su cui
        # fare guarded, mentre mask_fn e' la funzione da applicare come filtro sul grounding del predicato o termine.
        """
        mask_fn(mask_vars)
        """
        # verifico se ho una o piu' variabili su cui quantificare
        variables_groundings = [variables_groundings] if not isinstance(variables_groundings, list) \
            else variables_groundings
        # pesco le label delle variabili da quantificare
        aggregation_vars = set([var.free_variables[0] for var in variables_groundings])
        if mask_fn is not None and mask_vars is not None:
            # create and apply the mask
            # compute_mask creates the mask
            symbol_grounding, mask = compute_mask(symbol_grounding, mask_vars, mask_fn, aggregation_vars)
            # we apply the mask to the grounding of the predicate or term (forse solo predicato)
            # vedere se fare il prodotto element-wise qui
            #masked_symbol_grounding = torch.masked_select(symbol_grounding, mask)  # ritorna una sequenza dei valori del
            # predicato che soddisfanno la maschera
            # aggregate
            # dimensione della variabile su cui aggregare
            masked_symbol_grounding = torch.multiply(symbol_grounding, mask)
            print(masked_symbol_grounding)
            # metto dei NaN dove la maschera mette zero, il resto lascio invariato
            masked_symbol_grounding = torch.where(
                ~mask,
                np.nan,
                masked_symbol_grounding
            )
            print(masked_symbol_grounding)
            # TODO verificare che dove e' nan mi venga fatta l'aggregazione lo stesso
            aggregation_dims = [symbol_grounding.free_variables.index(var) for var in aggregation_vars]
            result = self.aggregation_operator(masked_symbol_grounding, aggregation_dims, **kwargs)
            print(result)
            # For some values in the tensor, the mask can result in aggregating with empty variables.
            #    e.g. forall X ( exists Y:condition(X,Y) ( p(X,Y) ) )
            #       For some values of X, there may be no Y satisfying the condition
            # The result of the aggregation operator in such case is often not defined (e.g. nan).
            # We replace the result with 0.0 if the semantics of the aggregator is exists,
            # or 1.0 if the semantics of the aggregator is forall.
            empty_quantifier = 1. if self.quantifier == "forall" else 0
            result = torch.where(
                torch.isnan(result),
                empty_quantifier,
                result
            )
        else:
            # aggregation_dim sono le dimensioni su cui fare l'aggregazione. queste dipendono dalle variabili su cui
            # fare aggregazione e l'operatore aggrega sugli assi di queste variabili
            aggregation_dims = [symbol_grounding.free_variables.index(var) for var in aggregation_vars]
            # queste sono le dimensioni su cui la media deve essere fatta
            result = self.aggregation_operator(symbol_grounding, dim=tuple(aggregation_dims))
        result.free_variables = [var for var in symbol_grounding.free_variables if var not in aggregation_vars]
        undiag(variables_groundings)
        return result


def compute_mask(symbol_grounding, mask_vars, mask_fn, aggregation_vars):
    """
    Qui il symbol grounding e' il grounding del predicato o termine. Mask_vars sono i groundings delle variabili su cui applicare la
    maschera. mask_fn e' la funzione di filtraggio della maschera. aggregation vars sono le label delle variabili su cui fare quantificazione (anche
    non guarded)
    :param symbol_grounding:
    :param mask_vars:
    :param mask_fn:
    :param aggregation_vars:
    :return:
    """
    # 1. cross symbol_grounding with groundings of variables that are in the mask but not yet in the formula
    mask_vars_not_in_symbol_grounding = [var for var in mask_vars
                                         if var.free_variables[0] not in symbol_grounding.free_variables]
    symbol_grounding = cross_grounding_values_of_symbols([symbol_grounding] + mask_vars_not_in_symbol_grounding)[0][0]
    print(symbol_grounding.shape)
    # 2. set the masked vars on the first axes
    vars_in_mask = [var.free_variables[0] for var in mask_vars]
    vars_in_mask_not_aggregated = [var for var in vars_in_mask if var not in aggregation_vars]
    vars_in_mask_aggregated = [var for var in vars_in_mask if var in aggregation_vars]
    vars_not_in_mask = [var for var in symbol_grounding.free_variables if var not in vars_in_mask]
    new_vars_order = vars_in_mask_not_aggregated + vars_in_mask_aggregated + vars_not_in_mask
    symbol_grounding = transpose_vars(symbol_grounding, new_vars_order)
    # 3. compute the boolean mask from the masked vars
    crossed_mask_vars, vars_order_in_mask, n_individuals_per_var = cross_grounding_values_of_symbols(mask_vars, flatten_dim0=True)
    mask = mask_fn(crossed_mask_vars)  # crea la maschera
    mask = torch.reshape(mask, tuple(n_individuals_per_var))  # la mette nella shape giusta
    # 4. shape it according to the var order in symbol_grounding
    mask.free_variables = vars_order_in_mask  # aggiunge le free variables alla mask
    mask = transpose_vars(mask, vars_in_mask_not_aggregated + vars_in_mask_aggregated)
    return symbol_grounding, mask


def transpose_vars(symbol_grounding, new_vars_order):
    perm = [symbol_grounding.free_variables.index(var) for var in new_vars_order]
    symbol_grounding = torch.permute(symbol_grounding, perm)
    symbol_grounding.free_variables = new_vars_order
    return symbol_grounding