import torch
from torch import nn
import numpy as np

# TODO ricordarsi di mettere il seed per le cose random
# TODO guardare le proposizioni nel tutorial
# TODO vedere se mettere controlli sulle shape di input ai predicati, se matchano con la dim del primo strato
# TODO implementare la variabile proposizionale
# TODO il fatto che cross_values restituisce una lista e' scomodo, vedere come fare perche' in certi casi risulta scomodo
# TODO rivedere il discorso active_doms, perche' non mi piace
import ltn


def constant(value, trainable=False):
    """Function that creates an LTN constant.

    An LTN constant denotes an individual grounded as a tensor in the Real field.
    The individual can be pre-defined (fixed data point) or learnable (embedding).

    Args:
        value: the value that becomes the grounding of the individual represented by the LTN constant. The value can be
        a tensor of any order.
        trainable: whether the LTN constant is trainable or not. If False, the PyTorch subgraph containing the constant
        will be excluded from the gradient computation. If True, the constant is initialized using the value parameter.
        Defaults to False.
    Returns:
        a `torch.FloatTensor` representing the LTN constant.
        A dynamic attribute `free_variables` is added to the tensor. In LTN, this attribute contains the labels of the
        free variables that appear on a constant, variable, formula, etc. `free_variables` is usually a list of labels
        that associates each variable to one of the axes of the grounding of a formula. Since in this case we have a
        constant, `free_variables` will be empty since a constant does not have free variables.
    """
    # we ensure that the tensor will be a float tensor and not a double tensor
    const = torch.tensor(value, requires_grad=trainable).float().to(ltn.device)
    const.free_variables = []
    return const


def variable(variable_name, individuals_seq):
    """Function that creates an LTN variable.

    An LTN variable denotes a sequence of individuals. It is grounded as a sequence of tensors (groundings of
    individuals) in the real field.
    Axis 0 is the batch dimension (it is associated with the number of individuals in the grounding of the variable).
    So, if `x` is an `ltn.Variable`, `x[0]` gives the first individual, `x[1]` gives the second individual,
    and so forth, i.e., the usual way.

    Args:
        variable_name: it is a string containing the name of the variable, for example 'x'.
        individuals_seq: it is a list of individuals that becomes the grounding the LTN variable. Notice that each
        individual in the sequence must have the same shape (i.e., must be of the same domain). Alternatively, it is
        possible to directly give a `torch.tensor`, which becomes the grounding of the variable.
    Returns:
        a `torch.FloatTensor` representing the LTN variable, where axis 0 is related with the number of individuals in
        the grounding of the variable. Like for the LTN constants, the dynamic attribute `free_variables` is added to
        the LTN variable. Moreover, a dynamic attribute called latent_variable is added to the output too. This attribute
        is used by LTN for performing diagonal quantification of the variables.
    """
    if variable_name.startswith("diag"):
        raise ValueError("Labels starting with diag are reserved.")
    if isinstance(individuals_seq, torch.Tensor):
        # we ensure that the tensor will be a float tensor and not a double tensor
        var = individuals_seq.float().to(ltn.device)
    else:
        # we ensure that the tensor will be a float tensor and not a double tensor
        var = torch.tensor(individuals_seq).float().to(ltn.device)

    if len(var.shape) == 1:
        # adds a dimension to transform the input in the correct shape to work with LTN
        # it transforms the input into a sequence of individuals in the case it is not a proper sequence
        # for example, [3, 1, 2] is transformed into [[3], [1], [2]]
        var = var.view(var.shape[0], 1)

    var.free_variables = [variable_name]
    var.latent_variable = variable_name

    return var

'''
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
'''


def get_n_individuals_of_var(grounding, var):
    """Returns the number of individuals of the variable var contained in the grounding given in input.
    Here, var is needed to specify the axis of the variable in the input grounding (tensor).
    """
    return grounding.size(grounding.free_variables.index(var))


def cross_grounding_values(input_groundings, flat_batch_dim=False):
    """
    This function creates the combination of all the possible values of the groundings given in input. These are
    the groundings of logical symbols or any expression built on them. These symbols can be ltn variables, constants,
    functions, predicates, or any expression built on those.

    It returns a list of tensors containing the combinations of values of the groundings given in input. Each one
    of these tensors is a component of the combination. If these tensors are concatenated along axis 1, the combinations
    are generated. The output list contains one tensor per input symbol.

    Moreover, it returns a list of variable labels and a list containing the number of individuals for each variable in
    the list of variable labels.
    The variable labels correspond to the variables contained in the groundings of the symbols that have been passed
    in input.

    Args:
        input_groundings: list of groundings of expressions of potentially different domains for which the combination of
        values have to be generated. These groundings are related to symbols that can be ltn variables, constants,
        functions, predicates, or any expression built on those.
        flat_batch_dim: if True, it removes the first dimension from the output tensors and flat it. For example, if one
        output tensor has size [3, 2, 2], if flatten_dim0 is set to True, its size becomes [6, 2]. In other words, it
        removes the batch dimensions.
    """
    vars_to_n_individuals = {}
    for grounding in input_groundings:
        for var in grounding.free_variables:
            vars_to_n_individuals[var] = get_n_individuals_of_var(grounding, var)
    vars = list(vars_to_n_individuals.keys())
    n_individuals_per_var = list(vars_to_n_individuals.values())
    crossed_groundings = []
    for grounding in input_groundings:
        vars_in_grounding = list(grounding.free_variables)
        vars_not_in_grounding = list(set(vars).difference(vars_in_grounding))
        for new_var in vars_not_in_grounding:
            new_idx = len(vars_in_grounding)
            grounding = torch.unsqueeze(grounding, dim=new_idx)
            grounding = torch.repeat_interleave(grounding, repeats=vars_to_n_individuals[new_var],
                                                       dim=new_idx)
            vars_in_grounding.append(new_var)
        perm = [vars_in_grounding.index(var) for var in vars] + list(range(len(vars_in_grounding),
                                                                        len(grounding.shape)))
        grounding = grounding.permute(perm)
        grounding.free_variables = vars
        if flat_batch_dim:
            #  this adds the batch dimension if there is not, for example for the constants
            shape_list = [-1] + list(grounding.shape[len(vars_in_grounding)::])
            grounding = torch.reshape(grounding, shape=tuple(shape_list))
        crossed_groundings.append(grounding)

    return crossed_groundings, vars, n_individuals_per_var


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
    """Predicate class for LTN.

    An LTN predicate is a mathematical function (either pre-defined or learnable) that maps
    from some n-ary domain of individuals to a real number in [0,1] (fuzzy) that can be interpreted as a truth value.
    Examples of predicates can be similarity measures, classifiers, etc.

    Predicates can be defined using any operations in PyTorch. They can be linear functions, Deep Neural Networks,
    and so forth.

    An LTN predicate implements a `nn.Module` instance that can "broadcast" ltn terms as follows:
    1. Evaluating a predicate with one variable of n individuals yields n output values,
    where the i-th output value corresponds to the predicate calculated with the i-th individual.
    2. Evaluating a predicate with k variables (x1,...,xk) with respectively n1,...,nk
    individuals each, yields a result with n1*...*nk values. The result is organized in a tensor
    where the first k dimensions can be indexed to retrieve the outcome(s) that correspond to each variable.
    The attribute `free_variables` tells which axis corresponds to which variable in the tensor output by
    the predicate (using the name of the variable).

    Args:
        model: PyTorch model that becomes the grounding of the predicate;
        layers_size: if a model is not given, it is possible to give `layers_size` and a fully-connected MLP with
        layers with dimensions specified by `layers_size` will be used as model;
        lambda_func: if a model is not given and layers_size is not given, it is possible to give a lambda function.
        In this case the lambda function will be used to define a non-trainable model for the LTN predicate.
    Attributes:
        model: the grounding of the LTN predicate. The grounding of a predicate is a non-trainable model implemented
        using a lambda function or a learnable model. When the groundings of the inputs are given to the predicate model,
        the model returns a real value in [0, 1] for each combination of the values of the groundings given in input.
        Then, at the output is attached a dynamic attribute called `free_variables`, which contains the list of free
        variables contained in the output tensor;
        model_type: it is a string containing the type of the model ('model', 'mlp', 'lambda'). This attribute is used
        to manage the inputs of the different types of models differently.
    """
    def __init__(self, model=None, layers_size=None, lambda_func=None):
        """
        Initializes the LTN predicate in three different ways:
            1. if `model` is not None, it initializes the predicate with the given PyTorch model;
            2. if `model` is None and `layers_size` is not None, it creates a MLP model with linear layers with
            dimensions specified by `layers_size` and uses that model as the LTN predicate;
            3. if `model` is None and `layers_size` is None, it uses the `lambda_func` as a lambda function to represent
            the LTN predicate. Note that in this case the LTN predicate is not learnable. So, the lambda function has
            to be used only for simple predicates.
        Note that if more than one of these parameters is not None, the first parameter that is not None in the order is
        preferred.
        """
        super(Predicate, self).__init__()
        if model is None and layers_size is None and lambda_func is None:
            raise ValueError("A model, or dimension of layers for constructing an MLP model, or a lambda function to "
                             "be used as a non-trainable model should be given in input.")
        if model is not None:
            assert isinstance(model, nn.Module), "The given model is not a PyTorch model."
            self.model = model
            self.model_type = 'model'
        elif layers_size is not None:
            assert isinstance(layers_size, tuple), "layers_size must be a tuple of integers."
            self.model = self.mlp(layers_size)
            self.model_type = 'mlp'
        else:
            self.model = self.lambda_operation(lambda_func)
            self.model_type = 'lambda'

    def forward(self, inputs, *args, **kwargs):
        """Encapsulates the "self.model.forward()" to handle the ltn-broadcasting.

        Args:
            inputs: list of tensors that are ltn terms (ltn variable, ltn constant or
                    output of a ltn function) for which the predicate has to be computed.
        Returns:
            a `torch.Tensor` of truth values representing the result of the predicate, with dimensions s.t.
            each variable corresponds to one axis.
        """
        assert isinstance(inputs, (list, torch.Tensor)), "The inputs parameter should be a list of tensors or a tensor."
        if isinstance(inputs, list):
            inputs, vars, n_individuals_per_var = cross_grounding_values(inputs, flat_batch_dim=True)
        else:
            # this is the case in which the predicate takes as input only one object (constant, variable, etc.)
            inputs, vars, n_individuals_per_var = cross_grounding_values([inputs], flat_batch_dim=True)

        outputs = None  # outputs initialization
        if self.model_type == 'model' or self.model_type == 'lambda':
            # the management of the input is left to the model or the lambda function
            outputs = self.model(inputs, *args, **kwargs)
        if self.model_type == 'mlp':
            # if the model is an mlp directly instantiated by LTN, it is necessary to flat the input
            flat_inputs = [torch.flatten(x, start_dim=1) for x in inputs]  # if len(x.shape) > 1 else x
            inputs = torch.cat(flat_inputs, dim=1) if len(flat_inputs) > 1 else flat_inputs[0]
            outputs = self.model(inputs, *args, **kwargs)

        outputs = torch.reshape(outputs, tuple(n_individuals_per_var))
        outputs = outputs.float()

        outputs.free_variables = vars
        return outputs

    @staticmethod
    def lambda_operation(lambda_function):
        """It construct a simple and non-trainable mathematical operation using the lambda function given in input.
        It is appropriate for small non-trainable mathematical operations that return a value in [0,1]."""
        model = LambdaModel(lambda_function)
        return model

    @staticmethod
    def mlp(layer_dims=(16, 16, 1)):
        """
        It constructs a fully-connected MLP with linear layers with the dimensions given in input. It uses an ELU
        activation on the hidden layers and a sigmoid on the final layer.
        :param layer_dims: dimensions of the layers of the MLP.
        :return: an MLP with architecture defined by `layers_dims` parameter. The first dimension is the dimension of
        the input layer, the last dimension is the dimension of the output layer.
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
        model: PyTorch model that becomes the grounding of the function;
        layers_size: if a model is not given, it is possible to give `layers_size` and a fully-connected MLP with
        layers with dimensions specified by `layers_size` will be used as model;
        lambda_func: if a model is not given and layers_size is not given, it is possible to give a lambda function.
        In this case the lambda function will be used to define a non-trainable model for the LTN function.
    Attributes:
        model: the grounding of the LTN function. The grounding of a function is a non-trainable model implemented
        using a lambda function or a learnable model. When the groundings of the inputs are given to the function model,
        the model returns a tensor in the real filed for each combination of the values of the groundings given in input.
        Then, at the output is attached a dynamic attribute called `free_variables`, which contains the list of free
        variables contained in the output tensor;
        model_type: it is a string containing the type of the model ('model', 'mlp', 'lambda'). This attribute is used
        to manage the inputs of the different types of models differently.
    """

    def __init__(self, model=None, layers_size=None, lambda_func=None):
        """
        Initializes the LTN function in three different ways:
            1. if `model` is not None, it initializes the function with the given PyTorch model;
            2. if `model` is None and `layers_size` is not None, it creates a MLP model with linear layers with
            dimensions specified by `layers_size` and uses that model as the LTN function;
            3. if `model` is None and `layers_size` is None, it uses the `lambda_func` as a lambda function to represent
            the LTN function. Note that in this case the LTN function is not learnable. So, the lambda function has
            to be used only for simple functions.
        Note that if more than one of these parameters is not None, the first parameter that is not None in the order is
        preferred.
        """
        super(Function, self).__init__()
        if model is None and layers_size is None and lambda_func is None:
            raise ValueError("A model, or dimension of layers for constructing an MLP model, or a lambda function to "
                             "be used as a non-trainable model should be given in input.")
        if model is not None:
            assert isinstance(model, nn.Module), "The given model is not a PyTorch model."
            self.model = model
            self.model_type = 'model'
        elif layers_size is not None:
            assert isinstance(layers_size, tuple), "layers_size must be a tuple of integers."
            self.model = self.mlp(layers_size)
            self.model_type = 'mlp'
        else:
            self.model = self.lambda_operation(lambda_func)
            self.model_type = 'lambda'

    def forward(self, inputs, *args, **kwargs):
        """Encapsulates the "self.model.forward()" to handle the ltn-broadcasting.

        Args:
            inputs: list of tensors that are ltn terms (ltn variable, ltn constant or
                    output of a ltn function) for which the predicate has to be computed.
        Returns:
            a `torch.Tensor` of output values (each output is a tensor too), with dimensions s.t. each variable
            corresponds to one axis.
        """
        assert isinstance(inputs, (list, torch.Tensor)), "The inputs parameter should be a list of tensors or a tensor."

        if isinstance(inputs, list):
            inputs, vars, n_individuals_per_var = cross_grounding_values(inputs, flat_batch_dim=True)
        else:
            # this is the case in which the function takes as input only one object (constant, variable, etc.)
            inputs, vars, n_individuals_per_var = cross_grounding_values([inputs], flat_batch_dim=True)

        outputs = None
        if self.model_type == 'model' or self.model_type == 'lambda':
            outputs = self.model(inputs, *args, **kwargs)

        if self.model_type == 'mlp':
            flat_inputs = [torch.flatten(x, start_dim=1) for x in inputs]
            inputs = torch.cat(flat_inputs, dim=1) if len(flat_inputs) > 1 else flat_inputs[0]
            outputs = self.model(inputs, *args, **kwargs)

        outputs = torch.reshape(outputs, tuple(n_individuals_per_var + list(outputs.shape[1::])))

        outputs = outputs.float()
        outputs.free_variables = vars
        return outputs

    @staticmethod
    def lambda_operation(lambda_function):
        """It construct a simple and non-trainable mathematical operation using the lambda function given in input.
        It is appropriate for small non-trainable mathematical operations."""
        model = LambdaModel(lambda_function)
        return model

    @staticmethod
    def mlp(layer_dims=(16, 16, 1)):
        """
        It constructs a fully-connected MLP with linear layers with the dimensions given in input. It uses an ELU
        activation on the hidden layers and a linear activation on the final layer.
        :param layer_dims: dimensions of the layers of the MLP.
        :return: an MLP with architecture defined by `layers_dims` parameter. The first dimension is the dimension of
        the input layer, the last dimension is the dimension of the output layer.
        """
        layers = []
        for i in range(1, len(layer_dims)):
            layers.append(nn.Linear(layer_dims[i - 1], layer_dims[i]))
            if i != (len(layer_dims) - 1):
                layers.append(nn.ELU())
        model = nn.Sequential(*layers)
        return model


def diag(variables_groundings):
    """Sets the given LTN variables for diagonal quantification (no broadcasting between these variables).

    Given 2 (or more) LTN variables, there are scenarios where one wants to express statements about
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
    LTN computes only the "zipped" results when diagonal quantification is performed.

    Args:
        variables_groundings: the grounding of the LTN variables for which the diagonal quantification has to be
    performed
    Returns:
        the groundings of the variables given in input, where the attribute `free_variables` has been changed to allow
        the use of the diagonal quantification.
    """
    # check if more than one variable has been given to the function
    assert len(variables_groundings) > 1, "It is not possible to perform diagonal quantification on a single variable." \
                                          " At least two variables have to be given."
    # check if variables have the same number of individuals
    n_individuals = [var.shape[0] for var in variables_groundings]
    assert len(set(n_individuals)) == 1, "The given variables have a different number of individuals between each other." \
                                         " It is not possible to perform diagonal quantification between variables that" \
                                         " have a different number of individuals."
    diag_vars_label = "diag_" + "_".join([var.latent_variable for var in variables_groundings])
    for var in variables_groundings:
        var.free_variables = [diag_vars_label]
    return variables_groundings


def undiag(variables_groundings):
    """Resets the usual broadcasting strategy for the given LTN variables.

    In practice, `ltn.diag` is designed to be used with quantifiers.
    Every quantifier automatically calls `ltn.undiag` after the aggregation is performed,
    so that the variables continue to keep their normal behavior outside of the formula.
    Therefore, it is recommended to use `ltn.diag` only in quantified formulas as follows:
        ```
        Forall(ltn.diag(x,l), C([x,l]))
        ```

    Args:
        variables_groundings: the grounding of the variables for which the diagonal setting has to be removed.
    Returns:
        the same variable groundings given in input with the attribute `free_variables` changed in such a way that
        the diagonal setting has been removed.
    """
    for var in variables_groundings:
        var.free_variables = [var.latent_variable]
    return variables_groundings


class WrapperConnective:
    """Class to wrap unary/binary connective operators to use them within LTN formulas.

    LTN supports various logical connectives. They are grounded using fuzzy semantics.
    The implementation of some common fuzzy logic operators using PyTorch primitives is in `ltn.fuzzy_ops`.
    The wrapper `ltn.WrapperConnective` allows to use these fuzzy operators with LTN formulas.
    It takes care of combining sub-formulas that have different variables appearing in them
    (the sub-formulas may have different dimensions that need to be "broadcasted").
    Attributes:
        connective_operator: the original unary/binary fuzzy connective operator (without broadcasting).
    """

    def __init__(self, connective_operator):
        self.connective_operator = connective_operator

    def __call__(self, *input_groundings, **kwargs):
        """
        It applies the selected fuzzy connective operator to the groundings given in input. To do so, it firstly
        broadcast the input groundings to make them compatible to apply the operator.
        :param input_groundings: the groundings of expressions to which the fuzzy connective operator has to be applied.
        :return: the grounding that is the result of the application of the fuzzy connective operator to the input
        groundings.
        """
        # TODO capire a cosa serviva l'eccezione qui
        input_groundings, vars, _ = cross_grounding_values(input_groundings)
        output = self.connective_operator(*input_groundings)
        output.free_variables = vars
        return output


class WrapperQuantifier:
    """Class to wrap quantification operators to use them within LTN formulas.

    LTN supports universal and existential quantification. They are grounded using fuzzy aggregation operators.
    The implementation of some common aggregators using PyTorch primitives is in `ltn.fuzzy_ops`.
    The wrapper allows to use the quantifiers with LTN formulas.
    It takes care of selecting the tensor dimensions to aggregate, given some variables in arguments.
    Additionally, boolean conditions (`mask_fn`,`mask_vars`) can be used for guarded quantification.
    Attributes:
        aggregation_operator: the fuzzy aggregation operator to perform the desired quantification;
        quantification_type: it is a string indicating the quantification that has to be performed (exists or forall).
    """

    def __init__(self, aggregation_operator, quantification_type):
        self.aggregation_operator = aggregation_operator
        if quantification_type not in ["forall", "exists"]:
            raise ValueError("The keyword for the quantification operator should be \"forall\" or \"exists\".")
        self.quantification_type = quantification_type

    def __call__(self, variables_groundings, formula_grounding, mask_vars=None, mask_fn=None, **kwargs):
        """
        It applies the desired quantification at the input formula (`formula_grounding`) based on the selected
        variables (`variables_groundings`). It is also possible to perform a guarded quantification. In that case,
        `mask_vars` and `mask_fn` have to be set properly. If 'mask_vars' and 'mask_fn' are left `None` it means
        that no guarded quantification has to be performed.

        Args:
            variables_groundings: groundings of the variables on which the quantification has to be performed;
            formula_grounding: grounding of the formula that has to be quantified;
            mask_vars: grounding of the variables that are included in the guarded quantification condition;
            mask_fn: function which implements the guarded quantification condition. The condition is based on the
            variables contained in `mask_vars`.
        """
        # check if the quantification has to be performed on one or more variables
        variables_groundings = [variables_groundings] if not isinstance(variables_groundings, list) \
            else variables_groundings
        # aggregation_vars contains the labels of the variables on which the quantification has to be performed
        aggregation_vars = set([var.free_variables[0] for var in variables_groundings])
        # check if guarded quantification has to be performed
        if mask_fn is not None and mask_vars is not None:  # in this case, the guarded quantification has to be performed
            # create the mask by using compute_mask() function
            formula_grounding, mask = compute_mask(formula_grounding, mask_vars, mask_fn, aggregation_vars)
            # we apply the mask to the grounding of the formula
            # the idea is to put NaN values where the mask is zero, while the rest of the grounding is kept untouched
            if formula_grounding.shape != mask.shape:
                # I have to rearrange the size of the mask if it has a different size respect to the formula_grounding
                n_new_dims = len(formula_grounding.shape) - len(mask.shape)
                mask = mask.reshape(mask.shape + (1,) * n_new_dims)
                mask = mask.expand(formula_grounding.shape)

            masked_formula_grounding = torch.where(
                ~mask,
                np.nan,
                formula_grounding.double()
            )
            # we perform the desired quantification after the mask has been applied
            aggregation_dims = [formula_grounding.free_variables.index(var) for var in aggregation_vars]
            output = self.aggregation_operator(masked_formula_grounding, aggregation_dims, **kwargs)
            # For some values in the tensor, the mask can result in aggregating with empty variables.
            #    e.g. forall X ( exists Y:condition(X,Y) ( p(X,Y) ) )
            #       For some values of X, there may be no Y satisfying the condition
            # The result of the aggregation operator in such case is often not defined (e.g. nan).
            # We replace the result with 0.0 if the semantics of the aggregator is exists,
            # or 1.0 if the semantics of the aggregator is forall.
            empty_quantifier = 1. if self.quantification_type == "forall" else 0.
            output = torch.where(
                torch.isnan(output),
                empty_quantifier,
                output
            )
        else:  # in this case, the guarded quantification has not to be performed
            # aggregation_dims are the dimensions on which the aggregation has to be performed
            # the aggregator aggregates only on the axes given by aggregations_dims
            aggregation_dims = [formula_grounding.free_variables.index(var) for var in aggregation_vars]
            output = self.aggregation_operator(formula_grounding, dim=tuple(aggregation_dims), **kwargs)
        # update the free variables on the output groundings based on which variables have been aggregated
        output.free_variables = [var for var in formula_grounding.free_variables if var not in aggregation_vars]
        undiag(variables_groundings)
        return output


def compute_mask(formula_grounding, mask_vars, mask_fn, aggregation_vars):
    """
    It computes the mask for performing the guarded quantification on the grounding of the formula given in input.
    :param formula_grounding: grounding of the formula that has to be quantified;
    :param mask_vars: grounding of the variables that are included in the guarded quantification condition;
    :param mask_fn: function which implements the guarded quantification condition;
    :param aggregation_vars: list of labels of the variables on which the quantification has to be performed.
    :return: a tuple where the first element is the grounding of the input formula transposed in such a way that the
    guarded variables are in the first dimensions, while the second element is the mask that has to be applied over
    the grounding of the formula in order to perform the guarded quantification.
    """
    # 1. cross formula_grounding with groundings of variables that are in the mask but not yet in the formula
    mask_vars_not_in_formula_grounding = [var for var in mask_vars
                                         if var.free_variables[0] not in formula_grounding.free_variables]
    formula_grounding = cross_grounding_values([formula_grounding] + mask_vars_not_in_formula_grounding)[0][0]
    # 2. set the masked (guarded) vars on the first axes
    vars_in_mask = [var.free_variables[0] for var in mask_vars]
    vars_in_mask_not_aggregated = [var for var in vars_in_mask if var not in aggregation_vars]
    vars_in_mask_aggregated = [var for var in vars_in_mask if var in aggregation_vars]
    vars_not_in_mask = [var for var in formula_grounding.free_variables if var not in vars_in_mask]
    new_vars_order = vars_in_mask_not_aggregated + vars_in_mask_aggregated + vars_not_in_mask
    formula_grounding = transpose_vars(formula_grounding, new_vars_order)
    # 3. compute the boolean mask from the masked vars
    crossed_mask_vars, vars_order_in_mask, n_individuals_per_var = cross_grounding_values(mask_vars,
                                                                                          flat_batch_dim=True)
    mask = mask_fn(crossed_mask_vars)  # creates the mask
    mask = torch.reshape(mask, tuple(n_individuals_per_var))  # reshape the mask in such a way that it is compatible with formula_grounding
    # 4. shape it according to the var order in formula_grounding
    mask.free_variables = vars_order_in_mask  # adds the free variables to the mask
    mask = transpose_vars(mask, vars_in_mask_not_aggregated + vars_in_mask_aggregated)

    return formula_grounding, mask


def transpose_vars(input_grounding, new_vars_order):
    """
    It transposes the input grounding using the order of variables given in `new_vars_order`.
    :param input_grounding: the grounding that has to be transposed.
    :param new_vars_order: the order of variables to transpose the input grounding.
    :return: the input grounding transposed according to the order in `new_vars_order`.
    """
    perm = [input_grounding.free_variables.index(var) for var in new_vars_order]
    input_grounding = torch.permute(input_grounding, perm)
    input_grounding.free_variables = new_vars_order
    return input_grounding