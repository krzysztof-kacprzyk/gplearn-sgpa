import sympy as sp
import numpy as np
from itertools import product

class Perturbation():
    def __init__(self, symbol_top, symbol_bottom, independent_variable):
        self.symbol_top = symbol_top
        self.symbol_bottom = symbol_bottom
        self.independent_variable = independent_variable
        self._validate_perturbation()

    # Function to validate the input to the constructor
    def _validate_perturbation(self):
        if self.symbol_top not in ['+','*']:
            raise ValueError('symbol_top must be either "+" or "*"')
        if self.symbol_bottom not in ['+','*']:
            raise ValueError('symbol_bottom must be either "+" or "*"')
        # independent_variable must be a sympy symbol
        if not isinstance(self.independent_variable, sp.Symbol):
            raise ValueError('independent_variable must be a sympy symbol')
        
    def __call__(self, f):
        # Check if f is a sympy expression
        if not isinstance(f, sp.Expr):
            raise ValueError('f must be a sympy expression')

        l = sp.symbols('l')
        
        # Create the perturbation
        if self.symbol_bottom == '+':
            perturbed_f = f.subs(self.independent_variable, self.independent_variable + l)
        else:
            perturbed_f = f.subs(self.independent_variable, self.independent_variable * l)
        
        # Calcuate the change
        if self.symbol_top == '+':
            change = perturbed_f - f
        else:
            change = perturbed_f / f
        
        return change
    
def is_empirically_active(f, variable, n_samples=20, n_measurements=10):

    # Random numpy generator
    gen = np.random.default_rng(seed=0)

    all_variables = f.free_symbols
    if variable not in all_variables:
        return False
    
    all_variables = list(all_variables - {variable})

    # Sample 3 sets of values for the variables
    values = [gen.uniform(-10, 10, len(all_variables)).reshape((1,-1)) for _ in range(n_samples)]

    values = np.concatenate([np.repeat(v,n_measurements, axis=0) for v in values], axis=0)

    # Sample values ofr variable
    variable_values = gen.uniform(-1, 1, n_samples*n_measurements).reshape((-1,1))

    # Create the data matrix
    data = np.concatenate([values, variable_values], axis=1)

    # Lambdify the function
    f_lambdified = sp.lambdify(all_variables + [variable], f, 'numpy')


    # Evaluate the function
    f_values = f_lambdified(*data.T).reshape((-1,1))

    for i in range(n_samples):
        if not np.all(np.isclose(f_values[i*n_measurements:(i+1)*n_measurements],f_values[i*n_measurements])):
            return True
    return False

    
def empirically_active_variables(f, independent_variables=None, constants=None):
    """
    Find the active variables in a sympy expression f with respect to a list of independent variables.

    Parameters
    ----------
    f : sympy.Expr
        A sympy expression.
    independent_variables : a list of sympy.Symbol
        A list of sympy symbols representing the independent variables.
    constants : a list of sympy.Symbol
        A list of sympy symbols representing the constants.
    
    Returns
    -------
    A set of sympy symbols representing the active variables.
    """

    # Check if f is a sympy expression
    if not isinstance(f, sp.Expr):
        raise ValueError('f must be a sympy expression')
    # Check if independent_variables is a list of sympy symbols
    if (independent_variables is not None) and not all(isinstance(var, sp.Symbol) for var in independent_variables):
        raise ValueError('independent_variables must be a list of sympy symbols or None')
    # Check if constants is a list of sympy symbols
    if (constants is not None) and not all(isinstance(var, sp.Symbol) for var in constants):
        raise ValueError('constants must be a list of sympy symbols or None')

    l = sp.symbols('l')
    if constants is None:
        constants = set()
    else:
        constants = set(constants)
    if independent_variables is None:
        return f.free_symbols - {l} - constants
    else:
        independent_variables = set(independent_variables)

    active_variables = set()
    for var in independent_variables:
        if is_empirically_active(f, var):
            active_variables.add(var)
    return active_variables

    



def active_variables(f, independent_variables=None, constants=None):
    """
    Find the active variables in a sympy expression f with respect to a list of independent variables.

    Parameters
    ----------
    f : sympy.Expr
        A sympy expression.
    independent_variables : a list of sympy.Symbol
        A list of sympy symbols representing the independent variables.
    constants : a list of sympy.Symbol
        A list of sympy symbols representing the constants.
    
    Returns
    -------
    A set of sympy symbols representing the active variables.
    """

    # Check if f is a sympy expression
    if not isinstance(f, sp.Expr):
        raise ValueError('f must be a sympy expression')
    # Check if independent_variables is a list of sympy symbols
    if (independent_variables is not None) and not all(isinstance(var, sp.Symbol) for var in independent_variables):
        raise ValueError('independent_variables must be a list of sympy symbols or None')
    # Check if constants is a list of sympy symbols
    if (constants is not None) and not all(isinstance(var, sp.Symbol) for var in constants):
        raise ValueError('constants must be a list of sympy symbols or None')

    l = sp.symbols('l')
    if constants is None:
        constants = set()
    else:
        constants = set(constants)
    if independent_variables is None:
        return f.simplify().free_symbols - {l} - constants
    else:
        independent_variables = set(independent_variables)
        return (f.simplify().free_symbols - {l}).intersection(independent_variables) - constants

def degree(f, independent_variables=None, constants=None, empirical=False):
    """
    Calculate the degree of a sympy expression f with respect to a list of independent variables.

    Parameters
    ----------
    f : sympy.Expr
        A sympy expression.
    independent_variables : a list of sympy.Symbol
        A list of sympy symbols representing the independent variables.
    constants : a list of sympy.Symbol
        A list of sympy symbols representing the constants.

    Returns
    -------
    An integer representing the degree of the expression.
    """

    # Check if f is a sympy expression
    if not isinstance(f, sp.Expr):
        raise ValueError('f must be a sympy expression')
    # Check if independent_variables is a list of sympy symbols
    if (independent_variables is not None) and not all(isinstance(var, sp.Symbol) for var in independent_variables):
        raise ValueError('independent_variables must be a list of sympy symbols or None')
    # Check if constants is a list of sympy symbols
    if (constants is not None) and not all(isinstance(var, sp.Symbol) for var in constants):
        raise ValueError('constants must be a list of sympy symbols or None')

    if empirical:
        return len(empirically_active_variables(f, independent_variables, constants))
    else:
        return len(active_variables(f, independent_variables, constants))

def complexity_1(f, independent_variables=None, operator_strategy='min_degree', reduce_strategy=None, empirical=False):
    """Calculate the complexity of a sympy expression f with respect to a list of independent variables.
    
    Parameters
    ----------
    f : sympy.Expr
        A sympy expression.
    independent_variables : list of sympy.Symbol
        A list of sympy symbols representing the independent variables.
    operator_strategy : str or callable, default='min_degree'
        A string from ['min_degree'] or a function that takes a dictionary of interaction functions {(+,+): f1, (+,*): f2, ...} 
        and returns a tuple of the form (operator_top, operator_bottom) that corresponds to the most natural perturbation operator. 
        The default 'min_degree' chooses the opertator with the interaction function with the smallest number of active variables.
    reduce_strategy : str or callable, default=None
        The strategy to use to reduce the complexity into a single number.
        A string from ['max', 'average'] or a function that takes a list of integers and returns a single number.

    Returns
    -------
    a single number or a dictionary of the form {independent_variable: degree} if reduce_strategy is None.
    """
    default_operator_strategies = {
        'min_degree': lambda interaction_functions: min(interaction_functions, key=lambda x: degree(interaction_functions[x], empirical=empirical))
    }
    default_reduce_strategies = {
        'max': max,
        'mean': np.mean
    }

    # Check if f is a sympy expression
    if not isinstance(f, sp.Expr):
        raise ValueError('f must be a sympy expression')
    # Check if independent_variables is a list of sympy symbols or None
    if independent_variables is not None and not all(isinstance(var, sp.Symbol) for var in independent_variables):
        raise ValueError('independent_variables must be a list of sympy symbols or None')
    # Check if operator_strategy is a string or a callable
    if not isinstance(operator_strategy, str) and not callable(operator_strategy):
        raise ValueError('operator_strategy must be a string or a callable')
    # If operator_strategy is a string, check if it is a valid string
    if isinstance(operator_strategy, str) and operator_strategy not in default_operator_strategies.keys():
        raise ValueError('operator_strategy must be one of {}'.format(default_operator_strategies.keys()))
    # Check if reduce_strategy is a string or a callable or None
    if reduce_strategy is not None and not isinstance(reduce_strategy, str) and not callable(reduce_strategy):
        raise ValueError('reduce_strategy must be a string, a callable or None')
    # If reduce_strategy is a string, check if it is a valid string
    if isinstance(reduce_strategy, str) and reduce_strategy not in default_reduce_strategies.keys():
        raise ValueError('reduce_strategy must be one of {}'.format(default_reduce_strategies.keys()))
    
    if independent_variables is None:
        independent_variables = list(f.free_symbols)

    constants = list(f.free_symbols - set(independent_variables))

    if isinstance(operator_strategy, str):
        operator_strategy = default_operator_strategies[operator_strategy]
    operators = list(product(['+', '*'], repeat=2))
    l = sp.symbols('l')
    degrees = {}
    for var in independent_variables:
        interaction_functions = {}
        for op in operators:
            interaction_function = Perturbation(op[0], op[1], var)(f)
            interaction_functions[op] = interaction_function
        chosen_operator = operator_strategy(interaction_functions)
        degrees[var] = degree(interaction_functions[chosen_operator], independent_variables=independent_variables, constants=constants, empirical=empirical)
    
    if reduce_strategy is not None:
        if isinstance(reduce_strategy, str):
            reduce_strategy = default_reduce_strategies[reduce_strategy]
        return reduce_strategy(list(degrees.values()))
    else:
        return degrees


class Node:

    def __init__(self, symbol, value, independent_variables=None, constants=None, id="0"):
        self.children = []
        self.symbol = symbol
        self.value = value
        self.independent_variables = independent_variables
        self.constants = constants
        self.expression = value
        self.symbol_counter = 0
        self.id_counter = 0
        self.id = id

    def is_symbol_terminal(self, symbol):
        return (symbol in self.independent_variables + self.constants) or symbol.is_Number
    
    def add_child(self, value):
        if self.is_symbol_terminal(value):
            child_symbol = value
        else:
            child_symbol = self.get_new_child_symbol(self.symbol_counter)
            self.symbol_counter += 1
        child = Node(child_symbol, value, self.independent_variables, self.constants, id=f"{self.id}{self.id_counter}")
        self.id_counter += 1
        self.children.append(child)
        return child
    
    def get_new_child_symbol(self, index):
        return sp.Symbol(f"{self.symbol.name}{index}")
    
    def set_value(self, value):
        self.value = value

    def set_expression(self):
        if self.value.func == sp.Symbol or self.value.is_Number:
            self.expression = self.value
        else:
            self.expression = self.value.func(*[child.symbol for child in self.children])
    
    def active_variables(self, empirical=False):
        if empirical:
            return empirically_active_variables(self.value, self.independent_variables, self.constants)
        else:
            return active_variables(self.value, self.independent_variables, self.constants)

    def is_terminal(self):
        return self.is_symbol_terminal(self.value)


class ComputationTree:

    def __init__(self, f, independent_variables, constants=None):
        self.f = f
        self.independent_variables = independent_variables
        if constants is None:
            constants = list(f.free_symbols - set(independent_variables))
        self.constants = constants
        self.nodes = {}
        self.root = Node(sp.Symbol('f'), f, independent_variables, constants)

        def process_node(node):
            if node.value.func == sp.Symbol or node.value.is_Number:
                return
            n_children = len(node.value.args)
            for i in range(n_children):
                child = node.add_child(node.value.args[i])
            node.set_expression()
            for child in node.children:
                process_node(child)

        process_node(self.root)

    
    def __repr__(self):
        self.result = ""
        def dfs_helper(node):
            if not node.is_terminal():
                self.result += f"{node.symbol}={node.expression}\n"
            for child in node.children:
                dfs_helper(child)
        dfs_helper(self.root)
        return self.result


def complexity_2(f, independent_variables=None, reduce_strategy=None, empirical=False):
    """
    Calculate the complexity of a sympy expression f with respect to a list of independent variables.

    Parameters
    ----------
    f : sympy.Expr
        A sympy expression.
    independent_variables : list of sympy.Symbol
        A list of sympy symbols representing the independent variables.
    reduce_strategy : str or callable, default=None
        The strategy to use to reduce the complexity into a single number.
        A string from ['max_all'] or a function that takes a dictionary of lists of integers and returns a single number.\

    Returns
    -------
    A dictionary of the form {independent_variable: [degree_1, degree_2, ...]}.
    """
    reduce_strategies = {
        'max_all': lambda x: max([max(i) for i in x.values()])
    }
    # Check if f is a sympy expression
    if not isinstance(f, sp.Expr):
        raise ValueError('f must be a sympy expression')
    # Check if independent_variables is a list of sympy symbols
    if independent_variables is not None and not all(isinstance(var, sp.Symbol) for var in independent_variables):
        raise ValueError('independent_variables must be a list of sympy symbols or None')
    # Check if reduce_strategy is a string or a callable or None
    if reduce_strategy is not None and not isinstance(reduce_strategy, str) and not callable(reduce_strategy):
        raise ValueError('reduce_strategy must be a string, a callable or None')
    # If reduce_strategy is a string, check if it is a valid string
    if isinstance(reduce_strategy, str) and reduce_strategy not in reduce_strategies.keys():
        raise ValueError('reduce_strategy must be one of {}'.format(reduce_strategies.keys()))
    
    if independent_variables is None:
        independent_variables = list(f.free_symbols)

    constants = list(f.free_symbols - set(independent_variables))
    l = sp.symbols('l')

    tree = ComputationTree(f, independent_variables, constants)
    result = {}
    for var in independent_variables:
        current_node = tree.root
        variable_queue = [current_node] 
        while True:
            if current_node.is_terminal():
                break
            intermediate_vars = current_node.children
            intermediate_vars_bound = []
            for v in intermediate_vars:
                if var in v.active_variables(empirical=empirical):
                    intermediate_vars_bound.append(v)
            if len(intermediate_vars_bound) == 1:
                main_var = intermediate_vars_bound[0]
                variable_queue.append(main_var)
                current_node = main_var
                continue
            else:
                variable_queue.append(var)
                break
        
        combinations = list(product(['+', '*'], repeat=len(variable_queue)))
        operators_degrees = {}
        for comb in combinations:
            barcode = []
            for i in range(len(variable_queue)-1):
                if (i == len(variable_queue)-2):
                    int_f = Perturbation(comb[i], comb[i+1], var)(variable_queue[i].value)
                else:
                    int_f = Perturbation(comb[i], comb[i+1], variable_queue[i+1].symbol)(variable_queue[i].expression)
                barcode.append(degree(int_f,constants=constants,empirical=empirical))
            # Combine consecutive 0s into a single 0
            j = 0
            while j < len(barcode)-1:
                if barcode[j] == 0 and barcode[j+1] == 0:
                    barcode.pop(j)
                else:
                    j += 1
            operators_degrees[comb] = barcode
        # Find the operator with the smallest maximum degree
        smallest_max_degree = min([0 if len(i)==0 else max(i) for i in operators_degrees.values()])
        # Among the operators with the smallest maximum degree, find the one with the shortest barcode
        best_comb = min([comb for comb in operators_degrees if (max(operators_degrees[comb]) if len(operators_degrees[comb])>0 else 0) == smallest_max_degree], key=lambda x: len(operators_degrees[x]))
        result[var] = operators_degrees[best_comb]

    if reduce_strategy is not None:
        if isinstance(reduce_strategy, str):
            reduce_strategy = reduce_strategies[reduce_strategy]
        return reduce_strategy(result)
    else:
        return result
    
def complexity_2_greedy(f, independent_variables=None, reduce_strategy=None, empirical=False):
    """
    Calculate the complexity of a sympy expression f with respect to a list of independent variables.

    Parameters
    ----------
    f : sympy.Expr
        A sympy expression.
    independent_variables : list of sympy.Symbol
        A list of sympy symbols representing the independent variables.
    reduce_strategy : str or callable, default=None
        The strategy to use to reduce the complexity into a single number.
        A string from ['max_all'] or a function that takes a dictionary of lists of integers and returns a single number.\

    Returns
    -------
    A dictionary of the form {independent_variable: [degree_1, degree_2, ...]}.
    """
    reduce_strategies = {
        'max_all': lambda x: max([max(i) for i in x.values()])
    }
    # Check if f is a sympy expression
    if not isinstance(f, sp.Expr):
        raise ValueError('f must be a sympy expression')
    # Check if independent_variables is a list of sympy symbols
    if independent_variables is not None and not all(isinstance(var, sp.Symbol) for var in independent_variables):
        raise ValueError('independent_variables must be a list of sympy symbols or None')
    # Check if reduce_strategy is a string or a callable or None
    if reduce_strategy is not None and not isinstance(reduce_strategy, str) and not callable(reduce_strategy):
        raise ValueError('reduce_strategy must be a string, a callable or None')
    # If reduce_strategy is a string, check if it is a valid string
    if isinstance(reduce_strategy, str) and reduce_strategy not in reduce_strategies.keys():
        raise ValueError('reduce_strategy must be one of {}'.format(reduce_strategies.keys()))
    
    if independent_variables is None:
        independent_variables = list(f.free_symbols)

    constants = list(f.free_symbols - set(independent_variables))
    l = sp.symbols('l')

    tree = ComputationTree(f, independent_variables, constants)
    result = {}
    for var in independent_variables:
        current_node = tree.root
        variable_queue = [current_node] 
        while True:
            if current_node.is_terminal():
                break
            intermediate_vars = current_node.children
            intermediate_vars_bound = []
            for v in intermediate_vars:
                if var in v.active_variables(empirical=empirical):
                    intermediate_vars_bound.append(v)
            if len(intermediate_vars_bound) == 1:
                main_var = intermediate_vars_bound[0]
                variable_queue.append(main_var)
                current_node = main_var
                continue
            else:
                variable_queue.append(var)
                break
        
        operators_degrees = {}
        first_last_op = list(product(['+', '*'], repeat=2))
        for comb in first_last_op:
            whole_comb = [comb[0]]
            first_op = comb[0]
            last_op = comb[1]
            barcode = []
            prev_op = first_op
            for i in range(len(variable_queue)-1):
                op1 = prev_op
                if i == len(variable_queue)-2:
                    op2 = last_op
                    int_f = Perturbation(op1, op2, var)(variable_queue[i].value)
                    deg = degree(int_f,constants=constants,empirical=empirical)
                    whole_comb.append(op2)
                    barcode.append(deg)
                else:
                    deg = {}
                    for op2 in ['+', '*']:
                        int_f = Perturbation(op1, op2, variable_queue[i+1].symbol)(variable_queue[i].value)
                        deg[op2] = degree(int_f,constants=constants,empirical=empirical)
                    if deg['+'] < deg['*']:
                        prev_op = '+'
                    elif deg['+'] > deg['*']:
                        prev_op = '*'
                    else:
                        prev_op = op1
                    whole_comb.append(prev_op)
                    barcode.append(deg[prev_op])
                        
            # Combine consecutive 0s into a single 0
            j = 0
            while j < len(barcode)-1:
                if barcode[j] == 0 and barcode[j+1] == 0:
                    barcode.pop(j)
                else:
                    j += 1
            operators_degrees[tuple(whole_comb)] = barcode

        # Find the operator with the smallest maximum degree
        smallest_max_degree = min([0 if len(i)==0 else max(i) for i in operators_degrees.values()])
        # Among the operators with the smallest maximum degree, find the one with the shortest barcode
        best_comb = min([comb for comb in operators_degrees if (max(operators_degrees[comb]) if len(operators_degrees[comb])>0 else 0) == smallest_max_degree], key=lambda x: len(operators_degrees[x]))
        result[var] = operators_degrees[best_comb]

    if reduce_strategy is not None:
        if isinstance(reduce_strategy, str):
            reduce_strategy = reduce_strategies[reduce_strategy]
        return reduce_strategy(result)
    else:
        return result


def constraint_1(f, independent_variables):
    """
    Checks if complexity_1(f, independent_variables) is equal to 0 for each independent variable.

    Parameters
    ----------
    f : sympy.Expr
        A sympy expression.
    independent_variables : list of sympy.Symbol
        A list of sympy symbols representing the independent variables.

    Returns
    -------
    A boolean indicating whether the constraint is satisfied.
    """
    # Check if the each value in the dictionary is equal to 0
    return all([complexity_1(f, independent_variables)[var] == 0 for var in independent_variables])



def constraint_2(f, independent_variables):
    """
    Checks if complexity_2(f, independent_variables) is at most 1 for each independent variable.

    Parameters
    ----------
    f : sympy.Expr
        A sympy expression.
    independent_variables : list of sympy.Symbol
        A list of sympy symbols representing the independent variables.

    Returns
    -------
    A boolean indicating whether the constraint is satisfied.
    """

    return all([complexity_1(f, independent_variables)[var] <= 1 for var in independent_variables])

def constraint_3(f, independent_variables):
    """
    Checks if for each independent variable complexity_2(f, independent_variables) has length at most two with each degree at most one.

    Parameters
    ----------
    f : sympy.Expr
        A sympy expression.
    independent_variables : list of sympy.Symbol
        A list of sympy symbols representing the independent variables.

    Returns
    -------
    A boolean indicating whether the constraint is satisfied.
    """

    return all([len(complexity_2(f, independent_variables)[var]) <= 2 and all([i <= 1 for i in complexity_2(f, independent_variables)[var]]) for var in independent_variables])

