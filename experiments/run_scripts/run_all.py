import os
from sklearn.metrics import mean_absolute_error
import sympy
from utils import *
from gplearn_sgpa.genetic import SymbolicRegressor, _convert_to_sympy

POPULATION_SIZE = 5000
GENERATIONS = 30

N_JOBS = 18
FUNCTION_SET = ('add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'sin', 'exp')
SEED = 0

def train(X_train,y_train,fitness_function,seed,complexities_to_track=['length']):
    sr_regressor = SymbolicRegressor(population_size=POPULATION_SIZE,
                                        generations=GENERATIONS,
                                        function_set=FUNCTION_SET,
                                        fitness_function=fitness_function,
                                        n_jobs=N_JOBS,
                                        verbose=1,
                                        random_state=seed,
                                        complexities_to_track=complexities_to_track,
                                        )

    sr_regressor.fit(X_train, y_train)
    return sr_regressor

def test(sr_regressor,X_test,y_test):
    y_pred = sr_regressor.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    return mae

def run(dataset_name,constraint,seed,complexities_to_track=False):
    X_train, X_test, y_train, y_test = load_dataset(dataset_name, seed=seed)
    fitness_function = create_fitness_function(constraint)
    model = train(X_train,y_train,fitness_function,seed,complexities_to_track=complexities_to_track)
    mae = test(model,X_test,y_test)
    sympy_expr = _convert_to_sympy(model._program.program)

    return model,mae,sympy_expr, model._program

def save(dataset_name,constraint_name,mae,sympy_expr,program):
    # Create a folder if does not exist
    folder_path = f'results/{dataset_name}'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    latex_str = sympy.latex(sympy_expr)

    length = len(program.program)
    
    # Save the results
    with open(f'{folder_path}/results.txt', 'a') as f:
        f.write(f'dataset={dataset_name}, constraint={constraint_name}, mae={mae}, expression={latex_str}, program={program}, len={length}\n')

def main():

    datasets = ['concrete']
    sgpa_constraints = [constraint_2]
    sgpa_constraints_names = ['constraint_2']
    max_lengths = [10, 20, 50, 90]

    for dataset in datasets:
        print(f'Running experiments for dataset={dataset}')

        for max_length in max_lengths:
            print(f'Running experiments for max_length={max_length}')
            length_constraint = create_length_constraint(max_length)
            model,mae,sympy_expr, program = run(dataset,length_constraint,SEED,complexities_to_track=['length'])
            save(dataset,f'length={max_length}',mae,sympy_expr,program)
            print(len(program.program))
        
    
        for constraint,constraint_name in zip(sgpa_constraints,sgpa_constraints_names):
            print(f'Running experiments for constraint={constraint_name}')
            if '3' in constraint_name:
                complexities_to_track = ['c2']
            else:
                complexities_to_track = ['c1','length']
                
            model,mae,sympy_expr, program = run(dataset,constraint,SEED,complexities_to_track=complexities_to_track)
            save(dataset,constraint_name,mae,sympy_expr,program)
            print(len(program.program))


if __name__ == '__main__':
    main() 