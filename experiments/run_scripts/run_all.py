import os
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
import sympy
from utils import *
from gplearn_sgpa.genetic import SymbolicRegressor, _convert_to_sympy
import xgboost as xgb
from interpret.glassbox import ExplainableBoostingRegressor


POPULATION_SIZE = 15000
# POPULATION_SIZE = 10
GENERATIONS = 50
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
    rmse = mean_absolute_error(y_test, y_pred)
    return rmse

def run(dataset_name,constraint,seed,complexities_to_track=False):
    X_train, X_test, y_train, y_test = load_dataset(dataset_name, seed=seed)
    fitness_function = create_fitness_function(constraint)
    model = train(X_train,y_train,fitness_function,seed,complexities_to_track=complexities_to_track)
    rmse = test(model,X_test,y_test)
    sympy_expr = _convert_to_sympy(model._program.program)

    return model,rmse,sympy_expr, model._program

def save(dataset_name,constraint_name,rmse,sympy_expr,program):
    # Create a folder if does not exist
    folder_path = f'results/{dataset_name}'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    latex_str = sympy.latex(sympy_expr)

    length = len(program.program)
    
    # Save the results
    with open(f'{folder_path}/results.txt', 'a') as f:
        f.write(f'dataset={dataset_name}, constraint={constraint_name}, rmse={rmse}, expression={latex_str}, program={program}, len={length}\n')

def train_xgb(X_train,y_train):
    xgb_model = xgb.XGBRegressor()
    xgb_model.fit(X_train,y_train)
    return xgb_model

def test_xgb(xgb_model,X_test,y_test):
    y_pred = xgb_model.predict(X_test)
    rmse = mean_absolute_error(y_test, y_pred)
    return rmse

def train_ebm(X_train,y_train):
    ebm_model = ExplainableBoostingRegressor(interactions=0)
    ebm_model.fit(X_train,y_train)
    return ebm_model

def test_ebm(ebm_model,X_test,y_test):
    y_pred = ebm_model.predict(X_test)
    rmse = mean_absolute_error(y_test, y_pred)
    return rmse

def main():

    datasets = ['concrete']
    # datasets = ['pollen']
    # sgpa_constraints = [constraint_1,constraint_2,constraint_3]
    # sgpa_constraints_names = ['constraint_1','constraint_2','constraint_3']
    sgpa_constraints = [constraint_1]
    sgpa_constraints_names = ['constraint_1']
    max_lengths = []
    # max_lengths = []

    for dataset in datasets:
        print(f'Running experiments for dataset={dataset}')

        # Train an XGBoost model
        # X_train, X_test, y_train, y_test = load_dataset(dataset, seed=SEED)
        # xgb_model = train_xgb(X_train,y_train)
        # xgb_rmse = test_xgb(xgb_model,X_test,y_test)
        # print(f'XGBoost RMSE={xgb_rmse}')

        # Train EBM model
        # ebm_model = train_ebm(X_train,y_train)
        # ebm_rmse = test_ebm(ebm_model,X_test,y_test)
        # print(f'EBM RMSE={ebm_rmse}')

        # Train linear regression
        


        for max_length in max_lengths:
            print(f'Running experiments for max_length={max_length}')
            length_constraint = create_length_constraint(max_length)
            model,rmse,sympy_expr, program = run(dataset,length_constraint,SEED,complexities_to_track=['length'])
            save(dataset,f'length={max_length}',rmse,sympy_expr,program)
            print(len(program.program))
        
    
        for constraint,constraint_name in zip(sgpa_constraints,sgpa_constraints_names):
            print(f'Running experiments for constraint={constraint_name}')
            if '3' in constraint_name:
                complexities_to_track = ['c2']
            else:
                complexities_to_track = ['c1','length']
                
            model,rmse,sympy_expr, program = run(dataset,constraint,SEED,complexities_to_track=complexities_to_track)
            save(dataset,constraint_name,rmse,sympy_expr,program)
            print(len(program.program))


if __name__ == '__main__':
    main() 