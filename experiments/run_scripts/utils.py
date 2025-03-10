import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

def create_fitness_function(constraint):
    def fitness_function(raw_fitness,complexity):
        if constraint(complexity):
            return raw_fitness
        else:
            return raw_fitness + 1e6
    return fitness_function


def constraint_1(complexity):
    c1_complexity = complexity['c1']
    length = complexity['length']
    if len(c1_complexity) == 0:
        if length < 200:
            return True
        else:
            return False
    max_degree = np.max([v for k,v in c1_complexity.items()])
    if max_degree > 0:
        return False
    else:
        if length < 200:
            return True
        else:
            return False

def constraint_2(complexity):
    c1_complexity = complexity['c1']
    if len(c1_complexity) == 0:
        return True
    max_degree = np.max([v for k,v in c1_complexity.items()])
    if max_degree > 1:
        return False
    else:
        return True
    
def constraint_3(complexity):
    c2_complexity = complexity['c2']
    if len(c2_complexity) == 0:
        return True
    for k,v in c2_complexity.items():
        if len(v) > 2:
            return False
        else:
            if len(v) == 0:
                continue
            elif max(v) > 1:
                return False
    return True

def constraint_4(complexity):
    c1_complexity = complexity['c1']
    if len(c1_complexity) == 0:
        return True
    degrees_list = [v for k,v in c1_complexity.items()]
    median_degree = np.median(degrees_list)
    if median_degree > 1:
        return False
    else:
        return True

def create_length_constraint(max_length):
    def length_constraint(complexity):
        if complexity['length'] > max_length:
            return False
        else:
            return True
    return length_constraint


def load_concrete_dataset(seed=42):
    
    dataset = fetch_openml(data_id=4353)
    data = dataset.data
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    # X_scales = np.array([1000,100,100,100,10,1000,1000,365]).T
    # X = X / X_scales
    # y = y / 100

    # Normalize the data
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    y = (y - y.mean()) / y.std()
    

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    return X_train, X_test, y_train, y_test


def load_dataset(name,seed=42):
    if name == 'concrete':
        return load_concrete_dataset(seed=seed)
    else:
        raise ValueError(f"Dataset {name} not found")
    


