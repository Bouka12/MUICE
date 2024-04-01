
""" **Ejecricio 3**: Escribe un script en Python (car.py) que calcule para cada uno de los datasets de los dos
ejercicios anteriores las siguientes medidas de caracterización de cada dataset (como mínimo
los estadísticos vistos en teoría) y que las muestra en pantalla de forma ordenada:
a. number of instances (n)
b. number of attributes (f)
c. number of labels (l)
d. cardinality (car)
e. density (den)
f. diversity (div, represents the percentage of labelsets present in the dataset divided by
the number of possible labelsets)
g. average Imbalance Ratio per label (avgIR, measures the average degree of imbalance
of all labels, the greater avgIR, the greater the imbalance of the dataset)
h. ratio of unconditionally dependent label pairs by chi-square test (rDep, measures the
"""

# Ejercicio 3
from skmultilearn.dataset import load_dataset
from itertools import combinations
import numpy as np
from scipy.stats import chi2_contingency

# Define functions for calculations
def number_of_instances(X):
    return X.shape[0]

def number_of_attributes(X):
    return X.shape[1]

def number_of_labels(y):
    return y.shape[1]

# Function to calculate cardinality
def calculate_cardinality(y):
    return (1 / y.shape[0]) * np.sum(np.abs(y))
import pandas as pd
# Function to calculate density
def calculate_density(y):
    return np.mean(y)

# Function to calculate diversity
import scipy.sparse as sp
def calculate_diversity(y):
    unique_label_sets = len({tuple(row) for row in y.toarray()})  # Convert lil_matrix to array for set operations
    possible_label_sets = 2 ** y.shape[1]
    print("unique_label_sets: ", unique_label_sets)
    diversity = unique_label_sets / possible_label_sets
    return diversity

# Function to calculate imbalance ratio for a label

def calculate_imbalance_ratio(y):
    label_counts = np.sum(y, axis=0)
    max_label_count = np.max(label_counts)
    imbalance_ratios = max_label_count / label_counts
    avgIR = np.mean(imbalance_ratios)
    return avgIR
# Function to convert label sets to integers
def label_sets_to_integers(y):
    return y.dot(1 << np.arange(y.shape[-1] - 1, -1, -1))

# Function to calculate ratio of unconditionally dependent label pairs using chi-square test
def calculate_dependent_label_pairs(y):
    print("shape of y", y.shape)
    label_pairs = list(combinations(range(y.shape[1]), 2))
    dependent_pairs = 0

    for pair in label_pairs:
        contingency_table = np.zeros((2, 2), dtype=np.int64)

        label_1 = y[:, pair[0]].toarray().flatten()
        label_2 = y[:, pair[1]].toarray().flatten()

        contingency_table[0, 0] = np.sum((label_1 == 0) & (label_2 == 0))  # Both labels absent
        contingency_table[0, 1] = np.sum((label_1 == 0) & (label_2 == 1))  # Label 1 absent, Label 2 present
        contingency_table[1, 0] = np.sum((label_1 == 1) & (label_2 == 0))  # Label 1 present, Label 2 absent
        contingency_table[1, 1] = np.sum((label_1 == 1) & (label_2 == 1))  # Both labels present

        _, p, _, _ = chi2_contingency(contingency_table)

        if p < 0.01:  # Considering 99% confidence level
            dependent_pairs += 1

    total_pairs = len(label_pairs)
    return dependent_pairs / total_pairs
# List of available datasets
datasets_list = [ 'delicious']

def carFunc(datasetslist):
    # Iterate through datasets and calculate measures
    for dataset_name in datasetslist:
    # Load the dataset
        X, y, _, _ = load_dataset(dataset_name, 'undivided')
    # Print the data types of X and y
        print(f"Data type of X: {type(X)}")
        print(f"Data type of y: {type(y)}")
    # Calculate measures
        n_instances = number_of_instances(X)
        n_attributes = number_of_attributes(X)
        n_labels = number_of_labels(y)
        car = calculate_cardinality(y)
        den = calculate_density(y)
        div = calculate_diversity(y)
        avgIR = calculate_imbalance_ratio(y)
        rDep = calculate_dependent_label_pairs(y)

     # Print results
        print(f"Dataset: {dataset_name}")
        print(f"Number of instances (n): {n_instances}")
        print(f"Number of attributes (f): {n_attributes}")
        print(f"Number of labels (l): {n_labels}")
        print(f"Cardinality (car): {car}")
        print(f"Density (den): {den}")
        print(f"Diversity (div): {div}")
        print(f"Average Imbalance Ratio per label (avgIR): {avgIR}")
        print(f"Ratio of unconditionally dependent label pairs (rDep): {rDep}")
        print("\n")  # Add a separator between datasets


carFunc(datasets_list)
