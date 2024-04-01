"""
Escribe un script en Python (cl-cv.py) seleccionando al menos 3 métodos (que NO
pertenezcan todos a la misma categoría) para evaluarlos con 5 de los datasets que recopilaste
anteriormente y calcula las métricas resultantes mediante validación cruzada. El script
mostrará el resultado de las métricas anteriores.

"""
import skmultilearn
import pandas as pd
import numpy as np
from skmultilearn.dataset import load_dataset
# here are the [ 'bibtex', 'enron',  'genbase', 'medical', 'yeast','rcv1subset4', 'delicious']
X_bibtex, y_bibtex, feature_names, label_names = load_dataset('bibtex', 'undivided')
X_enron, y_enron, feature_names, label_names = load_dataset('enron', 'undivided')
X_genbase, y_genbase, feature_names, label_names = load_dataset('genbase', 'undivided')
X_medical, y_medical, feature_names, label_names = load_dataset('medical', 'undivided')
X_yeast, y_yeast, feature_names, label_names = load_dataset('yeast', 'undivided')
X_Corel5k, y_Corel5k, feature_names, label_names = load_dataset('Corel5k', 'undivided')
#
import pandas as pd
X_bibtex = pd.DataFrame(X_bibtex.toarray())
y_bibtex = y_bibtex.toarray()
# 'enron'
X_enron = pd.DataFrame(X_enron.toarray())
y_enron = y_enron.toarray()
# 'genbase'
X_genbase = pd.DataFrame(X_genbase.toarray())
y_genbase = y_genbase.toarray()
# 'medical'
X_medical = pd.DataFrame(X_medical.toarray())
y_medical = y_medical.toarray()
# 'yeast'
X_yeast = pd.DataFrame(X_yeast.toarray())
y_yeast = y_yeast.toarray()
# 'Corel5k'
X_Corel5k = pd.DataFrame(X_Corel5k.toarray())
y_Corel5k = y_Corel5k.toarray()
""" Train test split of each dataset"""

# X and y are  features and labels
""" Train MLkNN """
from skmultilearn.adapt import MLkNN
import time
from sklearn.model_selection import KFold
import sklearn.metrics as metrics
# bibtex
print ("start training MLkNN on bibtex")
start_time = time.time()    # Registra el tiempo de inicio
kf = KFold(n_splits=5, shuffle=True, random_state=42)
# Initialize an array to store predictions
predictions_MLkNN_bibtex = np.empty_like(y_bibtex, dtype=int)

for train_index, test_index in kf.split(X_bibtex,y_bibtex):
    X_train, X_test = X_bibtex.iloc[train_index], X_bibtex.iloc[test_index]
    y_train, y_test =  y_bibtex[train_index], y_bibtex[test_index]
    # Initialize the classifier
    classifier = MLkNN(k=3)
    # Train the classifier
    classifier.fit(X_train, y_train)
    # Predict on the test set
    predictions = classifier.predict(X_test).toarray()  # Convert predictions to array
    # Store the predictions for each label in the corresponding test indices
    for label_index in range(y_bibtex.shape[1]):
        predictions_MLkNN_bibtex[test_index, label_index] = predictions[:, label_index]
end_time = time.time()  # Registra el tiempo de finalización
execution_time_bibtex_MLkNN = end_time - start_time    # Calcula la diferencia para obtener el tiempo de ejecución en segundos
print(f"Tiempo de ejecución bibtex-MLkNN: {execution_time_bibtex_MLkNN} segundos") # Imprime el tiempo de ejecución

# enron
print ("start training MLkNN on enron")
start_time = time.time() 
kf = KFold(n_splits=5, shuffle=True, random_state=42)
# Initialize an array to store predictions
predictions_MLkNN_enron = np.empty_like(y_enron, dtype=int)

for train_index, test_index in kf.split(X_enron,y_enron):
    X_train, X_test = X_enron.iloc[train_index], X_enron.iloc[test_index]
    y_train, y_test =  y_enron[train_index], y_enron[test_index]
    # Initialize the classifier
    classifier = MLkNN(k=3)
    # Train the classifier
    classifier.fit(X_train, y_train)
    # Predict on the test set
    predictions = classifier.predict(X_test).toarray()  # Convert predictions to array
    # Store the predictions for each label in the corresponding test indices
    for label_index in range(y_enron.shape[1]):
        predictions_MLkNN_enron[test_index, label_index] = predictions[:, label_index]
end_time = time.time()  # Registra el tiempo de finalización
execution_time_enron_MLkNN = end_time - start_time    # Calcula la diferencia para obtener el tiempo de ejecución en segundos
print(f"Tiempo de ejecución enron-MLkNN: {execution_time_enron_MLkNN} segundos") # Imprime el tiempo de ejecución

# genbase
print ("start training MLkNN on genbase")
start_time = time.time() 
kf = KFold(n_splits=5, shuffle=True, random_state=42)
# Initialize an array to store predictions
predictions_MLkNN_genbase = np.empty_like(y_genbase, dtype=int)

for train_index, test_index in kf.split(X_genbase,y_genbase):
    X_train, X_test = X_genbase.iloc[train_index], X_genbase.iloc[test_index]
    y_train, y_test =  y_genbase[train_index], y_genbase[test_index]
    # Initialize the classifier
    classifier = MLkNN(k=3)
    # Train the classifier
    classifier.fit(X_train, y_train)
    # Predict on the test set
    predictions = classifier.predict(X_test).toarray()  # Convert predictions to array
    # Store the predictions for each label in the corresponding test indices
    for label_index in range(y_genbase.shape[1]):
        predictions_MLkNN_genbase[test_index, label_index] = predictions[:, label_index]
end_time = time.time()  # Registra el tiempo de finalización
execution_time_genbase_MLkNN = end_time - start_time    # Calcula la diferencia para obtener el tiempo de ejecución en segundos
print(f"Tiempo de ejecución genbase-MLkNN: {execution_time_genbase_MLkNN} segundos") # Imprime el tiempo de ejecución
# medical
print("Start training MLkNN on 'medical'")
start_time = time.time() 
# Initialize an array to store predictions
predictions_MLkNN_medical = np.empty_like(y_medical, dtype=int)

for train_index, test_index in kf.split(X_medical,y_medical):
    X_train, X_test = X_medical.iloc[train_index], X_medical.iloc[test_index]
    y_train, y_test =  y_medical[train_index], y_medical[test_index]
    # Initialize the classifier
    classifier = MLkNN(k=3)
    # Train the classifier
    classifier.fit(X_train, y_train)
    # Predict on the test set
    predictions = classifier.predict(X_test).toarray()  # Convert predictions to array
    # Store the predictions for each label in the corresponding test indices
    for label_index in range(y_medical.shape[1]):
        predictions_MLkNN_medical[test_index, label_index] = predictions[:, label_index]
end_time = time.time()  # Registra el tiempo de finalización
execution_time_medical_MLkNN = end_time - start_time    # Calcula la diferencia para obtener el tiempo de ejecución en segundos
print(f"Tiempo de ejecución medical-MLkNN: {execution_time_medical_MLkNN} segundos") # Imprime el tiempo de ejecución

# yeast
print("Start training MLkNN on 'yeast'")
start_time = time.time() 
# Initialize an array to store predictions
predictions_MLkNN_yeast = np.empty_like(y_yeast, dtype=int)
for train_index, test_index in kf.split(X_yeast,y_yeast):
    X_train, X_test = X_yeast.iloc[train_index], X_yeast.iloc[test_index]
    y_train, y_test =  y_yeast[train_index], y_yeast[test_index]
    # Initialize the classifier
    classifier = MLkNN(k=3)
    # Train the classifier
    classifier.fit(X_train, y_train)
    # Predict on the test set
    predictions = classifier.predict(X_test).toarray()  # Convert predictions to array
    # Store the predictions for each label in the corresponding test indices
    for label_index in range(y_yeast.shape[1]):
        predictions_MLkNN_yeast[test_index, label_index] = predictions[:, label_index]
end_time = time.time()  # Registra el tiempo de finalización
execution_time_yeast_MLkNN = end_time - start_time    # Calcula la diferencia para obtener el tiempo de ejecución en segundos
print(f"Tiempo de ejecución yeast-MLkNN: {execution_time_yeast_MLkNN} segundos") # Imprime el tiempo de ejecución

# Corel5k
print("Start training MLkNN on 'Corel5k'")
start_time = time.time() 
# Initialize an array to store predictions
predictions_MLkNN_Corel5k = np.empty_like(y_Corel5k, dtype=int)

for train_index, test_index in kf.split(X_Corel5k,y_Corel5k):
    X_train, X_test = X_Corel5k.iloc[train_index], X_Corel5k.iloc[test_index]
    y_train, y_test =  y_Corel5k[train_index], y_Corel5k[test_index]
    # Initialize the classifier
    classifier = MLkNN(k=3)
    # Train the classifier
    classifier.fit(X_train, y_train)
    # Predict on the test set
    predictions = classifier.predict(X_test).toarray()  # Convert predictions to array
    # Store the predictions for each label in the corresponding test indices
    for label_index in range(y_Corel5k.shape[1]):
        predictions_MLkNN_Corel5k[test_index, label_index] = predictions[:, label_index]
end_time = time.time()  # Registra el tiempo de finalización
execution_time_Corel5k_MLkNN = end_time - start_time    # Calcula la diferencia para obtener el tiempo de ejecución en segundos
print(f"Tiempo de ejecución Corel5k-MLkNN: {execution_time_Corel5k_MLkNN} segundos") # Imprime el tiempo de ejecución

import sklearn.metrics as metrics

# Assuming you have multiple models and their predictions stored in a list or dictionary
models = ["MLkNN 'bibtex'", "MLkNN 'genbase'","MLkNN 'enron","MLkNN 'medical'","MLkNN 'yeast'", "MLkNN 'Corel5k'"]  # Add your model names

prediction_dict = {"MLkNN 'bibtex'":predictions_MLkNN_bibtex,
                  "MLkNN 'genbase'": predictions_MLkNN_genbase,
                  "MLkNN 'enron": predictions_MLkNN_enron,
                  "MLkNN 'medical'":predictions_MLkNN_medical,
                  "MLkNN 'yeast'": predictions_MLkNN_yeast,
                  "MLkNN 'Corel5k'": predictions_MLkNN_Corel5k}
# Create a DataFrame to store the metrics
metrics_MLkNN= pd.DataFrame(columns=[
    'Hamming Loss', 'Recall (micro)', 'Recall (macro)',
    'Accuracy', 'F1-score (micro)', 'F1-score (macro)',
    'Precision (micro)', 'Precision (macro)', 'execution time (s)'
])
execution_time_g = {"MLkNN 'bibtex'":execution_time_bibtex_MLkNN,
                  "MLkNN 'genbase'": execution_time_genbase_MLkNN,
                  "MLkNN 'enron": execution_time_enron_MLkNN,
                  "MLkNN 'medical'":execution_time_medical_MLkNN,
                  "MLkNN 'yeast'": execution_time_yeast_MLkNN,
                  "MLkNN 'Corel5k'": execution_time_Corel5k_MLkNN}
test_sets = {"MLkNN 'bibtex'":y_bibtex,
                  "MLkNN 'genbase'": y_genbase,
                  "MLkNN 'enron": y_enron,
                  "MLkNN 'medical'": y_medical,
                  "MLkNN 'yeast'": y_yeast,
                  "MLkNN 'Corel5k'": y_Corel5k}
# Add metrics for each model
for model in models:
    hamming_loss = metrics.hamming_loss(test_sets[model], prediction_dict[model])
    recall_micro = metrics.recall_score(test_sets[model], prediction_dict[model], average='micro', zero_division=1)
    recall_macro = metrics.recall_score(test_sets[model], prediction_dict[model], average='macro', zero_division=1)
    accuracy = metrics.accuracy_score(test_sets[model], prediction_dict[model])
    f1_micro = metrics.f1_score(test_sets[model], prediction_dict[model], average='micro',zero_division=1)
    f1_macro = metrics.f1_score(test_sets[model], prediction_dict[model], average='macro', zero_division=1)
    precision_micro = metrics.precision_score(test_sets[model], prediction_dict[model], average='micro', zero_division=1)
    precision_macro = metrics.precision_score(test_sets[model], prediction_dict[model], average='macro', zero_division=1)
    execution_time =execution_time_g[model]

    metrics_MLkNN.loc[model] = [
        hamming_loss, recall_micro, recall_macro,
        accuracy, f1_micro, f1_macro,
        precision_micro, precision_macro, execution_time
    ]
metrics_MLkNN.to_csv('metrics_MLkNN_8KFold.csv', index=True)
print(metrics_MLkNN)