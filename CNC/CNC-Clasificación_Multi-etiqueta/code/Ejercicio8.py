"""
Escribe un script en Python (cl-cv.py) seleccionando al menos 3 métodos (que NO
pertenezcan todos a la misma categoría) para evaluarlos con 5 de los datasets que recopilaste
anteriormente y calcula las métricas resultantes mediante validación cruzada. El script
mostrará el resultado de las métricas anteriores.

"""
import skmultilearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from skmultilearn.dataset import load_dataset
# here are the [ 'bibtex', 'enron',  'genbase', 'medical', 'yeast','rcv1subset4', 'delicious']
X_bibtex, y_bibtex, feature_names, label_names = load_dataset('bibtex', 'undivided')
X_enron, y_enron, feature_names, label_names = load_dataset('enron', 'undivided')
X_genbase, y_genbase, feature_names, label_names = load_dataset('genbase', 'undivided')
X_medical, y_medical, feature_names, label_names = load_dataset('medical', 'undivided')
X_yeast, y_yeast, feature_names, label_names = load_dataset('yeast', 'undivided')
X_rcv1subset4, y_rcv1subset4, feature_names, label_names = load_dataset('rcv1subset4', 'undivided')
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
# 'rcv1subset4'
X_rcv1subset4 = pd.DataFrame(X_rcv1subset4.toarray())
y_rcv1subset4 = y_rcv1subset4.toarray()
# 'Corel5k'
X_Corel5k = pd.DataFrame(X_Corel5k.toarray())
y_Corel5k = y_Corel5k.toarray()
""" Train test split of each dataset"""
from sklearn.model_selection import train_test_split

# X and y are  features and labels
# test_size to the proportion of data you want to allocate to the test set

X_bibtex_train, X_bibtex_test, y_bibtex_train, y_bibtex_test = train_test_split(X_bibtex, y_bibtex, test_size=0.2, random_state=42)
X_enron_train, X_enron_test, y_enron_train, y_enron_test = train_test_split(X_enron, y_enron, test_size=0.2, random_state=42)
X_genbase_train, X_genbase_test, y_genbase_train, y_genbase_test = train_test_split(X_genbase, y_genbase, test_size=0.2, random_state=42)
X_medical_train, X_medical_test, y_medical_train, y_medical_test = train_test_split(X_medical, y_medical, test_size=0.2, random_state=42)
X_yeast_train, X_yeast_test, y_yeast_train, y_yeast_test = train_test_split(X_yeast, y_yeast, test_size=0.2, random_state=42)
X_rcv1subset4_train, X_rcv1subset4_test, y_rcv1subset4_train, y_rcv1subset4_test = train_test_split(X_rcv1subset4, y_rcv1subset4, test_size=0.2, random_state=42)
X_Corel5k_train, X_Corel5k_test,y_Corel5k_train, y_Corel5k_test = train_test_split(X_Corel5k, y_Corel5k, test_size=0.2, random_state=42)
""" Train MLkNN """
from skmultilearn.adapt import MLkNN
import time

# bibtex
print ("start training MLkNN on bibtex")
start_time = time.time()    # Registra el tiempo de inicio
classifier = MLkNN(k=3)
classifier.fit(X_bibtex_train, y_bibtex_train)        # train 
predictions_MLkNN_bibtex = classifier.predict(X_bibtex_test)       # predict
end_time = time.time()  # Registra el tiempo de finalización
execution_time_bibtex_MLkNN = end_time - start_time    # Calcula la diferencia para obtener el tiempo de ejecución en segundos
print(f"Tiempo de ejecución bibtex-MLkNN: {execution_time_bibtex_MLkNN} segundos") # Imprime el tiempo de ejecución

# enron
print ("start training MLkNN on enron")
start_time = time.time() 
classifier = MLkNN(k=3)
classifier.fit(X_enron_train, y_enron_train)        # train 
predictions_MLkNN_enron = classifier.predict(X_enron_test)       # predict
end_time = time.time()  # Registra el tiempo de finalización
execution_time_enron_MLkNN = end_time - start_time    # Calcula la diferencia para obtener el tiempo de ejecución en segundos
print(f"Tiempo de ejecución enron-MLkNN: {execution_time_enron_MLkNN} segundos") # Imprime el tiempo de ejecución

# genbase
print ("start training MLkNN on genbase")
start_time = time.time() 
classifier = MLkNN(k=3)
classifier.fit(X_genbase_train, y_genbase_train)        # train 
predictions_MLkNN_genbase = classifier.predict(X_genbase_test)       # predict
end_time = time.time()  # Registra el tiempo de finalización
execution_time_genbase_MLkNN = end_time - start_time    # Calcula la diferencia para obtener el tiempo de ejecución en segundos
print(f"Tiempo de ejecución genbase-MLkNN: {execution_time_genbase_MLkNN} segundos") # Imprime el tiempo de ejecución

# medical
print("Start training MLkNN on 'medical'")
start_time = time.time() 
classifier = MLkNN(k=3)
classifier.fit(X_medical_train, y_medical_train)        # train 
predictions_MLkNN_medical = classifier.predict(X_medical_test)       # predict
end_time = time.time()  # Registra el tiempo de finalización
execution_time_medical_MLkNN = end_time - start_time    # Calcula la diferencia para obtener el tiempo de ejecución en segundos
print(f"Tiempo de ejecución medical-MLkNN: {execution_time_medical_MLkNN} segundos") # Imprime el tiempo de ejecución

# yeast
print("Start training MLkNN on 'yeast'")
start_time = time.time() 
classifier = MLkNN(k=3)
classifier.fit(X_yeast_train, y_yeast_train)        # train 
predictions_MLkNN_yeast = classifier.predict(X_yeast_test)       # predict
end_time = time.time()  # Registra el tiempo de finalización
execution_time_yeast_MLkNN = end_time - start_time    # Calcula la diferencia para obtener el tiempo de ejecución en segundos
print(f"Tiempo de ejecución yeast-MLkNN: {execution_time_yeast_MLkNN} segundos") # Imprime el tiempo de ejecución

# rcv1subset4
print("Start training MLkNN on 'rcv1subset4'")
start_time = time.time() 
classifier = MLkNN(k=3)
classifier.fit(X_rcv1subset4_train, y_rcv1subset4_train)        # train 
predictions_MLkNN_rcv1subset4= classifier.predict(X_rcv1subset4_test)       # predict
end_time = time.time()  # Registra el tiempo de finalización
execution_time_rcv1subset4_MLkNN = end_time - start_time    # Calcula la diferencia para obtener el tiempo de ejecución en segundos
print(f"Tiempo de ejecución rcv1subset4-MLkNN: {execution_time_rcv1subset4_MLkNN} segundos") # Imprime el tiempo de ejecución

# Corel5k
print("Start training MLkNN on 'Corel5k'")
start_time = time.time() 
classifier = MLkNN(k=3)
classifier.fit(X_Corel5k_train, y_Corel5k_train)        # train 
predictions_MLkNN_Corel5k= classifier.predict(X_Corel5k_test)       # predict
end_time = time.time()  # Registra el tiempo de finalización
execution_time_Corel5k_MLkNN = end_time - start_time    # Calcula la diferencia para obtener el tiempo de ejecución en segundos
print(f"Tiempo de ejecución Corel5k-MLkNN: {execution_time_Corel5k_MLkNN} segundos") # Imprime el tiempo de ejecución
import sklearn.metrics as metrics

# Assuming you have multiple models and their predictions stored in a list or dictionary
models = ["MLkNN 'bibtex'", "MLkNN 'genbase'","MLkNN 'enron","MLkNN 'medical'","MLkNN 'yeast'","MLkNN 'rcv1subset4'", "MLkNN 'Corel5k'"]  # Add your model names

prediction_dict = {"MLkNN 'bibtex'":predictions_MLkNN_bibtex,
                  "MLkNN 'genbase'": predictions_MLkNN_genbase,
                  "MLkNN 'enron": predictions_MLkNN_enron,
                  "MLkNN 'medical'":predictions_MLkNN_medical,
                  "MLkNN 'yeast'": predictions_MLkNN_yeast,
                  "MLkNN 'rcv1subset4'": predictions_MLkNN_rcv1subset4,
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
                  "MLkNN 'rcv1subset4'": execution_time_rcv1subset4_MLkNN,
                  "MLkNN 'Corel5k'": execution_time_Corel5k_MLkNN}
test_sets = {"MLkNN 'bibtex'":y_bibtex_test,
                  "MLkNN 'genbase'": y_genbase_test,
                  "MLkNN 'enron": y_enron_test,
                  "MLkNN 'medical'": y_medical_test,
                  "MLkNN 'yeast'": y_yeast_test,
                  "MLkNN 'rcv1subset4'": y_rcv1subset4_test,
                  "MLkNN 'Corel5k'": y_Corel5k_test}
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
metrics_MLkNN.to_csv('metrics_MLkNN_8_1.csv', index=True)
print(metrics_MLkNN)