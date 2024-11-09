# Primero colocar en la terminal: mlflow ui

import sys
import os
import multiprocessing
import mlflow
import mlflow.sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, RepeatedKFold
from sklearn.metrics import classification_report, accuracy_score

# Agregar la carpeta raíz al PYTHONPATH para poder importar data_preprocessing
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.data_preprocessing import load_data

# Establece la URI de seguimiento para el servidor local de MLflow
mlflow.set_tracking_uri('http://localhost:5000')

# Configuración del nombre del experimento en MLflow
experiment_name = 'gb'
mlflow.set_experiment(experiment_name)

# Carga y preprocesamiento de datos
x_train, x_test, y_train, y_test = load_data(r'data/BankChurners.csv')

# Definición de los hiperparámetros
param_grid = { 
    'max_iter': [100, 500, 1000],
    'max_depth': [5, 10, 20],
    'learning_rate': [0.001, 0.1]    
}

# Configuración de GridSearchCV
grid = GridSearchCV(
    estimator=HistGradientBoostingClassifier(random_state=223),
    param_grid=param_grid,
    scoring='accuracy',
    cv=RepeatedKFold(n_splits=5, n_repeats=5, random_state=12),
    n_jobs=multiprocessing.cpu_count() - 1,
    refit=True,
    verbose=0,
    return_train_score=True
)

grid.fit(x_train, y_train)

# Obtener el mejor modelo y sus parámetros
best_model = grid.best_estimator_
best_params = grid.best_params_
best_score = grid.best_score_

# Iniciar un run en MLflow y registrar el mejor modelo
with mlflow.start_run() as run:
    # Registrar los mejores parámetros
    mlflow.log_params(best_params)
    y_pred = best_model.predict(x_test)
    
    # Métricas
    test_accuracy = accuracy_score(y_test, y_pred)
    mlflow.log_metric("test_accuracy", test_accuracy)
    
    # Registro
    report = classification_report(y_test, y_pred, output_dict=True)
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            mlflow.log_metric(f"{label}_f1-score", metrics['f1-score'])
            mlflow.log_metric(f"{label}_recall", metrics['recall'])
            mlflow.log_metric(f"{label}_precision", metrics['precision'])
            mlflow.log_metric(f"{label}_support", metrics['support'])

    mlflow.log_metric("best_cv_accuracy", best_score)
    
    mlflow.sklearn.log_model(best_model, "best_model")
    
    print(f"Mejor modelo guardado en MLflow con accuracy: {best_score}")

