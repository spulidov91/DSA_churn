import json
from typing import Any

import numpy as np
import pandas as pd
import joblib
from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder
from loguru import logger
#from model import __version__ as model_version
#from model.model import make_prediction

from app import __version__, schemas
from app.config import settings

api_router = APIRouter()
modelo = joblib.load('despliegue/docker-api/api/model/modelo_gb.pkl')
scaler = joblib.load('despliegue/docker-api/api/model/scaler.pkl')


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesa los datos de entrada para que coincidan con los requisitos del modelo.
    """
    cat_col = data.select_dtypes(include=['object', 'category']).columns
    cat_col = cat_col.drop('Attrition_Flag', errors= 'ignore')
    
    num_col = data.select_dtypes(include=['int', 'double']).columns
    num_col = num_col.drop('CLIENTNUM', errors= 'ignore')   

    # Reemplazar valores categóricos en la columna 'Gender'
    data['Gender'] = data['Gender'].replace({'F': 1, 'M': 0})

    # Aplicar One-Hot Encoding a las columnas categóricas
    data = pd.get_dummies(data, columns=cat_col, drop_first=False, dtype=int)

    # Asegurar que todas las columnas requeridas por el modelo están presentes
    for col in modelo.feature_names_in_:
        if col not in data.columns:
            data[col] = 0

    # Escalar las columnas numéricas
    data[num_col] = scaler.transform(data[num_col])

    # Devolver solo las columnas requeridas por el modelo
    return data[modelo.feature_names_in_]


# Ruta para verificar que la API se esté ejecutando correctamente
@api_router.get("/health", response_model=schemas.Health, status_code=200)
def health() -> dict:
    """
    Root Get
    """
    health = schemas.Health(
        name=settings.PROJECT_NAME, api_version=__version__, model_version='0.01'
    )

    return health.dict()

# Ruta para realizar las predicciones
@api_router.post("/predict", response_model=schemas.PredictionResults, status_code=200)
async def predict(input_data: schemas.MultipleDataInputs) -> Any:
    """
    Prediccion usando el modelo de bankchurn
    """

    input_df = pd.DataFrame(jsonable_encoder(input_data.inputs))

    logger.info(f"Making prediction on inputs: {input_data.inputs}")
    
    data_preprocessed = preprocess_data(input_df)
    results = modelo.predict(input_data=data_preprocessed.replace({np.nan: None}))

    if results["errors"] is not None:
        logger.warning(f"Prediction validation error: {results.get('errors')}")
        raise HTTPException(status_code=400, detail=json.loads(results["errors"]))

    logger.info(f"Prediction results: {results.get('predictions')}")

    return results