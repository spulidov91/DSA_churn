import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def load_data(ruta):
    """Carga los datos desde un archivo CSV y realiza el preprocesamiento."""
    datos = pd.read_csv(ruta)

    # Identificar columnas categóricas y numéricas
    col_cat = datos.drop('Attrition_Flag', axis=1).select_dtypes(exclude=['int64', 'float64']).columns
    col_num = datos.drop(['Attrition_Flag', 'CLIENTNUM'], axis=1).select_dtypes(include=['int64', 'float64']).columns

    # Eliminar columna innecesaria
    datos = datos.drop('CLIENTNUM', axis=1)

    # Reemplazar la variable de respuesta por 0 y 1
    datos['Attrition_Flag'] = datos['Attrition_Flag'].replace({'Attrited Customer': 1, 'Existing Customer': 0})

    # Reemplazar en la columna gender F -> 1, M -> 0
    datos['Gender'] = datos['Gender'].replace({'F': 1, 'M': 0})

    # Aplicar one-hot encoding a las columnas categóricas
    datos = pd.get_dummies(datos, columns=col_cat, drop_first=False, dtype=int)

    # Aplicar StandardScaler solo a las columnas numéricas
    scaler = StandardScaler()
    datos[col_num] = scaler.fit_transform(datos[col_num])

    # División de datos en X e y
    y = datos.pop('Attrition_Flag')
    X = datos

    # División en conjunto de entrenamiento y prueba
    x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=69)

    # Aplicación de SMOTE para balancear las clases en el conjunto de entrenamiento
    sm = SMOTE(sampling_strategy='auto', random_state=1234)
    x_train, y_train = sm.fit_resample(x_train, y_train)

    return x_train, x_test, y_train, y_test