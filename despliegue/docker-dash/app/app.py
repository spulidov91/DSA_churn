import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import random
import datetime
import joblib
import os
import requests
import json

df = pd.read_csv('BankChurners.csv')
df.drop(columns = ['Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1', 
                      'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2' ], inplace = True)
num_col = df.select_dtypes(include=['int', 'double']).columns
num_col = num_col.drop('CLIENTNUM', errors= 'ignore')
cat_col = df.select_dtypes(include=['object', 'category']).columns
cat_col = cat_col.drop('Attrition_Flag', errors= 'ignore')
modelo = joblib.load('modelo_gb.pkl')
scaler = joblib.load('scaler.pkl')

api_url = os.getenv('API_URL')
api_url = "http://{}:8001/api/v1/predict".format(api_url)

def preprocess_data(data):
    data['Gender'] = data['Gender'].replace({'F': 1, 'M': 0})
    # Aplicar one-hot encoding
    data = pd.get_dummies(data, columns=cat_col, drop_first=False, dtype=int)

    # Asegurar que las columnas coincidan con el modelo
    for col in modelo.feature_names_in_:
        if col not in data.columns:
            data[col] = 0
     # Aplicar el scaler solo a las columnas numéricas
    data[num_col] = scaler.transform(data[num_col])

    return data[modelo.feature_names_in_]

st.set_page_config(layout="wide")

# Filtro de fecha
fecha_inicio = datetime.date(2022, 1, 1)
fecha_fin = datetime.date(2024, 12, 31)


def page_EDA():
    # ---- Titulo-----
    st.title('🏦 Tendencias de Churn de clientes con tarjeta de crédito')

    # Metricas
    attrition_percentage = round(df['Attrition_Flag'].value_counts(normalize=True) * 100,1)
    col1, col2, col3 = st.columns(3)
    col1.metric('% Tasa de abandono', attrition_percentage.iloc[1], "-12%")
    col2.metric('% Tasa de retencion', attrition_percentage.iloc[0], "12%")
    col3.metric('Cantidad de cleintes nuevos', random.randint(5,20), "5")

    # ------------- EDA ------------------

    st.subheader('Análisis descriptivo')
    col1, col2 = st.columns(2)
    with col1: 
        # Filtros
        cat_sel = st.selectbox('Selecciona una variable categórica', 
                        cat_col)
        # Grafico de caracterización
        bar_counts = df.groupby([cat_sel, 'Attrition_Flag']).size().reset_index(name = 'Count')

        fig = px.bar(
            bar_counts,
            x='Count',              
            y=cat_sel,               
            color='Attrition_Flag',  
            orientation='h',         
            title=f'Conteo de {cat_sel} por Attrition Flag'
        )       
        st.plotly_chart(fig, use_container_width = True)
        # Crear el gráfico de distribución
    with col2:
        num_sel = st.selectbox('Seleccione una variable numérica',num_col)
        fig = px.box(df, 
                        x=df[cat_sel], 
                        y=df[num_sel], 
                        color='Attrition_Flag',
                        title=f'Dstribución de {cat_sel} respecto a {num_sel} ')

        st.plotly_chart(fig, use_container_width = True)

    # Grafico de distribución variables numericas
    # Variable 1
    st.subheader('Análisis de correlaciones')
    col1, col2 = st.columns(2)
    with col1:
        var1 = st.selectbox('Selecciona una variable (Eje x)', 
                            num_col, 
                            key = 'var_num_1',
                            index=2)
    with col2:
        var2 = st.selectbox('Selecciona una variable (Eje y)', num_col, key = 'var_num_2')

    fig = px.scatter(df, x=df[var1], y=df[var2], color = 'Attrition_Flag'
                ,hover_name='CLIENTNUM')
        
    st.plotly_chart(fig, use_container_width=True)

def page_Prediction():
    ## Grafico de predicción
    ## Se realiza con datos aleatorios
    # Creación de formulario para la predicción
       
    st.title('🙎Predicción para decersión de clientes')
    col1, col2 = st.columns(2)
    with col1:   
        st.subheader('Predicción de deserción de clientes nuevos por semana')
        st.date_input('Seleccione el periodo de tiempo a evaluar',
                            (fecha_inicio, datetime.date(2022, 1, 7)),
                            fecha_inicio,
                            fecha_fin,
                            format='DD.MM.YYYY'
                        )
        
        semana = range(1,26)
        val_cliente = [random.randint(4,20) for _ in range(25)]
        val_decersion = [random.randint(1, int(valor*0.9)) for valor in val_cliente]
        df_decersion = pd.DataFrame({
            'Semana': semana,
            'Val_Cliente': val_cliente,
            'Val_Desercion': val_decersion
        })
        df_decersion['Desercion'] = round((df_decersion['Val_Desercion'] / df_decersion['Val_Cliente']) * 100,1)
            
        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=df_decersion['Semana'],
                y=df_decersion['Val_Cliente'],
                name="Clientes",
                yaxis="y1"
                #marker_color="blue"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df_decersion['Semana'],
                y=df_decersion['Desercion'],
                mode="lines+markers",
                name="% Desercion",
                yaxis="y2"
            )
        )

        fig.update_layout(
            title="Clientes vs. % Deserción por Semana",
            xaxis=dict(title="Semana"),
            yaxis=dict(
                title="Clientes",
            ),
            yaxis2=dict(
                title="% Deserción",
                overlaying="y",
                side="right",
                tickformat=".0f",
            ),
            legend=dict(x=0.1, y=1.1, orientation="h"),
        )

        st.plotly_chart(fig)
        
        # Seguimiento de rendimiento del modelo.
        
        reentrenamientos = list(range(1, 5))
        accuracy = np.random.uniform(0.9, 1.0, size=4)
        f1_score = np.random.uniform(0.88, 0.98, size=4)
        precision = np.random.uniform(0.85, 0.97, size=4)
        recall = np.random.uniform(0.87, 0.99, size=4)
        # Crear la figura
        fig = go.Figure()

        # Agregar líneas suavizadas para cada métrica
        fig.add_trace(go.Scatter(x=reentrenamientos, y=accuracy, mode='lines+markers', name="Accuracy", line=dict(color="blue")))
        fig.add_trace(go.Scatter(x=reentrenamientos, y=f1_score, mode='lines+markers', name="F1-Score", line=dict(color="purple")))
        fig.add_trace(go.Scatter(x=reentrenamientos, y=precision, mode='lines+markers', name="Precision", line=dict(color="orange")))
        fig.add_trace(go.Scatter(x=reentrenamientos, y=recall, mode='lines+markers', name="Recall", line=dict(color="green")))

        st.subheader('Seguimiento del modelo')
        st.write('''El seguimiento se realiza de manera mensual con un 
                re entrenamiento con los nuevos clientes ingresados''')
        # Configurar el diseño del gráfico
        fig.update_layout(
            title="Métricas del modelo",
            xaxis=dict(title="Mes"),
            yaxis=dict(title="Valor Métrica"),
            shapes=[
                # Línea horizontal para el umbral de 95% de accuracy
                dict(
                    type="line",
                    xref="paper", x0=0, x1=1,
                    yref="y", y0=0.95, y1=0.95,
                    line=dict(color="red", width=2, dash="dash"),
                )
            ]
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader('Realizar predicción de un cliente')
        with st.form("client_form"):
            st.write("Formulario de Registro de Cliente para Predicción")

            # Crear dos columnas para el formulario
            col1, col2 = st.columns(2)

            with col1:
                customer_age = st.number_input("Customer_Age", value = 46,min_value=18, max_value=100)
                gender = st.selectbox("Gender", ["M", "F"])
                dependent_count = st.number_input("Dependent_count", value = 3, min_value=0, max_value=10)
                education_level = st.selectbox("Education_Level", ["Uneducated", "High School", "College", "Graduate", "Post-Graduate", "Doctorate"])
                marital_status = st.selectbox("Marital_Status", ["Married", "Single", "Divorced", "Unknown"])
                income_category = st.selectbox("Income_Category", ["< $40K", "$40K - $60K", "$60K - $80K", "$80K - $120K", "> $120K"])
                card_category = st.selectbox("Card_Category", ["Blue", "Silver", "Gold", "Platinum"])
                months_on_book = st.number_input("Months_on_book",value = 36, min_value=1)
                total_relationship_count = st.number_input("Total_Relationship_Count", value = 4 ,min_value=1)

            with col2:
                months_inactive_12_mon = st.number_input("Months_Inactive_12_mon", value = 2,min_value=0, max_value=12)
                contacts_count_12_mon = st.number_input("Contacts_Count_12_mon", value = 2, min_value=0, max_value=12)
                credit_limit = st.number_input("Credit_Limit", value = 9000.0, min_value=0.0)
                total_resolving_bal = st.number_input("Total_Resolving_Bal",value = 1100.0, min_value=0.0)
                avg_open_to_buy = st.number_input("Avg_Open_To_Buy", value = 7000.0, min_value=0.0)
                total_amt_chng_q4_q1 = st.number_input("Total_Amt_Chng_Q4_Q1",value = 0.7, min_value=0.0)
                total_trans_amt = st.number_input("Total_Trans_Amt", value = 4000.0, min_value=0.0)
                total_trans_ct = st.number_input("Total_Trans_Ct", value = 65, min_value=0)
                total_ct_chng_q4_q1 = st.number_input("Total_Ct_Chng_Q4_Q1", value = 0.7, min_value=0.0)
                avg_utilization_ratio = st.number_input("Avg_Utilization_Ratio", value = 0.3, min_value=0.0, max_value=1.0)

            submitted = st.form_submit_button("Predicción")
        
        if submitted:
            # Crear la estructura JSON para la solicitud a la API
            myreq = {
                "inputs": [
                    {
                        "Customer_Age": customer_age,
                        "Gender": gender,
                        "Dependent_count": dependent_count,
                        "Education_Level": education_level,
                        "Marital_Status": marital_status,
                        "Income_Category": income_category,
                        "Card_Category": card_category,
                        "Months_on_book": months_on_book,
                        "Total_Relationship_Count": total_relationship_count,
                        "Months_Inactive_12_mon": months_inactive_12_mon,
                        "Contacts_Count_12_mon": contacts_count_12_mon,
                        "Credit_Limit": credit_limit,
                        "Total_Revolving_Bal": total_resolving_bal,
                        "Avg_Open_To_Buy": avg_open_to_buy,
                        "Total_Amt_Chng_Q4_Q1": total_amt_chng_q4_q1,
                        "Total_Trans_Amt": total_trans_amt,
                        "Total_Trans_Ct": total_trans_ct,
                        "Total_Ct_Chng_Q4_Q1": total_ct_chng_q4_q1,
                        "Avg_Utilization_Ratio": avg_utilization_ratio
                    }
                ]
            }

            headers = {"Content-Type": "application/json", "accept": "application/json"}

            # Realizar la solicitud POST a la API
            response = requests.post(api_url, data=json.dumps(myreq), headers=headers)

            if response.status_code == 200:
                data = response.json()
                st.success("¡Predicción realizada con éxito!")
                result = "ALTO riesgo de abandono" if round(data["predictions"][0]) == 1 else "BAJO riesgo de abandono"
                st.write(f"Resultado: {result}")
            else:
                st.error(f"Error al realizar la predicción. Código de estado: {response.status_code}")
                st.write(f"Mensaje: {response.text}")

    

pg = st.navigation([st.Page(page_EDA, 
                            title='Análisis Descriptivo',
                            icon='🧮'), 
                    st.Page(page_Prediction, 
                            title = 'Modelo de predicción',
                            icon = '🤖')])
pg.run()


