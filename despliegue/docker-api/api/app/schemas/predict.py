from typing import Any, List, Optional

from pydantic import BaseModel
#from model.processing.validation import DataInputSchema

class DataInputSchema(BaseModel):
    Customer_Age: int
    Gender: str
    Dependent_count: int
    Education_Level: str
    Marital_Status: str
    Income_Category: str
    Card_Category: str
    Months_on_book: int
    Total_Relationship_Count: int
    Months_Inactive_12_mon: int
    Contacts_Count_12_mon: int
    Credit_Limit: float
    Total_Revolving_Bal: float
    Avg_Open_To_Buy: float
    Total_Amt_Chng_Q4_Q1: float
    Total_Trans_Amt: int
    Total_Trans_Ct: int
    Total_Ct_Chng_Q4_Q1: float
    Avg_Utilization_Ratio: float



# Esquema de los resultados de predicción
class PredictionResults(BaseModel):
    predictions: List[int]

# Esquema para inputs múltiples
class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]

    class Config:
        schema_extra = {
            "example": {
                "inputs": [
                    {
                        "Customer_Age": 57,
                        "Gender": "M",
                        "Dependent_count": 4,
                        "Education_Level": "Graduate",
                        "Marital_Status": "Single",
                        "Income_Category": "$120K +",
                        "Card_Category": "Blue",
                        "Months_on_book":52,
                        "Total_Relationship_Count":2,
                        "Months_Inactive_12_mon":3,
                        "Contacts_Count_12_mon":2,
                        "Credit_Limit":25808,
                        "Total_Revolving_Bal":0,
                        "Avg_Open_To_Buy":25808,
                        "Total_Amt_Chng_Q4_Q1":0.712,
                        "Total_Trans_Amt":7794,
                        "Total_Trans_Ct":94,
                        "Total_Ct_Chng_Q4_Q1":0.843,
                        "Avg_Utilization_Ratio": 0
                    }
                ]
            }
        }