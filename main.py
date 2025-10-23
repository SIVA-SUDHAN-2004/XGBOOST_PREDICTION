from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pickle
import os
from datetime import datetime

MODEL_PATH = os.environ.get("MODEL_PATH", "productivity_model.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

class EmployeeFeatures(BaseModel):
    Employee_ID: int
    Department: str
    Gender: str
    Age: int
    Job_Title: str
    Hire_Date: str  # format: YYYY-MM-DD
    Years_At_Company: float
    Education_Level: str
    Monthly_Salary: float
    Work_Hours_Per_Week: float
    Projects_Handled: int
    Overtime_Hours: float
    Sick_Days: int
    Remote_Work_Frequency: float
    Team_Size: int
    Training_Hours: float
    Promotions: int
    Employee_Satisfaction_Score: float
    Resigned: int

app = FastAPI()

# Example mappings (must match training encoding)
department_map = {"Sales": 0, "HR": 1, "IT": 2, "Finance": 3, "Operations": 4}
gender_map = {"M": 0, "F": 1}
education_map = {"High School": 0, "Bachelor": 1, "Master": 2, "PhD": 3}
job_title_map = {"Sales Executive": 0, "Manager": 1, "Engineer": 2, "Analyst": 3}  # add more

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(features: EmployeeFeatures):
    try:
        # Convert categorical fields to numeric
        dept = department_map.get(features.Department, -1)
        gender = gender_map.get(features.Gender, -1)
        edu = education_map.get(features.Education_Level, -1)
        job = job_title_map.get(features.Job_Title, -1)
        
        # Convert Hire_Date to years since hire
        hire_date_obj = datetime.strptime(features.Hire_Date, "%Y-%m-%d")
        years_since_hire = (datetime.now() - hire_date_obj).days / 365.25
        
        # Build input array
        arr = np.array([[
            features.Employee_ID,
            dept,
            gender,
            features.Age,
            job,
            years_since_hire,
            features.Years_At_Company,
            edu,
            features.Monthly_Salary,
            features.Work_Hours_Per_Week,
            features.Projects_Handled,
            features.Overtime_Hours,
            features.Sick_Days,
            features.Remote_Work_Frequency,
            features.Team_Size,
            features.Training_Hours,
            features.Promotions,
            features.Employee_Satisfaction_Score,
            features.Resigned
        ]], dtype=float)

        # Make prediction
        prediction = model.predict(arr)[0]
        return {"productivity_score": float(prediction)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
