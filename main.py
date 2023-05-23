import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import dill


app = FastAPI()
with open('model/models/logreg_v1.pkl', 'rb') as file:
    model = dill.load(file)


class Form(BaseModel):
    session_id: str
    client_id: str
    visit_date: str
    visit_time: str
    visit_number: int
    utm_source: str
    utm_medium: str
    utm_campaign: str
    utm_adcontent: str
    utm_keyword: str
    device_category: str
    device_os: Optional[str] = None
    device_brand: str
    device_model: str
    device_screen_resolution: str
    device_browser: str
    geo_country: str
    geo_city: str




class Prediction(BaseModel):
    session_id: str
    purpose: int


@app.get('/status')
def status():
    return "I'm OK"


@app.get('/version')
def get_version():
    return model['metadata']


@app.post('/predict')
def predict(form: Form):

    df = pd.DataFrame.from_dict([form.dict()])
    y = model['model'].predict(df)

    return {
        'session_id': form.session_id,
        'purpose': int(y[0])
    }



