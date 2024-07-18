from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

def getpredict(number_of_days):
    price=number_of_days+1
    return price

# Define the Pydantic model
class GoldInput(BaseModel):
    number_of_days: int

# Define the POST endpoint
@app.post("/goldprice/")
def create_item(param: GoldInput):
    param_dict = param.dict()
    param_dict["goldprice"] = getpredict(param_dict["number_of_days"])
    return param_dict

#root
@app.get("/")
def root():
    return {'greeting': 'Hello'}
