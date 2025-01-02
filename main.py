from fastapi import FastAPI, HTTPException, Depends
from prediction import prediction
from pydantic import BaseModel
from fastapi.security.api_key import APIKeyHeader

# Initialize FastAPI
app = FastAPI()

# Initialize the ML model
model_path = "model.pkl"  # Path to your saved model
model = prediction(model_path)

LABELS = ["Setosa", "Versicolor", "Virginica"]

# Define the input schema using Pydantic
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


API_KEY = "DUK_90"
API_KEY_NAME = "API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)
# Security dependency
def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")

# Example route with API key protection
@app.get("/secure-data", dependencies=[Depends(verify_api_key)])
async def get_secure_data():
    return {"message": "This is secured data"}

@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI Prediction Service"}

@app.post('/predict/')
async def make_prediction(input_data: IrisInput, api_key: str = Depends(verify_api_key)):
    try: 
        # Prepare the input for the model
        input_features = [
            input_data.sepal_length,
            input_data.sepal_width,
            input_data.petal_length,
            input_data.petal_width,
        ]
        # Make a prediction
        prediction = model.predict(input_features)
        return {"input": input_data.dict(), "prediction": LABELS[prediction[0]]}
    except Exception as e: 
        raise HTTPException(status_code=500, detail=str(e))