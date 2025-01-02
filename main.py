from fastapi import FastAPI, HTTPException
from prediction import prediction
from pydantic import BaseModel

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

@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI Prediction Service"}

@app.post('/predict/')
def make_prediction(input_data: IrisInput):
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

