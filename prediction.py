import pickle 


class prediction: 
    def __init__ (self, model_path) :
        with open (model_path, "rb") as f: 
             self.model = pickle.load(f)
        
    def predict (self, input_data):
        return self.model.predict([input_data]).tolist() 
                
    
if __name__ == "__main__":
    obj = prediction("model.pkl")
    y_predict= obj.predict([20,10,13,14])
    print (type(y_predict))

