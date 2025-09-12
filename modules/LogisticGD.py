from sklearn.linear_model import LogisticRegression
from BaseModel import BaseModel

class ModeloLogistico(BaseModel):
    def __init__(self, file):
        super().__init__(file)
        self.model = LogisticRegression(random_state=42)
        print("Modelo Logístico Estándar inicializado.")
        
    def scale(self, choice="standard"):
        super().scale(choice)
        self.X_train_scaled = self.scaler.transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        print("¡Datos escalados!")
       
        

    def train(self):
        self.model.fit(self.X_train_scaled, self.y_train)
        print("¡Modelo entrenado!")
        
        