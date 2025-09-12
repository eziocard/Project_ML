from sklearn.linear_model import SGDClassifier
from BaseModel import BaseModel

class ModeloSGD(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = SGDClassifier(loss='log_loss', random_state=42)
        print("Modelo SGD inicializado.")

    def train(self):
        self.model.fit(self.X_train_scaled, self.y_train)
        print("¡Modelo entrenado!")