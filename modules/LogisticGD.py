from sklearn.linear_model import LogisticRegression
from BaseModel import BaseModel
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, accuracy_score

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
        
    def predict(self,usuario_data):
        usuario_data_scaled = self.scaler.transform(usuario_data)

        prediction = self.model.predict(usuario_data_scaled)
        probabilities = self.model.predict_proba(usuario_data_scaled)
        
        return prediction, probabilities
    
    def metrics(self):

        y_hat = self.model.predict(self.X_test_scaled)
        print(f"Accuracy: {accuracy_score(self.y_test, y_hat):.3f}")
        print(f"Precision: {precision_score(self.y_test, y_hat):.3f}")
        print(f"Recall:    {recall_score(self.y_test, y_hat):.3f}")
        print(f"F1:        {f1_score(self.y_test, y_hat):.3f}")

        cm = confusion_matrix(self.y_test, y_hat)
        ConfusionMatrixDisplay(cm).plot(cmap="Blues")

        plt.title("Matriz de confusión (SVM RBF)")
        plt.show()
        
        