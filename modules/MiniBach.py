import numpy as np
from sklearn.linear_model import SGDClassifier
from BaseModel import BaseModel
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, accuracy_score

class ModeloMiniBatch(BaseModel):
    def __init__(self,file, n_epochs=10, batch_size=128):
        super().__init__(file)
        self.model = SGDClassifier(loss='log_loss', random_state=42)
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        print("Modelo Mini-Batch inicializado.")
    
    def scale(self, choice="standard"):
        super().scale(choice)
        self.X_train_scaled = self.scaler.transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        print("¡Datos escalados!")

    def train(self):
        y_train_np = self.y_train.to_numpy()
        n_batches = int(np.ceil(len(self.X_train_scaled) / self.batch_size))

        for epoch in range(self.n_epochs):
            indices = np.random.permutation(len(self.X_train_scaled))
            X_train_shuffled = self.X_train_scaled[indices]
            y_train_shuffled = y_train_np[indices]
            
            for i in range(n_batches):
                start = i * self.batch_size
                end = start + self.batch_size
                self.model.partial_fit(
                    X_train_shuffled[start:end], 
                    y_train_shuffled[start:end], 
                    classes=np.array([0.0, 1.0])
                )
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