from BaseModel import BaseModel
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt

from sklearn.svm import SVC

class SVMclass(BaseModel):
    def __init__(self, file):
        super().__init__(file)
        self.model = None 

    def train(self,kernel='rbf', C=1.0, gamma='scale', random_state=42):
        match kernel:
            case "rbf":
                self.X_train_scaled = self.scaler.transform(self.X_train)

                self.model = SVC(kernel=kernel,C=C,gamma=gamma,random_state=random_state
                )
                self.model.fit(self.X_train_scaled, self.y_train)
            #me falta el linear y el poly

            case "linear":
                self.X_train_scaled = self.scaler.transform(self.X_train)
                self.model = SVC(kernel=kernel,C=C,random_state=random_state
                )
    def predict(self,usuario_data = None): #aqui es donde se pasan los valores escalados, recuerda programar un case para predecir valores por el usuario
        usuario_data_scaled = self.scaler.transform(usuario_data)

        prediction = self.model.predict(usuario_data_scaled)
        probabilities = None
        
        return prediction, probabilities
     
    def metrics(self):
        self.X_test_scaled = self.scaler.transform(self.X_test)
        y_hat = self.model.predict(self.X_test_scaled)
        print(f"Accuracy: {accuracy_score(self.y_test, y_hat):.3f}")
        print(f"Precision: {precision_score(self.y_test, y_hat):.3f}")
        print(f"Recall:    {recall_score(self.y_test, y_hat):.3f}")
        print(f"F1:        {f1_score(self.y_test, y_hat):.3f}")

        cm = confusion_matrix(self.y_test, y_hat)
        ConfusionMatrixDisplay(cm).plot(cmap="Blues")

        plt.title("Matriz de confusión (SVM RBF)")
        plt.show()


