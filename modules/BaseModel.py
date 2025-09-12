import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt


class BaseModel():

    OBJECTIVE = 'Diabetes_binary'

    def __init__(self, file):

        self.df = pd.read_csv(file)

        self._X = self.df[ self.df.columns[self.df.columns!= self.OBJECTIVE] ]
        self._y = self.df[self.OBJECTIVE]


    @property
    def X(self): 
        return self._X

    @property
    def y(self): 
        return self._y


    def split(self,test_size=0.2):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self._X, self._y, test_size=test_size, random_state=42, stratify=self._y
        )

    def scale(self, choice="standard"):
        match choice:
            case "standard":
                self.scaler = StandardScaler().fit(self.X_train) 

            case "minmax":
                self.scaler = MinMaxScaler().fit(self.X_train) 
            case _:
                raise NotDefinedException("Este escalado no existe")

    def train(self):
        raise NotDefinedException("Este método no ha sido implementado")

    def predict(self, usuario):
        usuario_scaled = self.scaler.transform(usuario)
        prediction = self.model.predict(usuario_scaled)
        
        return prediction
    
    
    def metrics(self,model,X):

        y_hat = model.predict(X)
        print(f"Accuracy: {accuracy_score(self.y_test, y_hat):.3f}")
        print(f"Precision: {precision_score(self.y_test, y_hat):.3f}")
        print(f"Recall:    {recall_score(self.y_test, y_hat):.3f}")
        print(f"F1:        {f1_score(self.y_test, y_hat):.3f}")

        cm = confusion_matrix(self.y_test, y_hat)
        ConfusionMatrixDisplay(cm).plot(cmap="Blues")

        plt.title("Matriz de confusión (SVM RBF)")
        plt.show()


