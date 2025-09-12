import numpy as np
from sklearn.linear_model import SGDClassifier
from BaseModel import BaseModel

class ModeloMiniBatch(BaseModel):
    def __init__(self, n_epochs=10, batch_size=128):
        super().__init__()
        self.model = SGDClassifier(loss='log_loss', random_state=42)
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        print("Modelo Mini-Batch inicializado.")

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