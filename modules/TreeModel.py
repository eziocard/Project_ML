from sklearn.tree import DecisionTreeClassifier
from sklearn import tree as sktree
from modules.BaseModel import BaseModel
import matplotlib.pyplot as plt


class TreeModel(BaseModel):

    # INFO: Los arboles de decisión son invariantes a escala, solo minimizan Gini o Entropy
    def scale(self, choice="standard"):
        pass


    def train(self):
        # INFO: scikit-learn utiliza una versión optimizada del algoritmo CART
        # INFO: También se regulariza utilizando Random Forest
        self.model = DecisionTreeClassifier(
            criterion="gini",   # impureza: 'gini' o 'entropy'
            max_depth=3,        # profundidad máxima (None = sin límite)
            min_samples_split=2,# mínimo para dividir un nodo interno
            min_samples_leaf=1, # mínimo por hoja
            max_features=2,
            random_state=42
        )

        self.model.fit(self.X_train, self.y_train)


    def predict(self):
        self.y_pred = self.model.predict_proba(self.X_test.iloc[[0]])
        self.diabetic = True if self.y_pred[0][1] > 0.75 else False
        print(f"{self.y_pred=}")
        print(f"{self.diabetic=}")


    def plot(self):

        plt.figure(figsize=(12,6))
        sktree.plot_tree(
            self.model,
            feature_names=['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
                           'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
                           'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
                           'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education',
                           'Income'],
            class_names=["No Diabetes","Diabetes"],
            filled=True, rounded=True, fontsize=10
        )
        plt.title("Árbol de decisión (criterio: Gini, max_depth=3)")
        plt.show()

