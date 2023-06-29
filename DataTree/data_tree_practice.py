from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from graphviz import Source
import matplotlib as plt
iris = load_iris(as_frame=True)
X_iris = iris.data[["petal-length (cm)", "petal-width (cm)"]].values
y_iris = iris.target

Tree = DecisionTreeClassifier(max_depth=2, random_state=42)
Tree.fit(X_iris, y_iris)

export_graphviz(Tree,
                out_file="iris_tree.dot",
                feature_names=["petal-length (cm)", "petal-width (cm)"],
                class_names=iris.target_names,
                rounded=True,
                filled=True
                )
Source.from_file("iris_tree.dot")