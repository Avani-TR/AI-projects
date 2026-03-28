# Step 1: Import libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Step 2: Load dataset
iris = load_iris()
X = iris.data
y = iris.target

print("Dataset loaded successfully!")
print("Total samples:", X.shape[0])
print("Features per sample:", X.shape[1])
print("Classes:", iris.target_names)

# Step 3: Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nData split completed!")
print("Training data:", len(X_train))
print("Testing data:", len(X_test))

# Step 4: Create model
model = KNeighborsClassifier(n_neighbors=5)

# Step 5: Train model
model.fit(X_train, y_train)
print("\nModel training completed!")

# Step 6: Predict
y_pred = model.predict(X_test)

# Step 7: Evaluate
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\nModel Evaluation:")
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", cm)

# Step 8: Sample prediction
sample = [[5.1, 3.5, 1.4, 0.2]]
prediction = model.predict(sample)
print("\nSample Prediction:", iris.target_names[prediction][0])
print("\nEnter your own flower details to predict:")

try:
    sepal_length = float(input("Sepal Length: "))
    sepal_width = float(input("Sepal Width: "))
    petal_length = float(input("Petal Length: "))
    petal_width = float(input("Petal Width: "))

    user_sample = [[sepal_length, sepal_width, petal_length, petal_width]]
    user_prediction = model.predict(user_sample)

    print("Predicted Flower:", iris.target_names[user_prediction][0])

except:
    print("Invalid input! Please enter numeric values.")