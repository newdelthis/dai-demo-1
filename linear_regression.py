import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Step 1: Load data
df = pd.read_csv('salary.csv')
X = df[['YearsExperience']]
y = df['Salary']

# Step 2: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Print test data
print("--- Test Set Data ---")
print("X_test (Experience):\n", X_test)
print("\ny_test (Salary):\n", y_test)
print("---------------------\n")

# Enable auto-logging
mlflow.sklearn.autolog()

# Step 3–8: Run MLflow experiment
with mlflow.start_run():

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)
    new_X_values_df = pd.DataFrame({'YearsExperience': [1, 3, 5, 8, 12, 15, 20, 25]})
    new_y_pred = model.predict(new_X_values_df)
    print('\nPredictions for new experience values (1, 3, 5, 8, 12, 15, 20, 25):\n', new_y_pred)

    # Evaluation
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\nModel Evaluation Metrics:")
    print("Mean Absolute Error (MAE):", mae)
    print("Mean Squared Error (MSE):", mse)
    print("R2 Score:", r2)

    # Manual logging (optional if autolog is used)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2", r2)

    # Log model manually (optional)
    mlflow.sklearn.log_model(model, "salary_model")

    # Save prediction results if needed
    result_df = pd.DataFrame({'YearsExperience': X_test.values.flatten(),
                              'ActualSalary': y_test.values,
                              'PredictedSalary': y_pred})
    result_df.to_csv("results.csv", index=False)
    mlflow.log_artifact("results.csv")

# Step 5: Visualize training results
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, model.predict(X_train), color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Step 6: Visualize test results
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, model.predict(X_train), color='blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# After running the code ...
# On the terminal, type mlflow ui
# then open browser: http://127.0.0.1:5000/

# Save the model to a pkl file
import joblib

# Save the trained model to a file
joblib.dump(model, 'salary_regression_model.pkl')

print("Model saved as salary_regression_model.pkl")

# To load model and make predictions
# Load the model from the file
loaded_model = joblib.load('salary_regression_model.pkl')

# Use it to predict
#sample = pd.DataFrame({'YearsExperience': [6.5]})
#predicted_salary = loaded_model.predict(sample)
#print("Predicted salary:", predicted_salary[0])



