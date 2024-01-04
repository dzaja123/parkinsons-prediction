import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
from sklearn.ensemble import RandomForestRegressor
from ucimlrepo import fetch_ucirepo


# Function to fetch dataset and return features (X) and targets (y)
def fetch_parkinsons_data():
    parkinsons_telemonitoring = fetch_ucirepo(id=189)
    X = parkinsons_telemonitoring.data.features
    y = parkinsons_telemonitoring.data.targets
    return X, y

# Function to display correlation matrix heatmap
def plot_correlation_matrix(X):
    correlation_matrix = X.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix")
    plt.show()

# Function to display correlation with target variables heatmap
def plot_correlation_with_targets(X, y):
    correlation_with_targets = pd.concat([X, y], axis=1).corr()[["motor_UPDRS", "total_UPDRS"]]
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_with_targets, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation with Target Variables")
    plt.show()

# Function to display histogram for Shimmer
def plot_distribution(X):
    plt.figure(figsize=(8, 6))
    sns.histplot(X, kde=True, stat="density")
    plt.title(f"Distribution of {X.name}")
    plt.show()

# Function to display scatter plots for selected features against motor_UPDRS
def plot_scatter_plots(X, y, features):
    for feature in features:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=X[feature], y=y["motor_UPDRS"], alpha=0.7)
        plt.title(f"Scatter Plot: {feature} vs. motor_UPDRS")
        plt.xlabel(feature)
        plt.ylabel("motor_UPDRS")
        plt.show()

# Function to display histograms for motor_UPDRS and total_UPDRS
def plot_target_variables_distribution(y):
    plt.figure(figsize=(12, 8))
    sns.histplot(y["motor_UPDRS"], kde=True, stat="density", color="blue", label="motor_UPDRS")
    sns.histplot(y["total_UPDRS"], kde=True, stat="density", color="orange", label="total_UPDRS")
    plt.title("Distribution of Target Variables")
    plt.legend()
    plt.show()

# Function to display pairplot for selected features and motor_UPDRS
def plot_pairplot_selected_features(data, selected_features):
    sns.pairplot(data[selected_features + ["motor_UPDRS"]], height=2)
    plt.suptitle("Pairplot of Selected Features and motor_UPDRS", y=1.02)
    plt.show()

# Function to display training and validation loss 
def plot_training_epochs(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Model Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

# Function to display scatter plot for actual vs predicted values
def plot_actual_vs_predicted(y_test, prediction, model_name):
    plt.figure(figsize=(10, 8))

    plt.scatter(y_test, y_test, color="blue", label="Actual", alpha=0.7)
    plt.scatter(y_test, prediction, color="red", label="Predicted", alpha=0.7)

    plt.title(f"Actual vs Predicted Values ({model_name})")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.legend()
    plt.show()

# Function to perform normality test for Shimmer
def normality_test_for_shimmer(X):
    normality_test_result = stats.normaltest(X.Shimmer)
    print(f"Normality Test Result for Shimmer: {normality_test_result}")

# Function to preprocess data and split into train and test sets
def preprocess_data(X, y):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(scaled_data, y["motor_UPDRS"].to_numpy(), shuffle=True, test_size=0.3)
    return X_train, X_test, y_train, y_test

# Function to create custom regression model
def create_custom_regression_model(X_train):
    tf.keras.backend.clear_session()

    input_load = tf.keras.layers.Input(shape=(X_train.shape[1],))
    x1 = tf.keras.layers.Dense(units=50, activation="relu")(input_load)
    x2 = tf.keras.layers.Dense(units=32, activation="relu")(x1)
    x3 = tf.keras.layers.Dropout(rate=0.25)(x2)
    x4 = tf.keras.layers.Dense(units=25, activation="relu")(x3)
    x5 = tf.keras.layers.Dropout(rate=0.25)(x4)
    x6 = tf.keras.layers.Dense(units=17, activation="sigmoid")(x5)
    x7 = tf.keras.layers.Dense(units=13, activation="tanh")(x6)
    x8 = tf.keras.layers.Dense(units=1, activation="exponential")(x7)

    model = tf.keras.Model(input_load, x8)
    return model

# Function to train custom regression model 
def train_custom_regression_model(model, X_train, y_train, X_test, y_test):
    optimizer = tf.keras.optimizers.Adam()
    model.compile(optimizer=optimizer, loss="mse")
    history = model.fit(X_train, y_train, epochs=200, batch_size=60, validation_data=(X_test, y_test), shuffle=True)
    return model, history

# Function to evaluate custom regression model
def evaluate_custom_regression_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    explained_var = explained_variance_score(y_test, y_pred)
    mean_abs_err = mean_absolute_error(y_test, y_pred)
    mean_sq_err = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return explained_var, mean_abs_err, mean_sq_err, r2

# Function to train and evaluate random forest regression model
def train_and_evaluate_random_forest(X_train, y_train, X_test, y_test):
    regressor = RandomForestRegressor(n_estimators=10, random_state=0)
    regressor.fit(X_train, y_train)

    y_pred = regressor.predict(X_test)

    explained_var = explained_variance_score(y_test, y_pred)
    mean_abs_err = mean_absolute_error(y_test, y_pred)
    mean_sq_err = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return regressor, explained_var, mean_abs_err, mean_sq_err, r2

def main():
    # Fetch dataset
    X, y = fetch_parkinsons_data()

    # Display correlation matrix heatmap
    plot_correlation_matrix(X)

    # Concatenate features and target variable
    data = pd.concat([X, y], axis=1)

    # Display correlation with target variables heatmap
    plot_correlation_with_targets(X, y)

    # Display histogram for Shimmer, Jitter(%) and NHR
    plot_distribution(X["Shimmer"])
    plot_distribution(X["Jitter(%)"])
    plot_distribution(X["NHR"])

    # Perform normality test for Shimmer
    normality_test_for_shimmer(X)

    # Display scatter plots for selected features against motor_UPDRS
    features_list = ["Jitter(%)", "Jitter(Abs)", "Jitter:RAP", "Jitter:PPQ5", "Jitter:DDP",
                     "Shimmer", "Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5", "Shimmer:APQ11",
                     "Shimmer:DDA", "NHR", "HNR", "RPDE", "DFA", "PPE"]
    plot_scatter_plots(X, y, features_list)

    # Display histograms for motor_UPDRS and total_UPDRS
    plot_target_variables_distribution(y)

    # Display pairplot for selected features and motor_UPDRS
    selected_features = ["Jitter(%)", "Shimmer", "NHR", "HNR"]
    plot_pairplot_selected_features(data, selected_features)

    X_train, X_test, y_train, y_test = preprocess_data(X, y)

    # Custom Regression Model
    custom_regression_model = create_custom_regression_model(X_train=X_train)
    trained_custom_regression_model, history_custom_regression = train_custom_regression_model(custom_regression_model, X_train, y_train, X_test, y_test)

    # Plot training epochs for the Custom Regression Model
    plot_training_epochs(history_custom_regression)

    # Evaluate custom regression model
    explained_var_rf, mean_abs_err_rf, mean_sq_err_rf, r2_rf = evaluate_custom_regression_model(trained_custom_regression_model, X_test, y_test)

    # Plot Actual vs Predicted for Custom Regression Model
    prediction_custom_regression = trained_custom_regression_model.predict(X_test)
    plot_actual_vs_predicted(y_test, prediction_custom_regression, "Custom Regression Model")
    
    print()
    print(f"Explained Variance for Custom Regression: {explained_var_rf}")
    print(f"Mean Absolute Error for Custom Regression: {mean_abs_err_rf}")
    print(f"Mean Squared Error for Custom Regression: {mean_sq_err_rf}")
    print(f"R^2 Score for Custom Regression: {r2_rf}")
    print()

    # Random Forest Model
    regressor, explained_var_rf, mean_abs_err_rf, mean_sq_err_rf, r2_rf = train_and_evaluate_random_forest(X_train, y_train, X_test, y_test)

    # Plot Actual vs Predicted for Random Forest Model
    y_pred_rf = regressor.predict(X_test)
    plot_actual_vs_predicted(y_test, y_pred_rf, "Random Forest Model")

    print(f"Explained Variance for Random Forest: {explained_var_rf}")
    print(f"Mean Absolute Error for Random Forest: {mean_abs_err_rf}")
    print(f"Mean Squared Error for Random Forest: {mean_sq_err_rf}")
    print(f"R^2 Score for Random Forest: {r2_rf}")
    print()


if __name__ == "__main__":
    main()
