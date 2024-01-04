import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


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