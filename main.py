from visualization.visualization import (plot_correlation_matrix, plot_distribution, 
                                        plot_correlation_with_targets, plot_target_variables_distribution, 
                                        plot_training_epochs, plot_scatter_plots, plot_actual_vs_predicted, 
                                        plot_pairplot_selected_features)
from models.models import (train_and_evaluate_random_forest, train_custom_regression_model, 
                            evaluate_custom_regression_model, create_custom_regression_model)
from preprocessing.preprocessing import preprocess_data, fetch_parkinsons_data
import pandas as pd


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
