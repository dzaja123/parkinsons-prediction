from visualization.visualization import (plot_correlation_matrix, plot_distribution, 
                                        plot_correlation_with_targets, plot_target_variables_distribution, 
                                        plot_training_epochs, plot_scatter_plots, plot_actual_vs_predicted, 
                                        plot_pairplot_selected_features, print_evaluation_metrics)

from models.models import (train_and_evaluate_random_forest, train_and_evaluate_decision_tree, 
                           train_and_evaluate_svm, train_custom_regression_model,
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
    features_list = ["Jitter(%)", "Jitter(Abs)", "Jitter:DDP",
                     "Shimmer", "Shimmer(dB)", "Shimmer:APQ3",
                     "NHR", "HNR", "RPDE", "DFA", "PPE"]
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
    
    # Print evaluation metrics for Custom Regression Model
    print_evaluation_metrics("Custom Regression Model", explained_var_rf, mean_abs_err_rf, mean_sq_err_rf, r2_rf)

    # Random Forest Model
    rf_regressor, explained_var_rf, mean_abs_err_rf, mean_sq_err_rf, r2_rf = train_and_evaluate_random_forest(X_train, y_train, X_test, y_test)

    # Plot Actual vs Predicted for Random Forest Model
    y_pred_rf = rf_regressor.predict(X_test)
    plot_actual_vs_predicted(y_test, y_pred_rf, "Random Forest Model")

    # Print evaluation metrics for Random Forest Model
    print_evaluation_metrics("Random Forest Model", explained_var_rf, mean_abs_err_rf, mean_sq_err_rf, r2_rf)

    # Train and evaluate Decision Tree Model
    dt_regressor, explained_var_dt, mean_abs_err_dt, mean_sq_err_dt, r2_dt = train_and_evaluate_decision_tree(X_train, y_train, X_test, y_test)

    # Plot Actual vs Predicted for Decision Tree Model
    y_pred_dt = dt_regressor.predict(X_test)
    plot_actual_vs_predicted(y_test, y_pred_dt, "Decision Tree Model")

    # Print evaluation metrics for Decision Tree Model
    print_evaluation_metrics("Decision Tree Model", explained_var_dt, mean_abs_err_dt, mean_sq_err_dt, r2_dt)

    # Train and evaluate SVM Model
    svm_regressor, explained_var_svm, mean_abs_err_svm, mean_sq_err_svm, r2_svm = train_and_evaluate_svm(X_train, y_train, X_test, y_test)

    # Plot Actual vs Predicted for SVM Model
    y_pred_svm = svm_regressor.predict(X_test)
    plot_actual_vs_predicted(y_test, y_pred_svm, "SVM Model")

    # Print evaluation metrics for SVM Model
    print_evaluation_metrics("SVM Model", explained_var_svm, mean_abs_err_svm, mean_sq_err_svm, r2_svm)

if __name__ == "__main__":
    main()
