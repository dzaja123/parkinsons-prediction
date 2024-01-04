from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
import tensorflow as tf


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

# Function to train and evaluate given model
def train_and_evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    explained_var = explained_variance_score(y_test, y_pred)
    mean_abs_err = mean_absolute_error(y_test, y_pred)
    mean_sq_err = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return explained_var, mean_abs_err, mean_sq_err, r2

# Function to train and evaluate random forest regression model
def train_and_evaluate_random_forest(X_train, y_train, X_test, y_test):
    rf_regressor = RandomForestRegressor(n_estimators=10, random_state=0)
    explained_var, mean_abs_err, mean_sq_err, r2 = train_and_evaluate_model(rf_regressor, X_train, y_train, X_test, y_test)
    return rf_regressor, explained_var, mean_abs_err, mean_sq_err, r2

# Function to train and evaluate Decision Tree regression model
def train_and_evaluate_decision_tree(X_train, y_train, X_test, y_test):
    dt_regressor = DecisionTreeRegressor(random_state=0)
    explained_var, mean_abs_err, mean_sq_err, r2 = train_and_evaluate_model(dt_regressor, X_train, y_train, X_test, y_test)
    return dt_regressor, explained_var, mean_abs_err, mean_sq_err, r2

# Function to train and evaluate Support Vector Machine (SVM) regression model
def train_and_evaluate_svm(X_train, y_train, X_test, y_test):
    svm_regressor = SVR()
    explained_var, mean_abs_err, mean_sq_err, r2 = train_and_evaluate_model(svm_regressor, X_train, y_train, X_test, y_test)
    return svm_regressor, explained_var, mean_abs_err, mean_sq_err, r2