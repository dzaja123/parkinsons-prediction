from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo


# Function to fetch dataset and return features (X) and targets (y)
def fetch_parkinsons_data():
    parkinsons_telemonitoring = fetch_ucirepo(id=189)
    X = parkinsons_telemonitoring.data.features
    y = parkinsons_telemonitoring.data.targets
    return X, y

# Function to preprocess data and split into train and test sets
def preprocess_data(X, y):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(scaled_data, y["motor_UPDRS"].to_numpy(), shuffle=True, test_size=0.3)
    return X_train, X_test, y_train, y_test