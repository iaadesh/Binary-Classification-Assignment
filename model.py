import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from prettytable import PrettyTable


train = pd.read_csv("training_set.csv")
test = pd.read_csv("test_set.csv")

train = train.drop(["Unnamed: 0"], axis = 1)
test = test.drop(["Unnamed: 0"], axis = 1)

X_train, X_valid, y_train, y_valid  = train_test_split(train.drop(["Y"], axis = 1), 
                                                       train["Y"].values, test_size = 0.2, 
                                                       random_state = 15, stratify = train["Y"].values)

X_test = test.copy()

# feature which do not contribute much to the training model
unimportant_features = ['X32', 'X34', 'X57', 'X56', 'X55', 'X54', 'X18', 'X28', 
                        'X29', 'X15', 'X30', 'X31', 'X33', 'X35','X38', 'X39', 
                        'X40', 'X43', 'X6', 'X47', 'X51', 'X3', 'X2', 'X22', 'X20']

X_train = X_train.drop(unimportant_features, axis = 1)
X_valid = X_valid.drop(unimportant_features, axis = 1)
X_test = X_test.drop(unimportant_features, axis = 1)

# normalizing the dataset
scaler = MinMaxScaler()
scaler.fit(X_train.values)

X_train_normalize = scaler.transform(X_train.values)
X_valid_normalize = scaler.transform(X_valid.values)
X_test_normalize = scaler.transform(X_test.values)

# training best model with best hyperparameter
xgb_classifier = XGBClassifier(n_estimators = 25, max_depth = 15, min_samples_split = 1, random_state = 15)
xgb_classifier.fit(X_train_normalize, y_train)

train_acc = xgb_classifier.score(X_train_normalize, y_train)
valid_acc = xgb_classifier.score(X_valid_normalize, y_valid)
train_auc = roc_auc_score(y_train, xgb_classifier.predict(X_train_normalize))
valid_auc = roc_auc_score(y_valid, xgb_classifier.predict(X_valid_normalize))


myTable = PrettyTable(["Best Model", "Train Accuracy", "Valid Accuracy", "Train AUC", "Validation AUC"])
myTable.add_row(["XGBoost", train_acc, valid_acc, train_auc, valid_auc])

print("In XGBoost (the Best Classification Model for our dataset): ")
print(myTable)