import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

import PreProcessing as pre

data_train = pd.read_csv('cars-train.csv')
data_test = pd.read_csv('cars-test.csv')

####################### Train data-Set ###################################

# Split car-info column
data_train = pre.split(data_train)
data_train.drop(['car_id'], axis=1, inplace=True)
# Change type of column ot datetime
pre.datetime(data_train, 'date')

# Convert  column to lowercase
pre.LowerCase(data_train, 'fuel_type')

# Fill NAN volume(cm3) colum with median
pre.Fill_median(data_train, 'volume(cm3)')

data_train['segment'].isnull().value_counts().plot.bar()
data_train['drive_unit'].isnull().value_counts().plot.bar()

# Fill NAN values in [segment , drive_unit] columns
cols = ['segment', 'drive_unit']
pre.Fill_Cate(data_train, cols)
# Check outliers
# for col in data_train.columns:
#     fig = px.histogram(data_train, x=col)
#     fig.show()
# Applying label Encoding
Cate_cols = list(data_train.select_dtypes('object'))

data_train = pre.Feature_Encoder(data_train, Cate_cols)

# # Create a Gaussian Classifier
# clf = RandomForestClassifier(n_estimators=100, max_depth=15)
#
# # Train the model using the training sets y_pred=clf.predict(X_test)
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# # Model Accuracy, how often is the classifier correct?
# print("Accuracy:", metrics.accuracy_score(y_test, y_pred))


####################### Test data-Set ###################################

# Split car-info column
data_test = pre.split(data_test)

# Change type of column ot datetime
pre.datetime(data_test, 'date')

# Convert  column to lowercase
pre.LowerCase(data_test, 'fuel_type')

# Fill NAN volume(cm3) colum with median
pre.Fill_median(data_test, 'volume(cm3)')

# Fill NAN values in [segment , drive_unit] columns
cols = ['segment', 'drive_unit']
pre.Fill_Cate(data_test, cols)
# Check outliers
# for col in data_test.columns:
#     fig = px.histogram(data_test, x=col)
#     fig.show()

# Applying label Encoding
Cate_cols = list(data_test.select_dtypes('object'))

data_test = pre.Feature_Encoder(data_test, Cate_cols)

# Correlation between columns
Num_cols = list(data_train.select_dtypes('int', 'float'))
X = pre.Corr(data_train, 'Price Category')  # Selected Features
Y = data_train['Price Category']  # Target
# Split dataset into training set and test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=10)

clf = RandomForestClassifier(n_estimators=100, criterion="entropy")
clf.fit(X_train, Y_train)
train_pred1 = clf.predict(X_train)
X_predict = data_test[['condition', 'fuel_type', 'transmission', 'segment']]
test_pred2 = clf.predict(X_predict)

print("train error /\ Root_mean_squared_error :{}".format(mean_squared_error(Y_train, train_pred1, squared=False)))
print("train error /\ r2_score :{}".format(r2_score(Y_train, train_pred1)))

print("test error /\ Root_mean_squared_error :{}".format(mean_squared_error(Y_test, test_pred2)))
print("test error /\ r2_score :{}".format(r2_score(Y_test, test_pred2)))

submission = pd.DataFrame()
submission = data_test[['car_id']]
submission['Price Category'] = test_pred2
submission.to_csv("sample_submission_random.csv", index=None)
