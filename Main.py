import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier

import PreProcessing as pre

data_train = pd.read_csv('cars-train.csv')
data_test = pd.read_csv('cars-test.csv')

####################### Train data-Set ###################################

# Split car-info column
data_train = pre.split(data_train)
data_train.drop(['car_id'], axis=1, inplace=True)

# Convert  column to lowercase
pre.LowerCase(data_train, 'fuel_type')

# Fill NAN volume(cm3) colum with median
pre.Fill_median(data_train, 'volume(cm3)')

# Fill NAN values in [segment , drive_unit] columns
cols = ['segment', 'drive_unit']
pre.Fill_Cate(data_train, cols)
# Check outliers
# for col in data_train.columns:
#     fig = px.histogram(data_train, x=col)
#     fig.show()
# Applying label Encoding
Cate_cols = list(data_train.select_dtypes('object'))
# data_train['Price Category'] = data_train['Price Category'] \
#     .replace(['cheap', 'moderate', 'expensive', 'very expensive'], [0, 1, 2, 3])
data_train = pre.Feature_Encoder(data_train, Cate_cols)
####################### Test data-Set ###################################

# Split car-info column
data_test = pre.split(data_test)
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
# pre.Corr(data_train, 'Price Category')  # Selected Features
X = data_train.iloc[:, :-1]  # Selected Features
Y = data_train['Price Category']  # Target

# Split dataset into training set and test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=10)
print(Y_train.isnull().sum())
# # Applying PCA
# pca = PCA(random_state=2022)
# pca_6 = pca.fit_transform(X_train, Y_train)

# # ################## RandomForestClassifier #######################
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, Y_train)
train_pred1 = clf.predict(X_train)

X_predict = data_test[
    ['Model', 'company', 'date', 'condition', 'mileage(kilometers)',
     'fuel_type', 'volume(cm3)', 'color', 'transmission', 'drive_unit', 'segment']]
test_pred2 = clf.predict(X_predict)

print('Accuracy of Random classifier on training set: {:.2f}'
      .format(clf.score(X_train, Y_train)))
print('Accuracy of Random classifier on test set: {:.2f}'
      .format(clf.score(X_test, Y_test)))
###################################################################

################# Gaussian Naive Bayes classifier #################
# instantiate the model
gnb = GaussianNB()
# fit the model
gnb.fit(X_train, Y_train)
train_pred_gnb = gnb.predict(X_train)
X_predict = data_test [['Model', 'company', 'date', 'condition', 'mileage(kilometers)',
     'fuel_type', 'volume(cm3)', 'color', 'transmission', 'drive_unit', 'segment']]
test_pred2_gnb = gnb.predict(X_predict)

print('Accuracy of  Gaussian classifier on training set: {:.2f}'
      .format(gnb.score(X_train, Y_train)))
print('Accuracy of  Gaussian classifier on test set: {:.2f}'
      .format(gnb.score(X_test, Y_test)))
###################################################################

# ################## DecisionTreeClassifier #######################
# from sklearn.tree import DecisionTreeClassifier

# Creating model object
model_dt = DecisionTreeClassifier()
# Training Model
model_dt.fit(X_train, Y_train)
pred_Tree = model_dt.predict(X_test)
X_predict = data_test[ ['Model', 'company', 'date', 'condition', 'mileage(kilometers)',
     'fuel_type', 'volume(cm3)', 'color', 'transmission', 'drive_unit', 'segment']]
test_pred2_Tree = model_dt.predict(X_predict)
print('Accuracy of DecisionTree classifier on training set: {:.2f}'
      .format(model_dt.score(X_train, Y_train)))
print('Accuracy of  DecisionTree classifier on test set: {:.2f}'
      .format(model_dt.score(X_test, Y_test)))
# ###################################################################
# #
dic = {0: 'cheap', 1: 'expensive', 2: 'moderate', 3: 'very expensive'}
submission = pd.DataFrame()
submission = data_test[['car_id']]
submission['Price Category'] = test_pred2
submission.replace({"Price Category": dic}, inplace=True)
submission.to_csv('submission_Random.csv' ,index=None)
