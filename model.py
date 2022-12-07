import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

data_train = pd.read_csv('cars-train.csv')
data_test = pd.read_csv('cars-test.csv')

data_train.head()
new = data_train["car-info"].str.split(",", n=2, expand=True)
data_train["Model"] = new[0]
data_train["company"] = new[1]
data_train["date"] = new[2]

data_train['date'] = data_train['date'].str.replace('\W', '', regex=True)
data_train['Model'] = data_train['Model'].str.replace('\W', '', regex=True)
data_train['company'] = data_train['company'].str.replace('\W', '', regex=True)

data_train.insert(1, 'Model', data_train.pop('Model'))
data_train.insert(2, 'company', data_train.pop('company'))
data_train.insert(3, 'date', data_train.pop('date'))
data_train.drop(columns=["car-info"], inplace=True)

data_train['date'] = pd.to_datetime(data_train['date'])

# Convert fuel_type column to lowercase
data_train['fuel_type'] = data_train['fuel_type'].apply(str.lower)

data_train['volume(cm3)'] = data_train['volume(cm3)'].fillna(data_train['volume(cm3)'].median())

data_train['segment'].isnull().value_counts().plot.bar()

data_train['drive_unit'].isnull().value_counts().plot.bar()

# Fill NAN values in [segment , drive_unit] columns
cols = ['segment', 'drive_unit']
for col in cols:
    # creates a random permutation of the categorical values
    permutation = np.random.permutation(data_train[col])

    # erase the empty values
    empty = np.where(permutation == "")
    permutation = np.delete(permutation, empty)

    # replace all empty values of the dataframe[segment]
    end = len(permutation)
    data_train[col] = data_train[col].apply(lambda x: permutation[np.random.randint(end)] if pd.isnull(x) else x)
    data_train[col] = data_train[col].fillna(data_train[col].mode()[0])
# Check outliers
# for col in data_train.columns:
#     fig = px.histogram(data_train, x=col)
#     fig.show()

cols = list(data_train.select_dtypes('object'))


# Applying label Encoding

def Feature_Encoder(data, cols):
    for c in cols:
        lbl = LabelEncoder()
        lbl.fit(list(data[c].values))
        data[c] = lbl.transform(list(data[c].values))
    return data


data_train = Feature_Encoder(data_train, cols)

# Correlation between columns
corr = data_train.corr()
# Top 50% Correlation training features with the Value
top_feature = corr.index[abs(corr['Price Category']) > 0.1]
# Correlation plot
plt.subplots(figsize=(12, 8))
top_corr = data_train[top_feature].corr()
sns.heatmap(top_corr, annot=True)
plt.show()
top_feature = top_feature.delete(-1)
X = data_train[top_feature]

X  # Selected Features
Y = data_train['Price Category']  # Target
# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# Create a Gaussian Classifier
clf = RandomForestClassifier(n_estimators=100, max_depth=15)

# Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
# Model Accuracy, how often is the classifier correct?
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
