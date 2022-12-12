import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


# Split car-info column
def split(data):
    new = data["car-info"].str.split(",", n=2, expand=True)
    data["Model"] = new[0]
    data["company"] = new[1]
    data["date"] = new[2]

    data['date'] = data['date'].str.replace('\W', '', regex=True)
    data['Model'] = data['Model'].str.replace('\W', '', regex=True)
    data['company'] = data['company'].str.replace('\W', '', regex=True)

    data.insert(1, 'Model', data.pop('Model'))
    data.insert(2, 'company', data.pop('company'))
    data.insert(3, 'date', data.pop('date'))
    data.drop(columns=["car-info"], inplace=True)
    data['date'] = pd.to_numeric(data['date'], errors='coerce')
    return data


# Change type of column ot datetime
def datetime(data, col):
    data[col] = pd.to_datetime(data[col])
    return data


# Convert  column to lowercase
def LowerCase(data, col):
    data[col] = data[col].apply(str.lower)
    return data


# Fill NAN num column with median
def Fill_median(data, col):
    data[col] = data[col].fillna(data[col].median())
    return data


# Fill NAN Categorical column with permutation
def Fill_Cate(data, cols):
    for col in cols:
        # creates a random permutation of the categorical values
        permutation = np.random.permutation(data[col])

        # erase the empty values
        empty = np.where(permutation == "")
        permutation = np.delete(permutation, empty)

        # replace all empty values of the dataframe[segment]
        end = len(permutation)
        data[col] = data[col].apply(lambda x: permutation[np.random.randint(end)] if pd.isnull(x) else x)
        # data[col] = data[col].fillna(data[col].mode()[0])
        return data


def one_hot_encode(data, cols):
    for i in cols:
        t = OneHotEncoder()
        data[i] = t.fit_transform(data[i])


# Applying label Encoding
def Feature_Encoder(data, cols):
    for c in cols:
        lbl = LabelEncoder()
        lbl.fit(list(data[c].values))
        data[c] = lbl.transform(list(data[c].values))
    return data


# Correlation between columns
def Corr(data, col):
    corr = data.corr()
    top_feature = corr.index[abs(corr[col]) > 0.1]
    # Correlation plot
    plt.subplots(figsize=(10, 6))
    top_corr = data[top_feature].corr()
    sns.heatmap(top_corr, annot=True)
    plt.show()
    top_feature = top_feature.delete(-1)
    C = data[top_feature]
    return C
