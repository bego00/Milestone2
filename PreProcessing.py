
# Split car-info column
def split(data):
    new = data["car-info"].str.split(",", n=2, expand=True)
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
