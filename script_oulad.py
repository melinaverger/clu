"""
The goal of this script is then to use educational data for clustering analysis.
"""

import zipfile
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import numpy as np


# set the number of clusters to be studied
# here, we could only have 2 as we have only 0-or-1-value result
NB_CLUSTER = 2
# set the number of principal components to lower dimensional space
# max is the number of features
NB_COMPONENT = 8


ZIP_FILE_NAME = "anonymisedData.zip"
FILE_NAME = "studentInfo.csv"


# ===== (start) Functions from another project =====

def load_data(file_name):
    try:
        # zip file handler
        zf = zipfile.ZipFile(ZIP_FILE_NAME)

        # extract a file from zip folder
        ext_file = zf.open(file_name)

        df = pd.read_csv(ext_file)
        return df

    except:
        raise NameError("file_name argument of function load_data must correspond to a name of a file present into the zip file (extension included).")


def prepare_dataset(df):
    keep_columns = ['id_student', 'gender', 'region', 'highest_education',
                    'imd_band', 'age_band', 'num_of_prev_attempts',
                    'studied_credits', 'disability', 'final_result']
    df = df[keep_columns]
    # drop rows when same students' id (redundant students' info)
    df = df.drop_duplicates(subset=['id_student'], keep='first')
    return df.drop(columns=['id_student'])


def encode_imd(x):
    if x in ['0-10%', '10-20', '20-30%', '30-40%', '40-50%']:
        return 1
    elif x in ['50-60%', '60-70%', '70-80%', '80-90%', '90-100%']:
        return 0
    else:
        raise ValueError("Missing values should have been removed.")


def encode_gender(x):
    if x == 'M':
        return 1
    else:
        return 0


def add_protected_imd(dataframe):
    """
    Apply encoding for IMD protected attribute.
    Use function encode_imd.
    """
    # remove when IMD missing
    new_dataframe = dataframe.dropna(subset=['imd_band'])
    new_dataframe.loc[:, 'imd_band'] = new_dataframe.imd_band.apply(encode_imd)  # specific syntax to avoid SettingWithCopyWarning
    return new_dataframe


def add_protected_gender(dataframe):
    """
    Apply encoding for gender protected attribute.
    Use function encode_gender.
    """
    dataframe.loc[:, 'gender'] = dataframe.gender.apply(encode_gender)  # specific syntax to avoid SettingWithCopyWarning
    return dataframe


def filter_final_result(dataframe):
    """
    Return the dataframe filtered on final_result column.
    """
    column = 'final_result'
    options = ['Pass', 'Fail']
    return dataframe[dataframe[column].isin(options)]  # keep associated rows


def encode_final_result(x):
    if x == "Pass":
        return 1
    else:  # "Fail"
        return 0


def encode_disability(x):
    if x == "Y":
        return 1
    else:  # "N"
        return 0


def encode_education(x):
    if x == "No Formal quals":
        return 0
    elif x == "Lower Than A Level":
        return 1
    elif x == "A Level or Equivalent":
        return 2
    elif x == "HE Qualification":
        return 3
    else:  # "Post Graduate Qualification"
        return 4


def encode_age(x):
    if x == "0-35":
        return 0
    elif x == "35-55":
        return 1
    else:  # "55<="
        return 2


def encode_region(dataframe):
    dict_region = {'East Anglian Region': 0,
                   'Scotland': 1,
                   'North Western Region': 2,
                   'South East Region': 3,
                   'West Midlands Region': 4,
                   "Wales": 5,
                   "North Region": 6,
                   "South Region": 7,
                   "Ireland": 8,
                   "South West Region": 9,
                   "East Midlands Region": 10,
                   "Yorkshire Region": 11,
                   "London Region": 12}
    dataframe['region'] = dataframe.region.map(dict_region)
    return dataframe


def encode_variables(dataframe):
    """
    Apply encoding for all variables except protected attributes and numerical
    variables.
    Parameters
    ----------
    dataframe : pd.DataFrame
        The initial dataframe
    Returns
    ----------
    pd.DataFrame
        The final dataframe
    """
    dataframe.loc[:, 'final_result'] = dataframe.final_result.apply(encode_final_result)  # specific syntax to avoid SettingWithCopyWarning
    dataframe.loc[:, 'disability'] = dataframe.disability.apply(encode_disability)
    dataframe.loc[:, 'highest_education'] = dataframe.highest_education.apply(encode_education)
    dataframe.loc[:, 'age_band'] = dataframe.age_band.apply(encode_age)
    dataframe = encode_region(dataframe)
    return dataframe

# ===== (end) Functions from another project =====


def get_data():
    data = load_data(FILE_NAME)
    data = prepare_dataset(data)
    data = add_protected_gender(data)
    data = add_protected_imd(data)
    data = filter_final_result(data)
    data = encode_variables(data)
    return data


data = get_data()
X_data, y_data = data.drop(columns=["final_result"]), data["final_result"]
print(X_data.shape)
print(y_data.shape)


# scale the data (NECESSARY step)
scaler = MinMaxScaler()
data = scaler.fit_transform(data)


# lower dimensional space 
estimator = PCA(n_components=NB_COMPONENT)
X_pca = estimator.fit_transform(X_data)


# plot clusters (2 colors at most depending on NB_CLUSTER)
colors = ["black", "blue", "orange", "yellow", "pink", "red", "lime", "cyan"]

for i in range(NB_CLUSTER):
    px = X_pca[:, 0][y_data == i] # first component
    py = X_pca[:, 1][y_data == i] # second component
    plt.scatter(px, py, c=colors[i])

plt.legend(('0', '1'))
plt.xlabel("Première composante principale")
plt.ylabel("Deuxième composante principale")

plt.show()


# now we want to see the clusters found by k-mean algorithm
kmeans = KMeans(n_clusters=NB_CLUSTER).fit(X_data)
kmeans_pred = kmeans.predict(X_data)
# kmeans_pred.shape == y_digits.shape


# therefore we plot the 2 first components like before but with the clusters learned by k-mean
for i in range(NB_CLUSTER):
    px = X_pca[:, 0][kmeans_pred == i]
    py = X_pca[:, 1][kmeans_pred == i]
    plt.scatter(px, py, c=colors[i])

plt.legend(('0', '1'))
plt.xlabel("Première composante principale")
plt.ylabel("Deuxième composante principale")

plt.show()


# === (start) see the optimal number of components ===
# https://www.mikulskibartosz.name/pca-how-to-choose-the-number-of-components/

pca = PCA(n_components=NB_COMPONENT).fit(data)

fig, ax = plt.subplots()
xi = np.arange(1, NB_COMPONENT+1, step=1)
print("xi :", xi)
print(xi.shape)
y = np.cumsum(pca.explained_variance_ratio_)
print("y :", y)
print(y.shape)

plt.ylim(0.0,1.1)
plt.plot(xi, y, marker='o', linestyle='--', color='b')

plt.xlabel('Number of Components')
plt.xticks(np.arange(0, 11, step=1)) #change from 0-based array index to 1-based human-readable label
plt.ylabel('Cumulative variance (%)')
plt.title('The number of components needed to explain variance')

plt.axhline(y=0.95, color='r', linestyle='-')
plt.text(0.5, 0.85, '95% cut-off threshold', color = 'red', fontsize=16)

ax.grid(axis='x')
plt.show()

# === (end) see the optimal number of components ===