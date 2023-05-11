import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import OrdinalEncoder

# TODO 1: import both datasets
df1 = pd.read_csv("application_record.csv")
df2 = pd.read_csv("credit_record.csv")

# TODO 2: merge datasets with join function
merged_df = pd.merge(left=df1, right=df2, on="ID", how="inner")

# TODO 3: clean data, adjust data types, remove outliers
# check data types
# print(merged_df.dtypes)

# column FLAG_MOBIL, FLAG_WORK_PHONE, FLAG_PHONE and FLAG_EMAIL can be dropped as these are just contact details
merged_df.drop("FLAG_MOBIL", axis=1, inplace=True)
merged_df.drop("FLAG_WORK_PHONE", axis=1, inplace=True)
merged_df.drop("FLAG_PHONE", axis=1, inplace=True)
merged_df.drop("FLAG_EMAIL", axis=1, inplace=True)

# converting boolean data types from string to integer values 0 and 1 for better analysis
merged_df["CODE_GENDER"] = merged_df["CODE_GENDER"].replace({"M": 0, "F": 1})
merged_df["FLAG_OWN_CAR"] = merged_df["FLAG_OWN_CAR"].replace({"N": 0, "Y": 1})
merged_df["FLAG_OWN_REALTY"] = merged_df["FLAG_OWN_REALTY"].replace({"N": 0, "Y": 1})

# if NAME_INCOME_TYPE is Pensioner, then OCCUPATION_TYPE should be Pensioner instead of blank.
# Drop all the remaining rows with blank values and convert categorical variables into numerical representations
merged_df.loc[merged_df["NAME_INCOME_TYPE"] == "Pensioner", "OCCUPATION_TYPE"] = "Pensioner"
merged_df.dropna(subset="OCCUPATION_TYPE", inplace=True)
merged_df["OCCUPATION_TYPE"] = merged_df["OCCUPATION_TYPE"].replace({"Cleaning staff": 1, "Cooking staff": 2, "Core staff": 3, "Drivers": 4, "High skill tech staff": 5, "HR staff": 6, "IT staff": 7, "Laborers": 8, "Low-skill Laborers": 9, "Managers": 10, "Medicine staff": 11, "Pensioner": 12, "Private service staff": 13, "Realty agents": 14, "Sales staff": 15, "Secretaries": 16, "Security staff": 17, "Waiters/barmen staff": 18, "Accountants": 19})

# convert your categorical variables into numerical representations
merged_df["NAME_INCOME_TYPE"] = merged_df["NAME_INCOME_TYPE"].replace({"Student": 1, "Working": 2, "Commercial associate": 3, "State servant": 4, "Pensioner": 5})
merged_df["NAME_EDUCATION_TYPE"] = merged_df["NAME_EDUCATION_TYPE"].replace({"Lower secondary": 1, "Secondary / secondary special": 2, "Incomplete higher": 3, "Higher education": 4, "Academic degree": 4})
merged_df["NAME_FAMILY_STATUS"] = merged_df["NAME_FAMILY_STATUS"].replace({"Single / not married": 1, "Civil marriage": 1, "Married": 1, "Separated": 2, "Widow": 3})
merged_df["NAME_HOUSING_TYPE"] = merged_df["NAME_HOUSING_TYPE"].replace({"With parents": 1, "Rented apartment": 2, "Municipal apartment": 3, "Office apartment": 4, "Co-op apartment": 5, "House / apartment": 6})

# adjusting days since birth to years old into positive values
merged_df["DAYS_BIRTH"] = merged_df["DAYS_BIRTH"].apply(lambda y: round(y / -365))
merged_df = merged_df.rename(columns={"DAYS_BIRTH": "YEARS_AGE"})

# all pensioners who are not working anymore, have 365243 days as DAYS_EMPLOYED.
# usually blank values should be replaced by the mean value, but in this case the max value (a little more actually, to differentiate) makes more sense as they are all pensioners.
number_to_replace = 365243
number_for_pensioners = -18250
merged_df["DAYS_EMPLOYED"] = merged_df["DAYS_EMPLOYED"].replace({number_to_replace: number_for_pensioners})
merged_df["DAYS_EMPLOYED"] = merged_df["DAYS_EMPLOYED"].apply(lambda z: round(z / -365))
merged_df = merged_df.rename(columns={"DAYS_EMPLOYED": "YEARS_EMPLOYED"})

# remove outliers for AMT_INCOME_TOTAL and perform data normalization
z_scores_income = np.abs((merged_df["AMT_INCOME_TOTAL"] - merged_df["AMT_INCOME_TOTAL"].mean()) / merged_df["AMT_INCOME_TOTAL"].std())
threshold = 3
merged_df = merged_df[z_scores_income <= threshold]
min_value = merged_df["AMT_INCOME_TOTAL"].min()
max_value = merged_df["AMT_INCOME_TOTAL"].max()
merged_df["AMT_INCOME_TOTAL"] = (merged_df["AMT_INCOME_TOTAL"] - min_value) / (max_value - min_value)

# remove outliers for CNT_FAM_MEMBERS
z_scores_family = np.abs((merged_df["CNT_FAM_MEMBERS"] - merged_df["CNT_FAM_MEMBERS"].mean()) / merged_df["CNT_FAM_MEMBERS"].std())
merged_df = merged_df[z_scores_family <= threshold]

# define the mapping of credit situation codes under STATUS to ordinal values
credit_labels = {
    '0': 2,
    '1': 3,
    '2': 4,
    '3': 5,
    '4': 6,
    '5': 7,
    'C': 1,
    'X': 0
}
merged_df["STATUS"] = merged_df["STATUS"].map(credit_labels)

# convert the MONTHS_BALANCE values into positive integers
merged_df["MONTHS_BALANCE"] = merged_df["MONTHS_BALANCE"].apply(lambda q: q * -1)

print(merged_df.dtypes)

# TODO 4: export merged and cleaned dataset to new .csv file (maybe continue the analysis there?)

merged_df.to_csv("cleaned_data_ML.csv", index=False)


# TODO 6: analyze which data points most correlate with creditworthiness

merged_df_X = merged_df.copy()
merged_df_X.drop("STATUS", axis=1, inplace=True)
merged_df_X.drop("ID", axis=1, inplace=True)

X = merged_df_X  # DataFrame containing selected variables
y = merged_df["STATUS"]  # Series or column containing loan repayment status (target)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")

# Print the evaluation metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)
print("")

# Get the coefficients or feature importances
if isinstance(model, LogisticRegression):
    # For logistic regression, get the coefficients
    coefficients = model.coef_[0]
    feature_names = X.columns
    feature_importances = dict(zip(feature_names, coefficients))
elif isinstance(model, RandomForestClassifier):
    # For random forest, get the feature importances
    feature_importances = model.feature_importances_
    feature_names = X.columns

# Sort the feature importances in descending order
sorted_importances = sorted(feature_importances.items(), key=lambda x: abs(x[1]), reverse=True)

# Print the feature importances
for feature, importance in sorted_importances:
    print(feature, ":", round(importance, 3))
