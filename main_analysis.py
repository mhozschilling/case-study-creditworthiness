import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

# TODO 1: import both datasets
df1 = pd.read_csv("application_record.csv")
df2 = pd.read_csv("credit_record.csv")

# TODO 2: merge datasets with join function
merged_df = pd.merge(left=df1, right=df2, on="ID", how="inner")

# TODO 3: clean data, adjust data types, remove outliers
# check data types
#print(merged_df.dtypes)
#print(len(merged_df))

# column FLAG_MOBIL, FLAG_WORK_PHONE, FLAG_PHONE and FLAG_EMAIL can be dropped as these are just contact details
merged_df.drop("FLAG_MOBIL", axis=1, inplace=True)
merged_df.drop("FLAG_WORK_PHONE", axis=1, inplace=True)
merged_df.drop("FLAG_PHONE", axis=1, inplace=True)
merged_df.drop("FLAG_EMAIL", axis=1, inplace=True)

# remove outliers for income
z_scores_income = np.abs((merged_df["AMT_INCOME_TOTAL"] - merged_df["AMT_INCOME_TOTAL"].mean()) / merged_df["AMT_INCOME_TOTAL"].std())
threshold = 3
merged_df = merged_df[z_scores_income <= threshold]

# remove outliers for CNT_FAM_MEMBERS
z_scores_family = np.abs((merged_df["CNT_FAM_MEMBERS"] - merged_df["CNT_FAM_MEMBERS"].mean()) / merged_df["CNT_FAM_MEMBERS"].std())
merged_df = merged_df[z_scores_family <= threshold]

# combining NAME_EDUCATION_TYPE options "Academic degree" and "Higher education" as they can be considered the same
# renaming secondary / secondary special variable to simplify to just secondary
merged_df["NAME_EDUCATION_TYPE"] = merged_df["NAME_EDUCATION_TYPE"].replace({"Academic degree": "Higher education", "Secondary / secondary special": "Secondary education"})

# in column NAME_FAMILY_STATUS renaming Civil marriage to Married and Single / not married to just Single
merged_df["NAME_FAMILY_STATUS"] = merged_df["NAME_FAMILY_STATUS"].replace({"Civil marriage": "Married", "Single / not married": "Single"})

# binning DAYS_BIRTH into new variable AGE_GROUP
merged_df["DAYS_BIRTH"] = merged_df["DAYS_BIRTH"].apply(lambda y: round(y / -365))
age_groups = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]
age_labels = ['20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69']
merged_df["AGE_GROUP"] = pd.cut(merged_df["DAYS_BIRTH"], bins=age_groups, labels=age_labels, right=False)
merged_df = merged_df.rename(columns={"DAYS_BIRTH": "YEARS_AGE"})

# all pensioners who are not working anymore, have 365243 days as DAYS_EMPLOYED.
# usually blank values should be replaced by the mean value, but in this case the max value makes more sense as they are all pensioners.
# then bin the values into groups of years of experience, from 5 to 5
number_to_replace = 365243
number_for_pensioners = -18251
merged_df["DAYS_EMPLOYED"] = merged_df["DAYS_EMPLOYED"].replace({number_to_replace: number_for_pensioners})
merged_df["DAYS_EMPLOYED"] = merged_df["DAYS_EMPLOYED"].apply(lambda z: round(z / -365))
years_groups = [-1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
years_labels = ['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49', 'Pensioners']
merged_df["YEARS_EMPLOYED_GROUPS"] = pd.cut(merged_df["DAYS_EMPLOYED"], bins=years_groups, labels=years_labels, right=False)
merged_df = merged_df.rename(columns={"DAYS_EMPLOYED": "YEARS_EMPLOYED"})

# if NAME_INCOME_TYPE is Pensioner, then OCCUPATION_TYPE should be Pensioner instead of blank.
# Drop all the remaining rows with blank values
merged_df.loc[merged_df["NAME_INCOME_TYPE"] == "Pensioner", "OCCUPATION_TYPE"] = "Pensioner"
merged_df.dropna(subset="OCCUPATION_TYPE", inplace=True)

# define the mapping of credit situation codes under STATUS to its descriptive labels for easier analysis
credit_labels = {
    '0': '1-29 days past due',
    '1': '30+ days past due',
    '2': '30+ days past due',
    '3': '30+ days past due',
    '4': '30+ days past due',
    '5': '30+ days past due',
    'C': 'Paid off',
    'X': 'No loan for the month'
}

merged_df["STATUS"] = merged_df["STATUS"].map(credit_labels)

# convert the MONTHS_BALANCE values into positive integers
merged_df["MONTHS_BALANCE"] = merged_df["MONTHS_BALANCE"].apply(lambda q: q * -1)

# TODO 4: export merged and cleaned dataset to new .csv file (maybe continue the analysis there?)

merged_df.to_csv("cleaned_data_analysis.csv", index=False)


# TODO 5: create graphs that describe the dataset

# MONTHS OF DATA AVAILABLE

print(merged_df["MONTHS_BALANCE"].describe())
plt.hist(merged_df["MONTHS_BALANCE"], bins=20)
plt.xlabel("Months of data available")
plt.ylabel("Frequency")
plt.title("Data Frequency on historic credit data")
plt.show()


# CREDIT STATUS

print(merged_df["STATUS"].describe())
credit_status = merged_df["STATUS"].value_counts(normalize=True) * 100
ax = sns.barplot(data=merged_df, x=credit_status.index, y=credit_status)
plt.xlabel("Credit status")
plt.ylabel("Percentage")
plt.title("Credit Status Distribution")
plt.xticks(rotation=25)
for i, percentage in enumerate(credit_status):
    ax.text(i, percentage, '{:.2f}%'.format(percentage), ha="center", va="bottom")
plt.show()


# INCOME

plt.boxplot(merged_df["AMT_INCOME_TOTAL"])
plt.xlabel("Income")
plt.ylabel("Value")
plt.title("Income Distribution")
plt.text(1.21, merged_df["AMT_INCOME_TOTAL"].median(), f"Median: {merged_df['AMT_INCOME_TOTAL'].median()}", ha="center", va="center", color="red")
plt.text(1.21, merged_df["AMT_INCOME_TOTAL"].quantile(0.25), f"Q1: {merged_df['AMT_INCOME_TOTAL'].quantile(0.25)}", ha="center", va="center", color="blue")
plt.text(1.21, merged_df["AMT_INCOME_TOTAL"].quantile(0.75), f"Q3: {merged_df['AMT_INCOME_TOTAL'].quantile(0.75)}", ha="center", va="center", color="blue")
plt.show()


# GENDER
print(merged_df["CODE_GENDER"].describe())
gender_counts = merged_df["CODE_GENDER"].value_counts()
gender_labels = ["Male", "Female"]
plt.pie(gender_counts.values, labels=gender_labels, autopct='%1.1f%%')
plt.title("Gender Distribution")
plt.show()

# AGE GROUP

print(merged_df["AGE_GROUP"].describe())
age_counts = merged_df["AGE_GROUP"].value_counts(normalize=True) * 100
age_counts = age_counts.reindex(age_labels)
ax = sns.barplot(data=merged_df, x=age_counts.index, y=age_counts, order=age_labels)
plt.xlabel("Age Group")
plt.ylabel("Percentage")
plt.title("Age Distribution")
for i, percentage in enumerate(age_counts.values):
    ax.text(i, percentage, '{:.2f}%'.format(percentage), ha="center", va="bottom")
plt.show()


# CHILDREN

print(merged_df["CNT_CHILDREN"].describe())
children_count = merged_df["CNT_CHILDREN"].value_counts(normalize=True) * 100
ax = sns.barplot(data=merged_df, x=children_count.index, y=children_count, order=[0, 1, 2, 3])
plt.xlabel("Children")
plt.ylabel("Percentage")
plt.title("Number of children")
for i, percentage in enumerate(children_count):
    ax.text(i, percentage, '{:.2f}%'.format(percentage), ha="center", va="bottom")
plt.show()

# FAMILY SIZE

print(merged_df["CNT_FAM_MEMBERS"].describe())
family_count = merged_df["CNT_FAM_MEMBERS"].value_counts(normalize=True) * 100
ax = sns.barplot(data=merged_df, x=family_count.index, y=family_count, order=[1, 2, 3, 4])
plt.xlabel("Family members")
plt.ylabel("Percentage")
plt.title("Number of family members")
for i, percentage in enumerate(family_count):
    ax.text(i, percentage, '{:.2f}%'.format(percentage), ha="center", va="bottom")
plt.show()

# MARITAL STATUS

print(merged_df["NAME_FAMILY_STATUS"].describe())
family_status = merged_df["NAME_FAMILY_STATUS"].value_counts(normalize=True) * 100
ax = sns.barplot(data=merged_df, x=family_status.index, y=family_status)
plt.xlabel("Family status")
plt.ylabel("Percentage")
plt.title("Family Status Distribution")
for i, percentage in enumerate(family_status):
    ax.text(i, percentage, '{:.2f}%'.format(percentage), ha="center", va="bottom")
plt.show()

# EDUCATION LEVEL

print(merged_df["NAME_EDUCATION_TYPE"].describe())
education_level = merged_df["NAME_EDUCATION_TYPE"].value_counts(normalize=True) * 100
ax = sns.barplot(data=merged_df, x=education_level.index, y=education_level)
plt.xlabel("Education")
plt.ylabel("Percentage")
plt.title("Education Level Distribution")
for i, percentage in enumerate(education_level):
    ax.text(i, percentage, '{:.2f}%'.format(percentage), ha="center", va="bottom")
plt.show()

# INCOME TYPE

print(merged_df["NAME_INCOME_TYPE"].describe())
income_type = merged_df["NAME_INCOME_TYPE"].value_counts(normalize=True) * 100
ax = sns.barplot(data=merged_df, x=income_type.index, y=income_type)
plt.xlabel("Income type")
plt.ylabel("Percentage")
plt.title("Income Type Distribution")
for i, percentage in enumerate(income_type):
    ax.text(i, percentage, '{:.2f}%'.format(percentage), ha="center", va="bottom")
plt.show()

# OCCUPATION TYPE

print(merged_df["OCCUPATION_TYPE"].describe())
occupation_type = merged_df["OCCUPATION_TYPE"].value_counts(normalize=True) * 100
occupation_type_table = pd.DataFrame({"Occupation type": occupation_type.index, "Percentage": occupation_type})
occupation_type_table.sort_values("Occupation type", inplace=True)
occupation_type_table.reset_index(drop=True, inplace=True)
# occupation_type_table.to_csv("Occupation Type Distribution.csv", index=False)

# YEARS EMPLOYED

print(merged_df["YEARS_EMPLOYED_GROUPS"].describe())
years_groups = merged_df["YEARS_EMPLOYED_GROUPS"].value_counts()
years_groups = years_groups.sort_index()
plt.bar(years_groups.index, years_groups)
plt.xlabel("Years employed")
plt.ylabel("Percentage")
plt.title("Years Employed Distribution")
total_years_groups = years_groups.sum()
for i, count in enumerate(years_groups):
    percentage = (count / total_years_groups) * 100
    plt.text(i, count, '{:.2f}%'.format(percentage), ha="center", va="bottom")
plt.show()

# CAR

print(merged_df["FLAG_OWN_CAR"].describe())
car_ownership = merged_df["FLAG_OWN_CAR"].value_counts()
car_labels = ["No car", "Owns at least one car"]
plt.pie(car_ownership.values, labels=car_labels, autopct='%1.1f%%')
plt.title("Car Ownership")
plt.show()

# REAL ESTATE

print(merged_df["FLAG_OWN_REALTY"].describe())
house_ownership = merged_df["FLAG_OWN_REALTY"].value_counts()
house_labels = ["Not house owner", "House owner"]
plt.pie(house_ownership.values, labels=house_labels, autopct='%1.1f%%')
plt.title("House Ownership")
plt.show()


# TODO 7: add graphs for the variables with the highest coefficients

# CORRELATION MATRIX WITH ALL RELEVANT VARIABLES
print(merged_df.columns)
variables = ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'CNT_CHILDREN',
       'AMT_INCOME_TOTAL', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE',
       'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'YEARS_AGE',
       'OCCUPATION_TYPE', 'CNT_FAM_MEMBERS', 'STATUS', 'YEARS_EMPLOYED_GROUPS'] #list of variables to include
subset_df = merged_df[variables]  # Subset your DataFrame with the desired variables

def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    return np.round(np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1))), decimals=1)

# Compute the correlation matrix using Cramér's V statistic
correlation_matrix = pd.DataFrame(index=subset_df.columns, columns=subset_df.columns, dtype=float)
for i in range(len(variables)):
    for j in range(i+1, len(variables)):
        correlation_matrix.iloc[i, j] = cramers_v(subset_df.iloc[:, i], subset_df.iloc[:, j])

correlation_matrix = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', square=True, fmt=".1f")
plt.xticks(rotation=65)
plt.title('Correlation Matrix Heatmap')
plt.tight_layout()
plt.show()


y_order = ['No loan for the month', 'Paid off', '1-29 days past due', '30+ days past due']

# 1. NAME_EDUCATION_TYPE
# grouped bar chart or a stacked bar chart
df_counts = merged_df.groupby(['NAME_EDUCATION_TYPE', 'STATUS']).size().unstack()
df_percentages = df_counts.div(df_counts.sum(axis=1), axis=0) * 100
x_order = ['Lower secondary', 'Secondary education', 'Incomplete higher', 'Higher education']
df_percentages = df_percentages.reindex(x_order)
df_percentages = df_percentages[y_order]
ax = df_percentages.plot(kind='bar', stacked=True)
ax.set_xlabel('Education Level')
plt.xticks(rotation=35)
ax.set_ylabel('Percentage')
ax.set_title('Debt Repayment Status by Education Level')
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 0.5))
plt.show()
# export findings into a .csv table
fig, ax = plt.subplots()
ax.axis('off')
ax.table(cellText=df_percentages.values, cellLoc='center', rowLabels=df_percentages.index, colLabels=df_percentages.columns, loc='center')
table_title = 'Debt Repayment Status by Education Level'
plt.title(table_title, fontsize=16)
plt.tight_layout()
output_file = 'table.csv'
df_percentages.to_csv(output_file)

# 2. NAME_HOUSING_TYPE

x_order_housing = y_order
df_counts = merged_df.groupby('NAME_HOUSING_TYPE')['STATUS'].value_counts().unstack().fillna(0)
df_percentages = df_counts.div(df_counts.sum(axis=1), axis=0) * 100
df_percentages = df_percentages[x_order_housing]
plt.figure(figsize=(10, 6))
sns.heatmap(df_percentages, annot=True, cmap='Blues', fmt='.1f', cbar=True, xticklabels=x_order_housing)
plt.title('Correlation between Debt Repayment Status and Housing Type')
plt.xlabel('Debt Repayment Status')
plt.xticks(rotation=45)
plt.ylabel('Housing Type')
plt.show()


# 3. AGE_GROUP

grouped_df = merged_df.groupby('AGE_GROUP')['STATUS'].value_counts(normalize=True).unstack() * 100
grouped_df = grouped_df.reindex(grouped_df.mean().sort_values(ascending=False).index, axis=1)
plt.figure(figsize=(10, 8))
grouped_df.plot(kind='bar', stacked=True)
plt.xlabel('Age Group')
plt.ylabel('Percentage')
plt.title('Debt Repayment Status by Age Group')
plt.legend(title='Debt Repayment Status', bbox_to_anchor=(1.05, 0.5))
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# 4. OCCUPATION_TYPE

# stacked bar plot
subset_df = merged_df[['STATUS', 'OCCUPATION_TYPE']]
grouped_df = subset_df.groupby(['OCCUPATION_TYPE', 'STATUS']).size().unstack()
grouped_df['Percentage'] = grouped_df['1-29 days past due'] / grouped_df.sum(axis=1)
grouped_df = grouped_df.sort_values('Percentage', ascending=False)
grouped_df.drop('Percentage', axis=1, inplace=True)
grouped_df_percent = grouped_df.apply(lambda x: x / x.sum() * 100, axis=1)
plt.figure(figsize=(10, 6))
grouped_df_percent.plot(kind='bar', stacked=True)
plt.xlabel('Occupation Type')
plt.xticks(rotation=65)
plt.ylabel('Percentage')
plt.title('Debt Repayment Status by Occupation Type')
plt.legend(title='Debt Repayment Status', bbox_to_anchor=(1.05, 0.5))
plt.tight_layout()
plt.show()

# turn into .csv file
grouped_df_percent.to_csv("occupation-type-correlation.csv")

# 5. NAME_FAMILY_STATUS

# stacked bar plot
subset_df = merged_df[['STATUS', 'NAME_FAMILY_STATUS']]
grouped_df = subset_df.groupby(['NAME_FAMILY_STATUS', 'STATUS']).size().unstack()
grouped_df['Percentage'] = grouped_df['1-29 days past due'] / grouped_df.sum(axis=1)
grouped_df = grouped_df.sort_values('Percentage', ascending=False)
grouped_df.drop('Percentage', axis=1, inplace=True)
grouped_df_percent = grouped_df.apply(lambda x: x / x.sum() * 100, axis=1)
plt.figure(figsize=(10, 6))
grouped_df_percent.plot(kind='bar', stacked=True)
plt.xlabel('Family status')
plt.xticks(rotation=65)
plt.ylabel('Percentage')
plt.title('Debt Repayment Status by Family status')
plt.legend(title='Debt Repayment Status', bbox_to_anchor=(1.05, 0.5))
plt.tight_layout()
plt.show()

# convert to .csv file
grouped_df_percent.to_csv("family-status-correlation.csv")


# 6. AMT_INCOME_TOTAL

# violinplot
sns.violinplot(data=merged_df, x="STATUS", y="AMT_INCOME_TOTAL")
plt.xlabel('Debt Repayment Status')
plt.ylabel('Income')
plt.title('Loan Status vs Income')
plt.show()

# unstacked barplots
income_bins = pd.qcut(merged_df['AMT_INCOME_TOTAL'], q=5, labels=['Low', 'Medium-Low', 'Medium', 'Medium-High', 'High'])
grouped_df = merged_df.groupby([income_bins, 'STATUS']).size().unstack().div(merged_df.groupby(income_bins).size(), axis=0) * 100
grouped_df.reset_index(inplace=True)
melted_df = grouped_df.melt(id_vars='AMT_INCOME_TOTAL', value_name='Percentage', var_name='Debt Repayment Status')
plt.figure(figsize=(10, 8))
sns.barplot(data=melted_df, x="Debt Repayment Status", y="Percentage", hue="AMT_INCOME_TOTAL")
plt.xlabel('Debt Repayment Status')
plt.ylabel('Percentage')
plt.title('Loan Status vs Income')
plt.legend(title='Income Category')
plt.show()

# 7. YEARS_EMPLOYED_GROUPS

grouped_df = merged_df.groupby('YEARS_EMPLOYED_GROUPS')['STATUS'].value_counts(normalize=True).unstack() * 100
grouped_df = grouped_df.reindex(grouped_df.mean().sort_values(ascending=False).index, axis=1)
plt.figure(figsize=(10, 8))
grouped_df.plot(kind='bar', stacked=True)
plt.xlabel('Years Employed')
plt.ylabel('Percentage')
plt.title('Debt Repayment Status by Years Employed')
plt.legend(title='Debt Repayment Status', bbox_to_anchor=(1.05, 0.5))
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# 8. CODE_GENDER

subset_df = merged_df[['STATUS', 'CODE_GENDER']]
contingency_table = pd.crosstab(subset_df['CODE_GENDER'], subset_df['STATUS'])
row_percentages = contingency_table.div(contingency_table.sum(axis=1), axis=0) * 100
plt.figure(figsize=(10, 8))
ax = row_percentages.plot(kind='bar', stacked=True)
plt.xlabel('Gender')
plt.ylabel('Percentage')
plt.title('Debt Repayment Status by Gender')
plt.legend(title='Debt Repayment Status')
gender_labels = ['Male', 'Female']
ax.set_xticklabels(gender_labels)
plt.show()


# 9. FLAG_OWN_CAR

subset_df = merged_df[['STATUS', 'FLAG_OWN_CAR']]
contingency_table = pd.crosstab(subset_df['FLAG_OWN_CAR'], subset_df['STATUS'])
row_percentages = contingency_table.div(contingency_table.sum(axis=1), axis=0) * 100
plt.figure(figsize=(10, 8))
ax = row_percentages.plot(kind='bar', stacked=True)
plt.xlabel('Car ownership')
plt.ylabel('Percentage')
plt.title('Debt Repayment Status by Car Ownership')
plt.legend(title='Debt Repayment Status')
gender_labels = ['Not car owner', 'Car owner']
ax.set_xticklabels(gender_labels)
plt.show()
