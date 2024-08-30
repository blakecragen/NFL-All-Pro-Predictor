import pandas as pd
import openpyxl
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Load player data
file_path = "data/WR.xlsx"
year = "2020"
df = pd.read_excel(file_path, sheet_name=year)
grouped_df = df.groupby(['id', 'name', 'position', 'team', 'season', 'season_type'], as_index=False).sum()
grouped_df.drop('week', axis=1, inplace=True)

# Load Pro Bowler data
pro_bowlers_df = pd.read_excel('all_pro_bowlers.xlsx')
pro_bowlers_df['Player'] = pro_bowlers_df['Player'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')

# Mark players as Pro Bowlers in the main dataset
# For training, put a column that indicated if a player was a pro bowler
grouped_df['is_pro_bowler'] = grouped_df['name'].apply(lambda x: any(pro_bowler in x for pro_bowler in pro_bowlers_df['Player']))

# Delete no numeric columns
X = grouped_df.drop(['id', 'name', 'position', 'team', 'season', 'season_type', 'is_pro_bowler'], axis=1)
y = grouped_df['is_pro_bowler']

grouped_df.to_excel('check.xlsx', index=False)
# Split the dataset for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))

# Visualize
plt.figure(figsize=(20,10))
plot_tree(model, filled = True, feature_names=X.columns.tolist(), class_names = ['Not All Pro', 'All Pro'])
plt.show()

y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize = (8, 6))
sns.heatmap(cm, annot = True, fmt = 'd', cmap = 'Blues', xticklabels = ['Not All Pro', 'All Pro'], yticklabels = ['Not All Pro', 'All Pro'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix for Decision Trees')
plt.show()