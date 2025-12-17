import pandas as pd
from matplotlib import pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

data = "Crop_recommendation.csv"
df = pd.read_csv(data)

print("Data Loaded:")
print(df.head())

X = df.drop("label", axis=1)
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

def crop_prediction(N,P,K,temperature,humidity,pH,rainfall):
    sample = [[N,P,K,temperature,humidity,pH,rainfall]]
    output = (model.predict(sample))
    return output[0]

print("Model Accuracy:", model.score(X_test, y_test))

#sample =[[90, 42, 43, 20.8, 82.0, 6.5, 202.9]]
#output = (model.predict(sample))
#print("\nRecommended Crop: ", output[0])

if __name__ == "__main__":
    crop = crop_prediction(90, 42, 43, 23.8, 82.0, 8, 85)
    print("\nRecommended Crop: ",crop)

# df.hist()
# plt.show()

#scatter_matrix(df)
#plt.show()