import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os


def create_model(data):
    x= data.drop(["diagnosis"], axis=1)
    y= data["diagnosis"]

    # scale the data
    scaler= StandardScaler()
    x= scaler.fit_transform(x)

    # split the data
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size= 0.2, random_state= 42
    )

    # train model
    model= LogisticRegression()
    model.fit(x_train, y_train)#

    # test model
    y_pred= model.predict(x_test)
    print("Accuracy of our model: ", accuracy_score(y_test, y_pred))
    print("Classification report: \n", classification_report(y_test, y_pred))

    return  model, scaler

def get_clean_data():
    data= pd.read_csv("../data/data.csv")
    #print(data.columns)

    data= data.drop(["Unnamed: 32", "id"], axis=1)

    return data

def main():
    data= get_clean_data()

    model, scaler= create_model(data)
    # get the folder, where this script is located
    base_dir= os.path.dirname(__file__)
    model = os.path.join(base_dir, "model.pkl")
    scaler = os.path.join(base_dir, "scaler.pkl")


    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)






if __name__ == "__main__":
    main()

