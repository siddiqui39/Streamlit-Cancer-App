import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os


def create_model(data):
    x= data.drop(["diagnosis"], axis=1)
    y= data["diagnosis"]

    # Define pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('log_reg', LogisticRegression())
    ])

    # split the data
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size= 0.2, random_state= 42
    )

    # train model
    pipeline.fit(x_train, y_train)

    # test model
    y_pred= pipeline.predict(x_test)
    print("Accuracy of our model: ", accuracy_score(y_test, y_pred))
    print("Classification report: \n", classification_report(y_test, y_pred))

    return  pipeline

def get_clean_data():
    data= pd.read_csv("../data/data.csv")
    #print(data.columns)

    data= data.drop(["Unnamed: 32", "id"], axis=1)

    return data

def main():
    # Get project base directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data = get_clean_data()

    # Create model and scaler
    pipeline = create_model(data)

    # Define model folder inside project
    model_folder = os.path.join(base_dir, "model")
    os.makedirs(model_folder, exist_ok=True)

    # Define absolute paths for pickle files
    model_path = os.path.join(model_folder, "pipeline.pkl")


    # Debug
    print("Pipeline path:", model_path)


    # Save pickle
    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)


if __name__ == "__main__":
    main()

