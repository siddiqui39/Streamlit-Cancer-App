import streamlit as st
import pickle
import pandas as pd

def get_clean_data():
    data= pd.read_csv("../data/data.csv")
    #print(data.columns)
    data= data.drop(["Unnamed: 32", "id"], axis=1)
    return data


# Create sidebar
def add_sidebar():
    st.sidebar.header("Cell Nuclei Measurements")

    data= get_clean_data()

    slider_labels = [
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"),
        ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst"),
    ]


# App interface
def main():
    st.set_page_config(
        page_title= "Breast Cancer Predictor",
        page_icon= ":female_doctor:",
        layout= "wide",
        initial_sidebar_state= "expanded"
    )

add_sidebar()

with st.container():
    st.title("Breast Cancer Predictor")
    st.write("Please connect us to your cytology lab to help diagnose breast cancer from your tissue sample. This app predicts using a machine learning model whether a breast mass is benign or malignant based on the measurements it receives from your lab. You can also update the measurements by hand using the sliders in the sidebar. ")

col1, col2= st.columns([4,1])

with col1:
    st.write("this is column 1")
with col2:
    st.write("this is column 2")





















if __name__ == "__main__":
    main()