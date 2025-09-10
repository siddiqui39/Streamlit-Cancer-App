import streamlit as st
import pickle
import pandas as pd
import os
import plotly.graph_objects as go
import numpy as np


#  Function to load and clean dataset
def get_clean_data():
    base_dir= os.path.dirname(os.path.dirname(__file__))
    file_path= os.path.join(base_dir, "data", "data.csv")
    data = pd.read_csv(file_path)

    data = data.drop(["Unnamed: 32", "id"], axis=1)
    data["diagnosis"] = data["diagnosis"].map({"B": 0, "M": 1})

    return data

# Create sidebar sliders for user input
def add_sidebar():
    st.sidebar.header("Cell Nuclei Measurements")

    data= get_clean_data()

    # Define slider labels and corresponding dataframe columns
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

    input_dict= {}
    # Create a slider for each feature
    for label, key in slider_labels:
        input_dict[key]= st.sidebar.slider(
            label,
            min_value= float(0),
            max_value= float(data[key].max()),
            value= float(data[key].mean())
        )
    return input_dict

# Normalize slider values between 0 to 1
def get_scaled_values(input_dict):
    data= get_clean_data()

    x= data.drop(["diagnosis"], axis=1)

    scaled_dict= {}

    # Min-max scaling
    for key, value in input_dict.items():
        max_val= x[key].max()
        min_val= x[key].min()
        # Correct min-max scaling formula
        scaled_value= (value - min_val / max_val - min_val)
        scaled_dict[key]= scaled_value

    return scaled_dict

# Generate radar chart using Plotly
def get_radar_chart(input_data):
    # Scale values for radar chart
    input_data= get_scaled_values(input_data)

    # Categories for the radar chart
    categories= ['Radius', 'Texture', 'Perimeter', 'Area',
                'Smoothness', 'Compactness',
                'Concavity', 'Concave Points',
                'Symmetry', 'Fractal Dimension']


    fig= go.Figure()

    # Add man values trace
    fig.add_trace(go.Scatterpolar(
        r= [
            input_data["radius_mean"], input_data["texture_mean"],
            input_data["perimeter_mean"], input_data["area_mean"],
            input_data["smoothness_mean"], input_data["compactness_mean"],
            input_data["concavity_mean"], input_data["concave points_mean"],
            input_data["symmetry_mean"], input_data["fractal_dimension_mean"]
        ],
        theta= categories,
        fill= "toself",
        name= "Mean Value"
    ))

    # Add standard error trace
    fig.add_trace(go.Scatterpolar(
        r= [
            input_data["radius_se"], input_data["texture_se"],
            input_data["perimeter_se"], input_data["area_se"],
            input_data["smoothness_se"], input_data["compactness_se"],
            input_data["concavity_se"], input_data["concave points_se"],
            input_data["symmetry_se"], input_data["fractal_dimension_se"]
        ],
        theta= categories,
        fill= "toself",
        name= "Standard Error"
    ))
    
    # Add worst value trace
    fig.add_trace(go.Scatterpolar(
        r=[
            input_data["radius_worst"], input_data["texture_worst"],
            input_data["perimeter_worst"], input_data["area_worst"],
            input_data["smoothness_worst"], input_data["compactness_worst"],
            input_data["concavity_worst"], input_data["concave points_worst"],
            input_data["symmetry_worst"], input_data["fractal_dimension_worst"]
        ],
        theta=categories,
        fill="toself",
        name="Worst Value"
    ))

    # Set radar chart layout
    fig.update_layout(
        polar= dict(
            radialaxis= dict(
                visible= True,
                range= [0, 1]
            )),
        showlegend= True
    )
    return fig

# Make predictions using pre-trained model
def add_predictions(input_data):
    # get absolute path to model folder
    base_dir = os.path.dirname(os.path.dirname(__file__))  # project root
    model_folder = os.path.join(base_dir, "model")

    model_path = os.path.join(model_folder, "pipeline.pkl")

    # Load model
    with open(model_path, "rb") as f:
        pipeline = pickle.load(f)

    # Convert input directionary to array
    input_array = np.array(list(input_data.values())).reshape(1, -1)

    # Make prediction
    prediction = pipeline.predict(input_array)

    st.subheader("Cell cluster prediction")
    st.write("The cell cluster is:")


    # Display predictions with colored HTML
    if prediction[0] == "B":
        st.write("<span class='diagnosis benign'>Benign</span>", unsafe_allow_html=True)
    else:
        st.write("<span class='diagnosis malicious'>Malicious</span>", unsafe_allow_html=True)

    # Show prediction probabilities
    probs= pipeline.predict_proba(input_array)
    st.write("Probability of being benign: ", probs[0][0])
    st.write("Probability of being malicious: ", probs[0][1])
    st.write("This app can assist medical professionals in making a diagnosis, but should not be used as a substitute for a professional diagnosis.")

    # Debug: show raw prediction
    st.write(prediction)


# App main interface
def main():
    st.set_page_config(
        page_title= "Breast Cancer Predictor",
        page_icon= ":female_doctor:",
        layout= "wide",
        initial_sidebar_state= "expanded"
    )

    # Load custom CSS
    with open("assets/style.css") as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)

    # Get user input from sidebar
    input_data= add_sidebar()

    # App description
    with st.container():
        st.title("Breast Cancer Predictor")
        st.write("Please connect us to your cytology lab to help diagnose breast cancer from your tissue sample. This app predicts using a machine learning model whether a breast mass is benign or malignant based on the measurements it receives from your lab. You can also update the measurements by hand using the sliders in the sidebar. ")

    # Layout: radar chart + predictions side by side
    col1, col2= st.columns([4,1])

    with col1:
        radar_chart= get_radar_chart(input_data)
        st.plotly_chart(radar_chart)
    with col2:
        add_predictions(input_data)



if __name__ == "__main__":
    main()
