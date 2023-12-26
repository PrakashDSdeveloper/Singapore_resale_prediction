import streamlit as st
from streamlit_option_menu import option_menu
import json
import requests
import pandas as pd
from geopy.distance import geodesic
import statistics
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

# -------------------------Reading the data on Lat and Long of all the MRT Stations in Singapore------------------------
data = pd.read_csv('mrt.csv')
mrt_location = pd.DataFrame(data)

# -------------------------------This is the configuration page for our Streamlit Application---------------------------
st.set_page_config(
    page_title="Singapore Resale Flat Prices Prediction",
    layout="wide"
)

# -------------------------------This is the sidebar in a Streamlit application, helps in navigation--------------------
with st.sidebar:
    selected = option_menu("Resale Flat Prices Prediction", ["Home", "Predict Resale Price"],
                           icons=["house", "book"],
                           styles={"nav-link": {"font": "sans serif", "font-size": "20px", "text-align": "centre"},
                                   "nav-link-selected": {"font": "sans serif", "background-color": "FF0000"},
                                   "icon": {"font-size": "20px"}
                                   }
                           )
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
background-color: #e1d7f2;
background-size :cover;
}
[data-testid="stSidebar"]{
background-image: url("https://blog.thomascook.in/wp-content/uploads/2022/08/Most-Iconic-Landmarks-in-Singapore-1024x682.jpg");
background-position :left;

}
[data-testid="stHeader"]{
background-color: #FF0000;
background-position :center;

}
[data-baseweb="tab"]{
background-color: #e1d7f2;
}

</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)
# -----------------------------------------------About Project Section--------------------------------------------------
if selected == "Home":
    st.markdown("# :red[Singapore Resale Flat Prices Prediction]")
    st.markdown('<div style="height: 50px;"></div>', unsafe_allow_html=True)
    st.write("### :red[Project overview :] The objective of this project is to develop a machine learning model and deploy it as a user-friendly web application to predict the resale prices of flats in Singapore. The motivation behind the project is to address the challenges in accurately estimating resale values in the competitive Singaporean resale flat market. The project aims to assist both potential buyers and sellers by providing an estimated resale price based on historical data of resale flat transactions. ")
    st.markdown("### :red[Technologies :] Python, Pandas, Numpy, Scikit-Learn, Streamlit, Python scripting, "
            "Machine Learning, Data Preprocessing, Visualization, EDA, Model Building, Data Wrangling, "
            "Model Deployment")
    st.markdown("### :red[Domain :] Real Estate")

# ------------------------------------------------Predictions Section---------------------------------------------------
if selected == "Predict Resale Price":
    st.markdown("### :orange[Predicting Resale Price]")

    try:
        with st.form("form1"):

            # -----New Data inputs from the user for predicting the resale price----- 
            street_name = st.text_input('Enter Street name')
            block = st.text_input("Block Number")
            floor_area_sqm = st.number_input('Floor Area (Per Square Meter)', min_value=1.0, max_value=500.0)
            lease_commence_date = st.number_input('Lease Commence Date')
            storey_range = st.text_input("Storey Range (Format: 'Value1' TO 'Value2')")

            # -----Submit Button for PREDICT RESALE PRICE-----
            submit_button = st.form_submit_button(label="PREDICT RESALE PRICE")

            if submit_button is not None:
                with open(r"model_1.pkl", 'rb') as file:
                    loaded_model = pickle.load(file)
                with open(r'scaler_1.pkl', 'rb') as f:
                    scaler_loaded = pickle.load(f)

                # -----Calculating lease_remain_years using lease_commence_date-----
                lease_remain_years = 99 - (2023 - lease_commence_date)

                # -----Calculating median of storey_range to make our calculations quite comfortable-----
                split_list = storey_range.split(' TO ')
                int_list = [int(val) for val in storey_range.split() if val.isdigit()]
                storey_median = statistics.median(int_list)

                # -----Getting the address by joining the block number and the street name-----
                origin = []

                # -----Getting the address by joining the block number and the street name-----
                address = block + " " + street_name
                data = pd.read_csv('df_coordinates.csv')
                

            # Filter the DataFrame based on the block number and road name
                filtered_data = data[(data['blk_no'] == block) & (data['road_name'] == street_name)]

                if not filtered_data.empty and len(filtered_data) > 0:
                    latitude = filtered_data.iloc[0]['latitude']
                    longitude = filtered_data.iloc[0]['longitude']
                    origin.append((latitude, longitude))

                # -----Appending the Latitudes and Longitudes of the MRT Stations-----
                # Latitudes and Longitudes are been appended in the form of a tuple  to that list
                mrt_lat = mrt_location['latitude']
                mrt_long = mrt_location['longitude']
                list_of_mrt_coordinates = []
                for lat, long in zip(mrt_lat, mrt_long):
                    list_of_mrt_coordinates.append((lat, long))

                # -----Getting distance to nearest MRT Stations (Mass Rapid Transit System)-----
                list_of_dist_mrt = []
                for destination in range(0, len(list_of_mrt_coordinates)):
                    list_of_dist_mrt.append(geodesic(origin, list_of_mrt_coordinates[destination]).meters)
                shortest = (min(list_of_dist_mrt))
                min_dist_mrt = shortest
                list_of_dist_mrt.clear()

                # -----Getting distance from CDB (Central Business District)-----
                cbd_dist = geodesic(origin, (1.2830, 103.8513)).meters  # CBD coordinates

                # -----Sending the user enter values for prediction to our model-----
                new_sample = np.array(
                    [[cbd_dist, min_dist_mrt, np.log(floor_area_sqm), lease_remain_years, np.log(storey_median)]])
                new_sample = scaler_loaded.transform(new_sample[:, :5])
                new_pred = loaded_model.predict(new_sample)[0]
                st.write('## :green[Predicted resale price:] ', np.exp(new_pred))

    except Exception as e:
        st.write("Enter the above values to get the predicted resale price of the flat")
