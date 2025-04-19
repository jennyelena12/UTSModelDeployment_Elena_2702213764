# Elena Nathanielle Budiman Angkawi - 2702213764
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import gdown

# setting the app
st.set_page_config(
    page_title="UTS Model Deployment Hotel Booking Status Prediction",
    layout="wide"
)

# header
st.title("Elena - 2702213764 - Hotel Booking Status Prediction")
st.write("Enter booking details to predict if the booking will be canceled or not")

# load Random Forest model (pickle)
# Load the saved Random Forest model from Google Drive
@st.cache_resource
def load_model():
    try:
        if not os.path.exists("random_forest_model.pkl"):
            with st.spinner():
                file_id = "1cnkRVwQ4-8zyi381TAWh5Yj1pPdrmGny"
                url = f"https://drive.google.com/uc?id={file_id}"
                output = "random_forest_model.pkl"
                gdown.download(url, output, quiet=False)
        with open("random_forest_model.pkl", 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# function to encode input data
def preprocess_input(data):
    data_processed = data.copy()
    
    meal_plan_mapping = {
        'Meal Plan 1': 1, 
        'Meal Plan 2': 2, 
        'Meal Plan 3': 3, 
        'Not Selected': -1
    }
    
    room_type_mapping = {
        'Room_Type 1': 0,
        'Room_Type 2': 1,
        'Room_Type 3': 2,
        'Room_Type 4': 3,
        'Room_Type 5': 4,
        'Room_Type 6': 5,
        'Room_Type 7': 6
    }
    
    market_segment_mapping = {
        'Aviation': 0,
        'Complementary': 1,
        'Corporate': 2,
        'Offline': 3,
        'Online': 4
    }
    
    # error validation
    try:
        data_processed['type_of_meal_plan'] = data_processed['type_of_meal_plan'].map(meal_plan_mapping)
        data_processed['room_type_reserved'] = data_processed['room_type_reserved'].map(room_type_mapping)
        data_processed['market_segment_type'] = data_processed['market_segment_type'].map(market_segment_mapping)
    except Exception as e:
        st.error(f"Mapping error: {e}")
        st.write("Available meal plans:", list(meal_plan_mapping.keys()))
        st.write("Available room types:", list(room_type_mapping.keys()))
        st.write("Available market segments:", list(market_segment_mapping.keys()))
    
    return data_processed

# input form
with st.form("booking_form"):
    col1, col2 = st.columns(2)
    with col1:
        adults = st.number_input("Number of Adults", min_value=0)
        children = st.number_input("Number of Children", min_value=0)
        weekend_nights = st.number_input("Number of Weekend Nights", min_value=0)
        week_nights = st.number_input("Number of Week Nights", min_value=0)
        meal_plan = st.selectbox(
            "Type of Meal Plan",
            options=['Meal Plan 1', 'Meal Plan 2', 'Meal Plan 3', 'Not Selected']
        )
        car_parking = st.selectbox("Required Car Parking Space", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        room_type = st.selectbox(
            "Room Type Reserved",
            options=['Room_Type 1', 'Room_Type 2', 'Room_Type 3', 'Room_Type 4', 'Room_Type 5', 'Room_Type 6', 'Room_Type 7']
        )
        lead_time = st.number_input("Lead Time (days)", min_value=0, value=30)
        market_segment = st.selectbox(
            "Market Segment Type",
            options=['Online', 'Offline', 'Corporate', 'Aviation', 'Complementary']
        )
    
    with col2:
        arrival_year = st.number_input("Arrival Year", min_value=2017, max_value=2018)
        arrival_month = st.number_input("Arrival Month", min_value=1, max_value=12, value=1)
        arrival_date = st.number_input("Arrival Date", min_value=1, max_value=31, value=15)
        repeated_guest = st.selectbox("Repeated Guest", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        prev_cancellations = st.number_input("Number of Previous Cancellations", min_value=0, value=0)
        prev_bookings_not_canceled = st.number_input("Number of Previous Bookings Not Canceled", min_value=0, value=0)
        avg_price = st.number_input("Average Price Per Room", min_value=0.0, value=100.0)
        special_requests = st.number_input("Number of Special Requests", min_value=0, value=0)
    
    submit_button = st.form_submit_button("Predict Booking Status")


if submit_button and model is not None:
    input_data = {
        'no_of_adults': adults,
        'no_of_children': children,
        'no_of_weekend_nights': weekend_nights,
        'no_of_week_nights': week_nights,
        'type_of_meal_plan': meal_plan,
        'required_car_parking_space': car_parking,
        'room_type_reserved': room_type,
        'lead_time': lead_time,
        'arrival_year': arrival_year,
        'arrival_month': arrival_month,
        'arrival_date': arrival_date,
        'market_segment_type': market_segment,
        'repeated_guest': repeated_guest,
        'no_of_previous_cancellations': prev_cancellations,
        'no_of_previous_bookings_not_canceled': prev_bookings_not_canceled,
        'avg_price_per_room': avg_price,
        'no_of_special_requests': special_requests
    }
    
    input_df = pd.DataFrame([input_data])
    processed_input = preprocess_input(input_df)
    
    # debug info
    st.write("Process Input data after preprocessing:")
    st.write(processed_input)
    
    try:
        prediction = model.predict(processed_input)
        probability = model.predict_proba(processed_input)
 
        st.subheader("Prediction Result")
        
        if prediction[0] == 0:
            st.error(f"📉 This booking is predicted to be **CANCELED** with {probability[0][0]*100:.2f}% probability.")
        else:
            st.success(f"✅ This booking is predicted to be **NOT CANCELED** with {probability[0][1]*100:.2f}% probability.")
            
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        st.write(f"Exception details: {str(e)}")


with st.expander("About this app"):
    st.write("""
    This app uses a Random Forest model trained on hotel booking data to predict whether a booking will be canceled or not.
    
    **Features used in prediction:**
    - Number of adults and children
    - Number of weekend and weekday nights
    - Type of meal plan
    - Car parking requirements
    - Room type reserved
    - Lead time before arrival
    - Arrival date information
    - Market segment type
    - Whether the guest has stayed before
    - Previous booking history
    - Average price per room
    - Number of special requests
    """)