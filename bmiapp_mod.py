import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np

# Custom CSS for Background and Header Styling
page_bg = """
<style>
    body {
        background-color: #f0f8ff;
    }
    h1 {
        background: linear-gradient(to right, rgba(173, 176, 170, 0.2), rgb(154, 155, 157));
        -webkit-background-clip: text;
        color: transparent;
        text-align: center;
    }
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# App Name with Gradient Effect
st.markdown("<h1>FitPredict</h1>", unsafe_allow_html=True)

# Sidebar Styling
st.sidebar.header("Enter Your Vital Stats")

# Load Dataset
@st.cache_data
def load_data():
    df = pd.read_csv('bmi.csv')  # Replace with your dataset path
    return df

data = load_data()

# Preprocess Data
le_gender = LabelEncoder()
data['Gender'] = le_gender.fit_transform(data['Gender'])

# Binning the 'Index' column into categories
bins = [-1, 0, 1, 2, 3, 4, 5]
health = ['malnourished', 'underweight', 'fit', 'slightly_overweight', 'overweight', 'extremely_overweight']

# Ensure 'Index' is numeric (convert if necessary)
data['Index'] = pd.to_numeric(data['Index'], errors='coerce')

# Apply binning
data['BMI_Category'] = pd.cut(data['Index'], bins=bins, labels=health)

# Drop rows with NaN values resulting from binning
data.dropna(subset=['BMI_Category'], inplace=True)

# Add a synthetic 'Age' column to demonstrate integration
data['Age'] = np.random.randint(18, 60, size=len(data))

# Split Features and Target
X = data[['Weight', 'Height', 'Gender', 'Age']]
y = data['BMI_Category']

# Standardize the Features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM Model with Cross-Validation
model = SVC(probability=True, kernel='linear', random_state=42)
cross_val_scores = cross_val_score(model, X_train, y_train, cv=5)  # 5-fold cross-validation
model.fit(X_train, y_train)

# Remedies for BMI Categories
bmi_remedies = {
    'malnourished': """
    - Eat high-calorie, nutrient-rich foods like proteins, carbs, and vitamins.  
    - Take vitamin or high-energy nutritional supplements.  
    - Severe cases may need hospitalization for feeding and rehydration.  
    - Use feeding tubes or IV nutrition if eating is difficult.  
    - Seek advice from doctors, dietitians, or therapists.  
    """,
    'underweight': """
    - Focus on eating calorie-dense, nutrient-rich foods like:
      - Nuts, avocados, whole grains, and dairy products.
    - Incorporate strength-training exercises to build muscle mass.
    - Avoid skipping meals; maintain a regular eating schedule.
    - Include healthy snacks like smoothies and dried fruits.
    """,
    'fit': """
    - Great job! Keep up your healthy routine.
    - Maintain a balanced diet with proper portions of proteins, carbs, and fats.
    - Continue regular physical activity like walking, jogging, or yoga.
    - Stay hydrated and ensure adequate sleep.
    """,
    'slightly_overweight': """
    - Control portion sizes to reduce excess calorie intake.
    - Increase your consumption of fruits, vegetables, and lean proteins.
    - Avoid sugary drinks and processed foods.
    - Include at least 30 minutes of daily physical activity such as:
      - Walking, cycling, or light cardio workouts.
    """,
    'overweight': """
    - Adopt a calorie-deficit diet focusing on nutrient-dense foods.
    - Engage in regular physical activities like:
      - Walking, swimming, cycling, or aerobics.
    - Limit sugary foods, beverages, and fried items.
    - Practice mindful eating and manage stress.
    """,
    'extremely_overweight': """
    - Seek professional help from a doctor or nutritionist for a guided plan.
    - Adopt a healthy lifestyle with:
      - Low-calorie meals, rich in vegetables and lean proteins.
      - Gradual and consistent physical activity like walking or swimming.
    - Avoid crash diets and focus on sustainable weight loss.
    - Monitor your progress with regular health check-ups.
    """
}

# User Input Section
def user_input_features():
    weight = st.sidebar.text_input('Enter your weight (in kg)')
    height = st.sidebar.text_input('Enter your height (in cm)')
    age = st.sidebar.text_input('Enter your age (in years)')
    gender = st.sidebar.selectbox('Select your gender', ['Male', 'Female'])
    
    # Validate user inputs
    if weight and height and age:
        gender_encoded = le_gender.transform([gender])[0]  # Encode gender using LabelEncoder
        data = {'Weight': float(weight),
                'Height': float(height),
                'Gender': gender_encoded,
                'Age': int(age)}
        features = pd.DataFrame(data, index=[0])
        return features, gender
    else:
        return None, None

# Get user inputs
input_df, gender_input = user_input_features()

if input_df is not None:
    # Scale User Input
    scaled_input = scaler.transform(input_df)

    # Make Prediction
    prediction = model.predict(scaled_input)

    # Display Congratulations Message and Image if Fit
    if prediction[0] == 'fit':
        if gender_input.lower() == 'male':
            st.markdown("<h3 style='color:rgb(174, 178, 175);'>Congratulations, Sir! You are FIT!</h3>", unsafe_allow_html=True)
        else:
            st.markdown("<h3 style='color:rgb(171, 179, 173);'>Congratulations, Ma'am! You are FIT!</h3>", unsafe_allow_html=True)
        
        st.markdown("<p style='color:rgb(183, 187, 184); font-size: 18px;'>Stay Healthy and Keep it Up!</p>", unsafe_allow_html=True)
    
    # Display Prediction Result and Remedies
    st.success(f"Predicted BMI Category: **{prediction[0]}**")

    # Display Remedies
    st.subheader("Suggested Remedies:")
    st.markdown(bmi_remedies[prediction[0]])
else:
    st.warning("Please fill out all fields to get a prediction.")
