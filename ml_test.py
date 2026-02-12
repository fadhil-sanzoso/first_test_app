import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# 1. Page Config
st.set_page_config(page_title="PLN ML Predictor", page_icon="⚙️")

st.title("⚙️ Grid Asset Maintenance Predictor")
st.write("This app uses a **Random Forest** model to predict asset failure risk.")

# 2. GENERATE SAMPLE DATA (To avoid needing an external CSV)
@st.cache_data
def load_data():
    # Creating 100 rows of synthetic PLN asset data
    np.random.seed(42)
    data = pd.DataFrame({
        'age_years': np.random.randint(1, 30, 100),
        'load_percentage': np.random.randint(50, 120, 100),
        'internal_temp': np.random.randint(40, 95, 100),
        'needs_maintenance': np.random.choice([0, 1], 100) # 0=Stable, 1=Repair
    })
    return data
df = load_data()

# 3. TRAIN THE MODEL
# We define features (X) and target (y)
X = df[['age_years', 'load_percentage', 'internal_temp']]
y = df['needs_maintenance']

model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# 4. USER INTERFACE (SIDEBAR)
st.sidebar.header("Input Asset Specs")
user_age = st.sidebar.slider("Asset Age (Years)", 1, 30, 10)
user_load = st.sidebar.slider("Current Load (%)", 0, 150, 85)
user_temp = st.sidebar.slider("Internal Temp (°C)", 30, 100, 65)

# 5. PREDICTION LOGIC
if st.button("Run ML Diagnostic"):
    # Format the user input for the model
    user_input = np.array([[user_age, user_load, user_temp]])
    prediction = model.predict(user_input)
    probability = model.predict_proba(user_input)

    st.divider()
    
    if prediction[0] == 1:
        st.error(f"🔴 **RESULT: MAINTENANCE REQUIRED**")
        st.write(f"Confidence Level: {probability[0][1]:.2%}")
    else:
        st.success(f"🟢 **RESULT: ASSET STABLE**")
        st.write(f"Confidence Level: {probability[0][0]:.2%}")

# Optional: Show the training data so you can see what's happening
if st.checkbox("Show Training Data"):
    st.dataframe(df.head(10))