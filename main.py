import streamlit as st

import joblib 
import pandas as pd
@st.cache_resource
def load_model():
    return joblib.load("Mobiles_price.joblib")

model = None

try:
    model = load_model()
except Exception as e:
    st.error(f"Model loading error: {e}")

df = pd.read_csv("Flipkart_Mobiles.csv")
df = df.dropna()

st.title("Mobile Price Prediction")
st.image("MOBILE_Photo.jpeg", width=1000)
st.sidebar.header("Dataset Information")

st.sidebar.write("Total Mobiles:", df.shape[0])
st.sidebar.write("Total Brands:", df['Brand'].nunique())
st.sidebar.write("Total Models:", df['Model'].nunique())


def load_css():
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()




brand = st.selectbox("Select Brand", sorted(df['Brand'].unique()))
df_brand = df[df['Brand'] == brand]

m = st.selectbox("Select Model", sorted(df_brand['Model'].unique()))
df_model = df_brand[df_brand['Model'] == m]

color = st.selectbox("Select Color", df_model['Color'].unique())
df_color = df_model[df_model['Color'] == color]

memory = st.selectbox("Select Ram", df_color['Memory'].unique())
df_memory = df_color[df_color['Memory'] == memory]

storage = st.selectbox("Select Storage", df_memory['Storage'].unique())
rating = st.slider("Select Rating",1.0, 5.0, 4.0, 0.2)

if st.button("Predict Price"):

    input_df = pd.DataFrame({ 
        "Brand":[brand],
        "Model":[m],
        "Color":[color],
        "Memory":[memory],
        "Storage":[storage],
        "Rating":[rating]
    })

    st.markdown(f"""
    <div class="mobile-card">
    <b>Brand:</b> {brand} <br>
    <b>Model:</b> {m} <br>
    <b>Color:</b> {color} <br>
    <b>RAM:</b> {memory} <br>
    <b>Storage:</b> {storage} <br>
    <b>Rating:</b> {rating}
    </div>
    """, unsafe_allow_html=True)

    prediction = model.predict(input_df)

    st.markdown(f"""
    <div class="price-text">
    💰 Predicted Price: ₹ {prediction[0]:,.0f}
    </div>
    """, unsafe_allow_html=True)
st.markdown("---")

st.markdown('<p class="main-title">📱 Mobile Price Prediction</p>', unsafe_allow_html=True)
st.markdown("""
### 🔎 Predict Smartphone Prices Using Machine Learning

Select mobile specifications and click **Predict Price** to estimate the expected market price.
""")

import matplotlib.pyplot as plt

st.markdown("### 📊 Mobile Price Distribution")

fig, ax = plt.subplots()
ax.hist(df["Selling Price"], bins=30)
ax.set_xlabel("Price")
ax.set_ylabel("Count")
ax.set_title("Mobile Price Distribution")

st.pyplot(fig)

st.markdown("""
---
<center>
Developed by <b>Ghanshyam Kumar</b> | ML Project
</center>
""", unsafe_allow_html=True)