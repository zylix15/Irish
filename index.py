import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder

# Load Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
target_names = iris.target_names

# Train Linear Regression model
X = df.drop('target', axis=1)
y = df['target']
model = LinearRegression()
model.fit(X, y)

# Streamlit UI
st.title("Iris Flower Predictor (Linear Regression)")

sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, step=0.1)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, step=0.1)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, step=0.1)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, step=0.1)

if st.button("Predict Iris Type"):
    input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = model.predict(input_data)[0]
    predicted_class = int(round(prediction))

    if 0 <= predicted_class < len(target_names):
        st.success(f"Iris Type: **{target_names[predicted_class].capitalize()}**")
    else:
        st.error("⚠️ Prediction out of expected range. Please check input values.")

