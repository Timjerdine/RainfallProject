import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Rainfall Predictor", page_icon="🌧️")

@st.cache_resource
def load_model():
    with open("rainfall_prediction_model.pkl", "rb") as f:
        return pickle.load(f)

data = load_model()
model = data["model"]
# Récupère les colonnes exactes utilisées lors de l'entraînement
features = data["feature_names"]

st.title("🌧️ Prédiction de Pluie")
st.write(f"Veuillez entrer les {len(features)} paramètres ci-dessous :")

user_inputs = {}
# Création dynamique des champs
cols = st.columns(2)
for i, col_name in enumerate(features):
    with cols[i % 2]:
        user_inputs[col_name] = st.number_input(f"{col_name.strip()}", value=0.0)

if st.button("Prédire"):
    # Création du DataFrame avec l'ordre exact des colonnes
    input_df = pd.DataFrame([user_inputs])[features]
    
    prediction = model.predict(input_df)
    
    st.divider()
    if prediction[0] == 1:
        st.error("### Résultat : IL VA PLEUVOIR 🌧️")
    else:
        st.success("### Résultat : PAS DE PLUIE ☀️")


