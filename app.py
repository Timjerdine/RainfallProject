import streamlit as st
import pandas as pd
import pickle
import numpy as np

st.set_page_config(page_title="Rainfall Predictor", page_icon="🌧️")

# Charger le modèle
@st.cache_resource
def load_model():
    with open("rainfall_prediction_model.pkl", "rb") as f:
        return pickle.load(f)

try:
    data = load_model()
    model = data["model"]
    feature_names = data["feature_names"]
except Exception as e:
    st.error(f"Erreur de chargement du modèle : {e}")
    st.stop()

st.title("🌧️ Prédiction de Pluie")
st.markdown("Saisissez les paramètres météo ci-dessous.")

with st.form("input_form"):
    col1, col2 = st.columns(2)
    with col1:
        v1 = st.number_input("Pression (pressure)", value=1010.0)
        v2 = st.number_input("Temp Max (maxtemp)", value=25.0)
        v3 = st.number_input("Temp Moyenne (temparature)", value=20.0)
        v4 = st.number_input("Temp Min (mintemp)", value=15.0)
        v5 = st.number_input("Point de rosée (dewpoint)", value=12.0)
    with col2:
        v6 = st.slider("Humidité (humidity)", 0, 100, 60)
        v7 = st.slider("Nuages (cloud)", 0, 100, 40)
        v8 = st.number_input("Ensoleillement (sunshine)", value=6.0)
        v9 = st.number_input("Direction Vent (winddirection)", value=180)
        v10 = st.number_input("Vitesse Vent (windspeed)", value=15.0)
    
    submit = st.form_submit_button("Lancer la prédiction")

if submit:
    # Création du tableau de valeurs dans l'ORDRE EXACT du modèle
    # On ignore les noms de colonnes du dictionnaire pour éviter les KeyError
    values = [v1, v2, v3, v4, v5, v6, v7, v8, v9, v10]
    
    # On crée le DataFrame en imposant les noms de colonnes du modèle
    input_df = pd.DataFrame([values], columns=feature_names)
    
    prediction = model.predict(input_df)
    
    st.divider()
    if prediction[0] == 1:
        st.error("### 🌧️ Résultat : IL VA PLEUVOIR")
    else:
        st.success("### ☀️ Résultat : PAS DE PLUIE")

