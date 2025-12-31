import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Configuration de la page
st.set_page_config(page_title="Rainfall Predictor", page_icon="🌧️")

# 1. Fonction pour charger le modèle
@st.cache_resource
def load_model():
    # Assurez-vous que le nom du fichier .pkl est EXACTEMENT celui-ci
    with open("rainfall_prediction_model.pkl", "rb") as f:
        data = pickle.load(f)
    return data

try:
    model_data = load_model()
    model = model_data["model"]
    # Les colonnes exactes attendues par votre RandomForest
    feature_names = model_data["feature_names"]
except Exception as e:
    st.error(f"Erreur de chargement du modèle : {e}")
    st.stop()

st.title("🌧️ Prédiction de Pluie")
st.markdown("Saisissez les paramètres météo pour savoir s'il va pleuvoir ou non.")

# 2. Formulaire de saisie
with st.form("my_form"):
    st.subheader("Paramètres Atmosphériques")
    
    col1, col2 = st.columns(2)
    
    with col1:
        pressure = st.number_input("Pression (hPa)", value=1015.0, step=0.1)
        maxtemp = st.number_input("Température Max (°C)", value=22.0, step=0.1)
        temparature = st.number_input("Température Moyenne (°C)", value=18.0, step=0.1)
        mintemp = st.number_input("Température Min (°C)", value=14.0, step=0.1)
        dewpoint = st.number_input("Point de Rosée", value=12.0, step=0.1)

    with col2:
        humidity = st.slider("Humidité (%)", 0, 100, 75)
        cloud = st.slider("Couverture Nuageuse (%)", 0, 100, 50)
        sunshine = st.number_input("Ensoleillement (heures)", value=5.0, step=0.1)
        winddirection = st.number_input("Direction du vent (degrés)", value=180, step=1)
        windspeed = st.number_input("Vitesse du vent (km/h)", value=15.0, step=0.1)

    submit_button = st.form_submit_button(label="Prédire")

# 3. Traitement de la prédiction
if submit_button:
    # On crée le dictionnaire avec les noms EXACTS (espaces inclus)
    input_dict = {
        'pressure ': pressure,           # Notez l'espace après pressure
        'maxtemp': maxtemp,
        'temparature': temparature,
        'mintemp': mintemp,
        'dewpoint': dewpoint,
        'humidity ': humidity,           # Notez l'espace après humidity
        'cloud ': cloud,                 # Notez l'espace après cloud
        'sunshine': sunshine,
        'winddirection': winddirection,  # Vérifiez s'il y a des espaces ici aussi
        'windspeed': windspeed
    }
    
    # Création du DataFrame
    input_df = pd.DataFrame([input_dict])
    
    # Réorganiser les colonnes pour correspondre au modèle
    input_df = input_df[feature_names]
    
    # Prédiction
    prediction = model.predict(input_df)

    st.divider()
    
    if prediction[0] == 1:
        st.error(f"### 🌧️ Résultat : IL VA PLEUVOIR")
        st.write(f"Probabilité de pluie : **{probability[0][1]:.2%}**")
    else:
        st.success(f"### ☀️ Résultat : PAS DE PLUIE")
        st.write(f"Probabilité de ciel sec : **{probability[0][0]:.2%}**")


st.info("Note : Ce modèle utilise un RandomForestClassifier entraîné sur votre dataset Rainfall.")
