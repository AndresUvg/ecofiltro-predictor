import streamlit as st
import joblib
import numpy as np
import pandas as pd
import requests
from datetime import date

st.set_page_config(page_title="Predictor Ecofiltro", page_icon="💧", layout="wide")

@st.cache_resource
def cargar_modelo():
    modelo = joblib.load("modelo_ecofiltro.pkl")
    imputer = joblib.load("imputer_ecofiltro.pkl")
    columnas_raw = joblib.load("columnas_ecofiltro.pkl")
    columnas = columnas_raw.tolist() if hasattr(columnas_raw, "tolist") else list(columnas_raw)
    return modelo, imputer, columnas

modelo, imputer, columnas = cargar_modelo()

def obtener_clima(fecha):
    try:
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": 14.5197, "longitude": -90.7589,
            "start_date": str(fecha), "end_date": str(fecha),
            "daily": ["temperature_2m_max","temperature_2m_min",
                      "precipitation_sum","windspeed_10m_max"],
            "timezone": "America/Guatemala"
        }
        r = requests.get(url, params=params).json()
        return {
            "temperature_2m_max": r["daily"]["temperature_2m_max"][0] or 22.0,
            "temperature_2m_min": r["daily"]["temperature_2m_min"][0] or 14.0,
            "precipitation_sum": r["daily"]["precipitation_sum"][0] or 0.0,
            "windspeed_10m_max": r["daily"]["windspeed_10m_max"][0] or 7.0
        }
    except:
        return {"temperature_2m_max": 22.0, "temperature_2m_min": 14.0,
                "precipitation_sum": 0.0, "windspeed_10m_max": 7.0}

st.title("💧 Predictor de Tasa de Filtracion — Ecofiltro")
st.markdown("Ingrese los datos del lote para predecir la tasa de filtracion esperada.")
st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🪨 Composicion del Barro")
    limite_liquido = st.number_input("Limite liquido (%)", 42.0, 54.0, 50.4, 0.1)
    indice_plastico = st.number_input("Indice de plasticidad", 14.0, 20.0, 16.5, 0.1)
    arcilla = st.number_input("Arcilla (%)", 29.0, 42.0, 36.9, 0.1)
    arena = st.number_input("Arena (%)", 14.0, 23.0, 18.6, 0.1)
    limo = st.number_input("Limo (%)", 41.0, 51.0, 44.7, 0.1)
    barro_humedad = st.number_input("Humedad del barro (%)", 4.0, 6.0, 4.9, 0.1)
    barro_peso = st.number_input("Peso del barro (lb)", 60.0, 75.0, 75.0, 0.5)

with col2:
    st.subheader("🪵 Aserrin y Mezcla")
    humedad = st.number_input("Humedad aserrin (%)", 4.0, 5.0, 5.0, 0.1)
    peso = st.number_input("Peso aserrin (lb)", 10.0, 29.0, 26.0, 0.5)
    mayor_2mm = st.number_input("Granulometria >2mm (%)", 0.0, 1.0, 0.0, 0.01)
    entre_2_y_05mm = st.number_input("Granulometria 0.5-2mm (%)", 70.0, 77.0, 74.5, 0.1)
    menor_05mm = st.number_input("Granulometria <0.5mm (%)", 23.0, 30.0, 25.4, 0.1)
    fm_humedad = st.number_input("Humedad formulacion (%)", 4.0, 5.0, 5.0, 0.1)
    fm_peso = st.number_input("Peso formulacion (lb)", 10.0, 29.0, 26.0, 0.5)
    aserrinLB = st.number_input("Libras de aserrin", 10.0, 29.0, 26.0, 0.5)
    barroLB = st.number_input("Libras de barro", 60.0, 75.0, 75.0, 0.5)

with col3:
    st.subheader("🔥 Proceso")
    horno = st.selectbox("Horno", ["Horno 1","Horno 2","Horno 3","Horno 4","Horno 5"])
    horno_map = {"Horno 1":0,"Horno 2":1,"Horno 3":2,"Horno 4":3,"Horno 5":4}
    turno = st.selectbox("Turno", ["Dia","Noche"])
    turno_map = {"Dia":0,"Noche":1}
    grupo = st.selectbox("Grupo de produccion", ["Grupo 1","Grupo 2"])
    grupo_map = {"Grupo 1":0,"Grupo 2":1}

    st.subheader("📊 Resultado del Horneado")
    Temperatura_horno = st.number_input("Temperatura horneado (C)", 23.0, 830.0, 733.0, 1.0)
    porcentajeAprobado = st.number_input("% Aprobados en horno", 0.0, 100.0, 75.6, 0.5)
    altos = st.number_input("Filtros altos en horno", 0, 151, 3, 1)
    bajos = st.number_input("Filtros bajos en horno", 0, 219, 15, 1)
    rajados = st.number_input("Filtros rajados en horno", 0, 44, 8, 1)
    aprobados = st.number_input("Filtros aprobados en horno", 0, 274, 135, 1)
    temp_tunel = st.number_input("Temperatura promedio tunel (C)", 20.0, 95.0, 82.0, 1.0)

    st.subheader("📐 Dimensiones del Crudo")
    diametro = st.number_input("Diametro (cm)", 32.7, 33.0, 32.9, 0.1)
    alturaH1 = st.number_input("Altura H1 (cm)", 26.0, 26.5, 26.3, 0.1)
    alturaH2 = st.number_input("Altura H2 (cm)", 26.0, 26.6, 26.3, 0.1)
    grosor1 = st.number_input("Grosor 1 (mm)", 17.9, 19.9, 18.9, 0.1)
    grosor2 = st.number_input("Grosor 2 (mm)", 17.7, 19.8, 18.8, 0.1)
    grosorFondo = st.number_input("Grosor fondo (mm)", 15.4, 20.8, 19.1, 0.1)
    pesouf = st.number_input("Peso UF (lb)", 16.2, 17.9, 16.7, 0.1)

    st.subheader("📅 Fecha de produccion")
    fecha_prod = st.date_input("Fecha", value=date.today())

st.divider()

if st.button("🔮 Predecir Tasa de Filtracion", type="primary", use_container_width=True):
    clima = obtener_clima(fecha_prod)

    datos = {
        'limite_liquido': float(limite_liquido),
        'indice_plastico': float(indice_plastico),
        'arcilla': float(arcilla),
        'arena': float(arena),
        'limo': float(limo),
        'barro_humedad': float(barro_humedad),
        'barro_peso': float(barro_peso),
        'humedad': float(humedad),
        'peso': float(peso),
        'mayor_2mm': float(mayor_2mm),
        'entre_2_y_05mm': float(entre_2_y_05mm),
        'menor_05mm': float(menor_05mm),
        'fm_humedad': float(fm_humedad),
        'fm_peso': float(fm_peso),
        'Temperatura_horno': float(Temperatura_horno),
        'porcentajeAprobado_horno': float(porcentajeAprobado),
        'altos_horno': float(altos),
        'bajos_horno': float(bajos),
        'rajadosCC_horno': float(rajados),
        'Aprobados_horno': float(aprobados),
        'temp_tunel_promedio': float(temp_tunel),
        'diametro': float(diametro),
        'alturaH1': float(alturaH1),
        'alturaH2': float(alturaH2),
        'grosor1': float(grosor1),
        'grosor2': float(grosor2),
        'grosorFondo': float(grosorFondo),
        'pesouf': float(pesouf),
        'barroLB': float(barroLB),
        'aserrinLB': float(aserrinLB),
        'temperature_2m_max': float(clima['temperature_2m_max']),
        'temperature_2m_min': float(clima['temperature_2m_min']),
        'precipitation_sum': float(clima['precipitation_sum']),
        'windspeed_10m_max': float(clima['windspeed_10m_max']),
        'Horno': int(horno_map[horno]),
        'nombre_turno': int(turno_map[turno]),
        'grupoProd': int(grupo_map[grupo])
    }

    df_pred = pd.DataFrame([datos]).reindex(columns=columnas)
    df_imp = imputer.transform(df_pred)
    tasa_pred = max(100, float(modelo.predict(df_imp)[0]))

    st.subheader("📊 Resultado de la Prediccion")
    col_r1, col_r2, col_r3 = st.columns(3)

    with col_r1:
        st.metric("Tasa predicha", f"{tasa_pred:.0f} ml/hora")

    with col_r2:
        if 800 <= tasa_pred <= 1600:
            st.success("✅ RANGO ACEPTABLE (800-1,600 ml/h)")
        elif tasa_pred < 800:
            st.error(f"⚠️ BAJO — {800-tasa_pred:.0f} ml/h por debajo del minimo")
        else:
            st.warning(f"⚠️ ALTO — {tasa_pred-1600:.0f} ml/h por encima del maximo")

    with col_r3:
        st.info(f"🌤️ Clima: {clima['temperature_2m_max']:.1f}C max | "
                f"Lluvia: {clima['precipitation_sum']:.1f}mm | "
                f"Viento: {clima['windspeed_10m_max']:.1f} km/h")

    st.markdown("---")
    st.caption("Modelo: XGBoost optimizado | R2=0.40 | RMSE=331 ml/h | "
               "Datos: Oct 2025 - May 2026 | Ciudad Vieja, Sacatepequez")
