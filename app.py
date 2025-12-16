# ================================================================
# IMPORTS Y CONFIGURACIÓN INICIAL
# ================================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import folium
import math
import requests
from tqdm import tqdm
import time
from sklearn.metrics import silhouette_score
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import hdbscan
from geopy.distance import geodesic
from sklearn.neighbors import BallTree
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import pydeck as pdk
import altair as alt

# ================================================================
# CONFIGURACIÓN STREAMLIT
# ================================================================
st.set_page_config(
    page_title="Proyecto Estaciones",
    page_icon="🚉",
    layout="wide"  
)

st.markdown(
    """
    <style>
    .block-container {
        max-width: 60%;
        margin-left: auto;
        margin-right: auto;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ================================================================
# CARGA Y PREPROCESADO DE DATOS
# ================================================================

ruta_csv = 'df_clean.csv'
df_clean = pd.read_csv(ruta_csv)
df_clean.head()

df_clean['tsO'] = pd.to_datetime(df_clean['tsO'])
df_clean['tsD'] = pd.to_datetime(df_clean['tsD'], dayfirst=True, errors='coerce')


# ================================================================
# MAPA DE DENSIDAD DE VIAJES
# ================================================================

# Asegurar que las columnas de coordenadas son numéricas
df_clean["latO"] = pd.to_numeric(df_clean["latO"], errors="coerce")
df_clean["lonO"] = pd.to_numeric(df_clean["lonO"], errors="coerce")

st.title("📍 Mapa de densidad de viajes ")

# Crear capa hexagonal
hex_layer = pdk.Layer(
    "HexagonLayer",
    data=df_clean,
    get_position=["lonO", "latO"],  # orden: lon, lat
    radius=100,                     # radio en metros
    elevation_scale=4,
    elevation_range=[0, 1000],
    pickable=True,
    extruded=True,
)

# Configuración de la vista inicial centrada en Roma
view_state = pdk.ViewState(
    latitude=41.9,
    longitude=12.5,
    zoom=12,
    pitch=40,
)

# Crear el gráfico
deck = pdk.Deck(
    layers=[hex_layer],
    initial_view_state=view_state,
    tooltip={"text": "Viajes: {elevationValue}"}
)

# Mostrar en Streamlit
st.pydeck_chart(deck)


# ================================================
# HDBSCAN               
# ================================================

st.title("🚀 Configuración del modelo")

min_dist_m = st.slider("Radio de la estacion (m)", 100, 1000, 400, 50)
min_cluster_size = st.slider("Tamaño mínimo de cluster", 5, 200, 60, 5)
min_samples = st.slider("Muestras mínimas por cluster", 0, 100, 15, 5)

# Combina origen y destino en un solo array de coordenadas
coords_origen = df_clean[['latO', 'lonO']].dropna()
coords_destino = df_clean[['latD', 'lonD']].dropna()
coords = pd.concat([coords_origen, coords_destino]).drop_duplicates().reset_index(drop=True)
coords_array = coords[['latO', 'lonO']].values if 'latO' in coords else coords.values

coords_radians = np.radians(coords_array)

# Parámetros iniciales recomendados
clusterer = hdbscan.HDBSCAN(
    min_cluster_size,     # Tamaño de los clusters
    min_samples,
    metric='haversine'
)
labels = clusterer.fit_predict(coords_radians)
coords['cluster'] = labels
clusters_validos = coords[coords['cluster'] != -1]

# Combina origen y destino como puntos independientes
centroides = []
for cluster_id in clusters_validos['cluster'].unique():
    cluster = clusters_validos[clusters_validos['cluster'] == cluster_id]

    # Toma todos los puntos de origen y destino como filas separadas
    puntos_origen = cluster[['latO', 'lonO']].dropna().values
    puntos_destino = cluster[['latD', 'lonD']].dropna().values
    puntos = np.vstack([puntos_origen, puntos_destino])

    # Calcula el centroide
    centroide = np.mean(puntos, axis=0)
    centroides.append(centroide)

centroides = np.array(centroides)

def filtra_centroides(centroides, min_dist):
    seleccionados = []
    for centro in centroides:
        if all(geodesic(tuple(centro), tuple(sel)).m >= min_dist for sel in seleccionados):
            seleccionados.append(tuple(centro))
    return np.array(seleccionados)

estaciones_finales = filtra_centroides(centroides, min_dist_m)
# Filtra NaNs de coordenadas de origen
origen_validos = coords[['latO', 'lonO']].dropna()


# Convertimos estaciones a array (lat, lon)
est_array = np.array(estaciones_finales)

# Convertir a radianes para BallTree
est_rad = np.radians(est_array)


# ================================================================
# ASIGNACIÓN DE ESTACIONES DE ORIGEN Y DESTINO
# ================================================================

# Convertimos estaciones a array (lat, lon)
est_array = np.array(estaciones_finales)

# Convertir a radianes para BallTree
est_rad = np.radians(est_array)

# Construir BallTree
tree = BallTree(est_rad, metric='haversine')

# Convertir lat/lon de viajes a radianes
viajes_origen_rad = np.radians(df_clean[['latO', 'lonO']].values)
viajes_destino_rad = np.radians(df_clean[['latD', 'lonD']].values)

# Buscar estación más cercana al origen y destino
dist_o, idx_o = tree.query(viajes_origen_rad, k=1)
dist_d, idx_d = tree.query(viajes_destino_rad, k=1)

# Convertir a metros
dist_o_m = dist_o[:,0] * 6371000
dist_d_m = dist_d[:,0] * 6371000

# Guardar estación de origen y destino en el dataframe
df_clean['estacion_origen'] = idx_o[:,0]
df_clean['dist_origen_m'] = dist_o_m

df_clean['estacion_destino'] = idx_d[:,0]
df_clean['dist_destino_m'] = dist_d_m

# Identificar viajes que empiezan y terminan en la misma estación
df_clean['mismo_origen_destino'] = (df_clean['estacion_origen'] == df_clean['estacion_destino']).astype(int)

# Filtrar viajes dentro del rango (origen o destino dentro del radio)
df_clean['dist_min_m'] = np.minimum(dist_o_m, dist_d_m)
df_cubiertos = df_clean[df_clean['dist_min_m'] <= min_dist_m]

# ================================================================
# MÉTRICAS POR ESTACIÓN Y DÍA
# ================================================================

# Día de la semana
df_cubiertos['dia_semana'] = df_cubiertos['tsO'].dt.dayofweek

# Conteo de viajes por estación de destino y día (para dimensionar carga)
viajes_por_estacion_y_dia = (
    df_cubiertos
    .groupby(['estacion_destino', 'dia_semana'])
    .size()
    .unstack(fill_value=0)
)

# Columna total de viajes por estación
viajes_por_estacion_y_dia['total_viajes'] = viajes_por_estacion_y_dia.sum(axis=1)

# Columna de media diaria de viajes
viajes_por_estacion_y_dia['mean_day'] = viajes_por_estacion_y_dia['total_viajes'] / 28

# Columna de puntos de carga
def calcular_slots(media):
    if media < 10:
        return 10
    elif media < 20:
        return 15
    else:
        return 20

viajes_por_estacion_y_dia['puntos_carga'] = viajes_por_estacion_y_dia['mean_day'].apply(calcular_slots)

viajes_por_estacion_y_dia.head()


# Viaje cubierto si origen o destino están dentro del radio
viaje_cubierto = (dist_o_m <= min_dist_m) | (dist_d_m <= min_dist_m)
cubiertos = viaje_cubierto.sum()
cubiertos_origen = (dist_o_m <= min_dist_m).sum()
cubiertos_destino = (dist_d_m <= min_dist_m).sum()
total_viajes = len(df_clean)

# Distancia mínima a estación para viajes NO cubiertos
dist_no_cubiertos = np.minimum(dist_o_m, dist_d_m)[~viaje_cubierto]
dist_media_no_cubiertos = dist_no_cubiertos.mean() if len(dist_no_cubiertos) > 0 else 0


# ================================================================
# DATAFRAME DE ESTACIONES
# ================================================================

# Media de viajes
media_viajes = viajes_por_estacion_y_dia["mean_day"]

# Slots de carga por estación
slots = viajes_por_estacion_y_dia['puntos_carga']

# Tabla de estaciones limpia
df_estaciones = pd.DataFrame({
    "estacion_id": media_viajes.index,
    "lat": [estaciones_finales[i][0] for i in media_viajes.index],
    "lon": [estaciones_finales[i][1] for i in media_viajes.index],
    "media_viajes_dia": media_viajes.values,
    "puntos_carga": slots.values
})

# ================================================================
# NUEVAS MÉTRICAS DE VIAJES POR ESTACIÓN
# ================================================================

# Viajes cuyo origen está dentro del radio de acción
viajes_origen = pd.Series((dist_o_m <= min_dist_m).astype(int), index=idx_o[:,0]).groupby(level=0).sum()

# Viajes cuyo destino está dentro del radio de acción
viajes_destino = pd.Series((dist_d_m <= min_dist_m).astype(int), index=idx_d[:,0]).groupby(level=0).sum()

# Viajes que empiezan y terminan en la misma estación
viajes_mismo_mask = (idx_o[:,0] == idx_d[:,0]) & (dist_o_m <= min_dist_m) & (dist_d_m <= min_dist_m)
viajes_mismo = pd.Series(viajes_mismo_mask.astype(int), index=idx_o[:,0]).groupby(level=0).sum()

# ------------------------------------------------
# NUEVAS MÉTRICAS
# ------------------------------------------------

# Total de viajes cubiertos (origen o destino dentro del radio)
viajes_totales_cubiertos = (viajes_origen.add(viajes_destino, fill_value=0)).astype(int)

# Máximo de viajes en la misma hora de un mismo día por estación (usando destino)
max_viajes_mismo_momento = (
    df_clean[df_clean['dist_min_m'] <= min_dist_m]  # solo viajes cubiertos
    .groupby(['estacion_origen', df_clean['tsO'].dt.floor('H')])
    .size()
    .groupby('estacion_origen')
    .max()
)

# ------------------------------------------------
# MERGE FINAL
# ------------------------------------------------
df_extra = pd.DataFrame({
    "estacion_id": df_estaciones["estacion_id"],
    "viajes_origen": df_estaciones["estacion_id"].map(viajes_origen).fillna(0).astype(int),
    "viajes_destino": df_estaciones["estacion_id"].map(viajes_destino).fillna(0).astype(int),
    "viajes_mismo": df_estaciones["estacion_id"].map(viajes_mismo).fillna(0).astype(int),
    "viajes_totales_cubiertos": df_estaciones["estacion_id"].map(viajes_totales_cubiertos).fillna(0).astype(int),
    "max_viajes_mismo_momento": df_estaciones["estacion_id"].map(max_viajes_mismo_momento).fillna(0).astype(int),
})

df_estaciones = df_estaciones.merge(df_extra, on="estacion_id", how="left")


# ================================================================
# CÁLCULO DE FLOTA DE PATINETES POR ESTACIÓN
# ================================================================
margen_pct = 0.20  # 20% adicional
margen_min_abs = 0 # siempre al menos 1 patinete extra

def calcular_flota(max_viajes, puntos_carga):
    # 1 + 20% del máximo, redondeado hacia arriba
    extra = math.ceil(margen_min_abs + (max_viajes * margen_pct))
    flota = max_viajes + extra
    # no llenar todos los slots
    return min(flota, puntos_carga - 1)

df_estaciones['flota_patinetes'] = df_estaciones.apply(
    lambda row: calcular_flota(row['max_viajes_mismo_momento'], row['puntos_carga']),
    axis=1
).astype(int)


# ================================================================
# MAPA DE ESTACIONES GENERADAS
# ================================================================

st.title("🚉 Mapa de estaciones generadas")

col_mapa, col_datos = st.columns([2, 1])  # proporción 2:1
with col_mapa:
# Crear capa de puntos
    scatter_layer = pdk.Layer(
        "ScatterplotLayer",
        data=df_estaciones,
        get_position=["lon", "lat"],   # orden: lon, lat
        get_color=[255, 0, 0],         # color rojo
        get_radius=50,                 # radio en metros
        pickable=True,
    )

    # Configuración de la vista inicial centrada en Roma
    view_state = pdk.ViewState(
        latitude=41.9,
        longitude=12.5,
        zoom=12,
        pitch=0,
    )

    # Crear el gráfico
    deck = pdk.Deck(
        layers=[scatter_layer],
        initial_view_state=view_state,
        tooltip={"text": "Estación en ({lat}, {lon})"}
    )

    # Mostrar en Streamlit
    st.pydeck_chart(deck)

with col_datos:
    st.write(" COBERTURA DEL MODELO ")
    st.write(f"Total de estaciones: {len(estaciones_finales)}")
    st.write(f"Radio usado: {min_dist_m} m\n")

    st.write(f"Viajes totales: {total_viajes}")
    st.write(f"Viajes cubiertos: {cubiertos}  ({cubiertos/total_viajes*100:.2f}%)\n")
    st.write(f"Cobertura de ORIGEN:  {cubiertos_origen/total_viajes*100:.2f}%")
    st.write(f"Cobertura de DESTINO: {cubiertos_destino/total_viajes*100:.2f}%\n")
    st.write(f"Distancia media de viajes NO cubiertos a la estación más cercana: {dist_media_no_cubiertos:.2f} m")


# ================================================
# ENERGÍA (CORREGIDA Y COHERENTE)
# ================================================

# ---------- Constantes físicas ----------
masa_total = 22.5 + 75          # kg (patinete + usuario)
g = 9.81                        # gravedad (m/s²)
capacidad_bateria_Wh = 576      # Wh
potencia_motor = 250            # W
dias_mes_equiv = 28

# ---------- Factor dinámico de bajada ----------
k = 0.05
min_factor = 0.05

def factor_bajada_dinamico(grade_pct):
    if grade_pct >= 0:
        return 1.0
    return max(min_factor, 1 - k * abs(grade_pct))


# ---------- Energía por viaje (J) ----------
E_motor_J = potencia_motor * df_clean['tt']

df_clean['factor_bajada'] = df_clean['pendiente'].apply(factor_bajada_dinamico)

# Energía potencial SOLO en subida
E_pot_J = (
    masa_total
    * g
    * np.maximum(0, df_clean['diff_altura'])
    * df_clean['factor_bajada']
)

# Energía total (sin negativos)
E_total_J = E_motor_J + E_pot_J


# ---------- Conversión a Wh ----------
df_clean['energia_Wh'] = E_total_J / 3600
df_clean['porcentaje_bateria'] = (
    df_clean['energia_Wh'] / capacidad_bateria_Wh * 100
)


# ================================================
# VIAJES CUBIERTOS (UNA SOLA VEZ)
# ================================================

viaje_cubierto = (dist_d_m <= min_dist_m)
df_cubiertos = df_clean.loc[viaje_cubierto].copy()

if 'dia_semana' not in df_cubiertos.columns:
    df_cubiertos['dia_semana'] = df_cubiertos['tsO'].dt.dayofweek


# ================================================================
# CÁLCULO DE ENERGÍA POR ESTACIÓN (agrupando por estacion_id)
# ================================================================

# Totales sin escalar (kWh)
total_kWh_dataset = df_cubiertos['energia_Wh'].sum() / 1000

# Agrupar energía por estación destino pero renombrando a estacion_id
energia_por_estacion_raw = (
    df_cubiertos
    .groupby('estacion_destino', as_index=False)['energia_Wh']
    .sum()
    .rename(columns={'estacion_destino': 'estacion_id', 'energia_Wh': 'energia_total_Wh'})
)

total_kWh_estaciones_raw = energia_por_estacion_raw['energia_total_Wh'].sum() / 1000

# ========= Cálculo de factor de escala mensual =========
dias_cubiertos = df_cubiertos['tsO'].dt.normalize().nunique()
escala_mensual = 28 / dias_cubiertos if dias_cubiertos and dias_cubiertos > 0 else 1.0

# ========= Aplicar escala UNA sola vez =========
energia_por_estacion = energia_por_estacion_raw.copy()
energia_por_estacion['energia_total_Wh'] = energia_por_estacion['energia_total_Wh'] * escala_mensual
energia_por_estacion['energia_kWh'] = energia_por_estacion['energia_total_Wh'] / 1000

# ========= Ajustes energéticos extra (sumados a columnas existentes) =========
eta_cargador = 0.90              # eficiencia del cargador
wh_reposo_por_pat_dia = 3.0      # Wh/día por patinete en reposo
potencia_parasita_estacion_w = 15.0  # W constantes por estación
dias_mes_equiv = 28              # coherente con tu escalado mensual

# Ineficiencia del cargador
energia_por_estacion['energia_kWh'] = energia_por_estacion['energia_kWh'] / eta_cargador

# Reposo de patinetes
energia_por_estacion['energia_kWh'] += (df_estaciones['flota_patinetes'] * wh_reposo_por_pat_dia * dias_mes_equiv) / 1000

# Consumo parásito fijo por estación
energia_por_estacion['energia_kWh'] += (potencia_parasita_estacion_w * 24 * dias_mes_equiv) / 1000

# Actualizar Wh en coherencia
energia_por_estacion['energia_total_Wh'] = energia_por_estacion['energia_kWh'] * 1000

# ========= Integración con df_estaciones (sin crear columna extra) =========
df_estaciones = df_estaciones.merge(
    energia_por_estacion,
    on="estacion_id",
    how="left"
)

df_estaciones["energia_total_Wh"] = df_estaciones["energia_total_Wh"].fillna(0)
df_estaciones["energia_kWh"] = df_estaciones["energia_total_Wh"] / 1000


# ================================================
# ECONOMÍA
# ================================================

def calcular_capex(tipo, df_estaciones):
    """Calcula el CAPEX neto para un tipo de estación."""
    c_fijo = TIPOS_ESTACION[tipo]["coste_fijo"]
    c_slot = TIPOS_ESTACION[tipo]["coste_slot"]
    coste_estacion = c_fijo + df_estaciones["puntos_carga"] * c_slot
    coste_patinetes = df_estaciones["flota_patinetes"] * COSTE_PATINETE
    capex_bruto = (coste_estacion + coste_patinetes).sum()
    ayuda = AYUDAS_ESTADO[tipo]
    return round(capex_bruto * (1 - ayuda), 2)

def calcular_opex_mensual(tipo, df_estaciones):
    """Calcula el OPEX mensual para un tipo de estación."""
    if tipo == "Electrica":
        coste_energia = (df_estaciones["energia_kWh"] * COSTE_ENERGIA_KWH).sum()
    elif tipo == "Solar":
        coste_energia = 0
    elif tipo == "Mixta":
        coste_energia = (df_estaciones["energia_kWh"] * COSTE_ENERGIA_KWH * 0.5).sum()
    else:
        raise ValueError("Tipo desconocido")

    coste_mantenimiento = len(df_estaciones) * COSTE_MANTENIMIENTO_MENSUAL
    ciclos_estimados = df_estaciones["energia_kWh"] / 0.576
    coste_baterias = ((ciclos_estimados / CICLOS_BATERIA) * COSTE_BATERIA).sum()

    return round(coste_energia + coste_mantenimiento + coste_baterias, 2)

def serie_creciente_mensual(base_mensual, meses, tasa_mensual):
    factores = (1 + tasa_mensual) ** np.arange(meses)
    return np.round(base_mensual * factores, 2)

def ingresos_mensuales(meses, ganancia_por_viaje, viajes_diarios, crecer=True, tasa=0.0):
    base = round(ganancia_por_viaje * viajes_diarios * 30, 2)  # aproximación: 30 días/mes
    if crecer:
        return serie_creciente_mensual(base, meses, tasa)
    else:
        return np.round(np.full(meses, base), 2)

def calcular_capex_detallado(tipo, df_estaciones):
    """Devuelve desglose de CAPEX por tipo de estación."""
    c_fijo = TIPOS_ESTACION[tipo]["coste_fijo"]
    c_slot = TIPOS_ESTACION[tipo]["coste_slot"]

    coste_estacion = (c_fijo + df_estaciones["puntos_carga"] * c_slot).sum()
    num_patinetes = int(df_estaciones["flota_patinetes"].sum())
    coste_patinetes = (df_estaciones["flota_patinetes"] * COSTE_PATINETE).sum()

    capex_bruto = coste_estacion + coste_patinetes
    ayuda = AYUDAS_ESTADO[tipo]
    capex_neto = capex_bruto * (1 - ayuda)

    return {
        "Construcción estación": round(coste_estacion, 2),
        f"Compra patinetes ({num_patinetes})": round(coste_patinetes, 2),
        "Subvención estatal": round(capex_bruto * ayuda, 2),
        "CAPEX neto": round(capex_neto, 2)
    }

def calcular_opex_detallado(tipo, df_estaciones):
    """Devuelve desglose de OPEX mensual por tipo de estación."""
    if tipo == "Electrica":
        coste_energia = (df_estaciones["energia_kWh"] * COSTE_ENERGIA_KWH).sum()
    elif tipo == "Solar":
        coste_energia = 0
    elif tipo == "Mixta":
        coste_energia = (df_estaciones["energia_kWh"] * COSTE_ENERGIA_KWH * 0.5).sum()
    else:
        raise ValueError("Tipo desconocido")

    coste_mantenimiento = len(df_estaciones) * COSTE_MANTENIMIENTO_MENSUAL
    ciclos_estimados = df_estaciones["energia_kWh"] / 0.576
    coste_baterias = ((ciclos_estimados / CICLOS_BATERIA) * COSTE_BATERIA).sum()

    total = coste_energia + coste_mantenimiento + coste_baterias

    return {
        "Energía": round(coste_energia, 2),
        "Mantenimiento": round(coste_mantenimiento, 2),
        "Sustitución baterías": round(coste_baterias, 2),
        "OPEX mensual": round(total, 2)
    }


# ================================================================
# Parámetros
# ================================================================

st.title("⚙️ Configuración estaciones")

COSTE_PATINETE = st.slider("Coste por patinete (€)", 100, 1000, 300, 10)
COSTE_ENERGIA_KWH = st.slider("Coste energía (€/kWh)", 0.05, 0.50, 0.20, 0.01)
TIPO_ESTACION_USADA = st.selectbox("Tipo de estación", ["Electrica", "Solar", "Mixta"])
AÑOS = st.slider("Años de análisis", 1, 20, 10, 1)
COSTE_MANTENIMIENTO_MENSUAL = st.slider("Coste mantenimiento mensual por estación (€)", 50, 500, 200, 10)
COSTE_BATERIA = st.slider("Coste por batería (€)", 50, 500, 150, 10)
CICLOS_BATERIA = st.slider("Ciclos antes de sustitución de batería", 200, 2000, 500, 50)

GANANCIA_POR_VIAJE = df_clean["price"].mean()    # € por viaje
INGRESOS_CRECIENTE = True                  # activar crecimiento de ingresos
TASA_CRECIMIENTO_INGRESOS_MENSUAL = 0.0025  # 0.5% mensual (Alto)
TASA_CRECIMIENTO_OPEX_MENSUAL = 0.004      # 0.6% mensual (Alto)
MESES = 12 * AÑOS

# Tipos de estación con costes base
TIPOS_ESTACION = {
    "Electrica": {"coste_fijo": 25000, "coste_slot": 1000},
    "Solar": {"coste_fijo": 40000, "coste_slot": 1500},
    "Mixta": {"coste_fijo": 30000, "coste_slot": 1200}
}

# Selección del tipo de estación
COSTE_FIJO = TIPOS_ESTACION[TIPO_ESTACION_USADA]["coste_fijo"]
COSTE_SLOT = TIPOS_ESTACION[TIPO_ESTACION_USADA]["coste_slot"]

CICLOS_BATERIA = 1000  # ciclos antes de sustitución

AYUDAS_ESTADO = {
    "Electrica": 0.10,  # 10% subvención sobre inversión inicial
    "Solar": 0.30,      # 30% subvención (más apoyo a renovables)
    "Mixta": 0.20       # 20% subvención
}

st.subheader("📊 Datos de la estación")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("🏭 Tipo de estación", TIPO_ESTACION_USADA)
    st.metric("💰 Coste fijo", f"{COSTE_FIJO} €")
    st.metric("🔌 Coste por punto de carga", f"{COSTE_SLOT} €")

with col2:
    st.metric("🛴 Coste patinete", f"{COSTE_PATINETE} €")
    st.metric("⚡ Coste energía", f"{COSTE_ENERGIA_KWH} €/kWh")
    st.metric("🛠️ Mantenimiento mensual", f"{COSTE_MANTENIMIENTO_MENSUAL} €")

with col3:
    st.metric("🔋 Coste batería", f"{COSTE_BATERIA} €")
    st.metric("♻️ Ciclos batería", f"{CICLOS_BATERIA}")
    st.metric("🎁 Subvención estatal", f"{AYUDAS_ESTADO[TIPO_ESTACION_USADA]*100:.0f}%")


# ================================================================
# RESULTADOS
# ================================================================

capex_total_neto = calcular_capex(TIPO_ESTACION_USADA, df_estaciones)
opex_mensual_total = calcular_opex_mensual(TIPO_ESTACION_USADA, df_estaciones)

# CAPEX detallado
capex_detalle = calcular_capex_detallado(TIPO_ESTACION_USADA, df_estaciones)
capex_total = capex_detalle["CAPEX neto"]

# OPEX detallado
opex_detalle = calcular_opex_detallado(TIPO_ESTACION_USADA, df_estaciones)
opex_total = opex_detalle["OPEX mensual"]

# Quitar totales para calcular porcentajes por partida
capex_partidas = {k: v for k, v in capex_detalle.items() if k != "CAPEX neto"}
opex_partidas = {k: v for k, v in opex_detalle.items() if k != "OPEX mensual"}

# CAPEX limpio (sin porcentaje)
capex_mostrar = {k: f"{v:,.2f} €" for k, v in capex_partidas.items()}

# OPEX con porcentaje
opex_mostrar = {k: f"{v:,.2f} € ({(v/opex_total*100):.1f}%)" for k, v in opex_partidas.items()}

st.title("💸 Economía del sistema")

col_capex, col_opex = st.columns(2)

with col_capex:
    st.subheader("🏗️ CAPEX (Inversión inicial)")
    for k, v in capex_mostrar.items():
        st.markdown(f"- **{k}**: {v}")
    st.markdown(
        f"**Total CAPEX neto:** <span style='color:red'>{capex_total:,.2f} €</span>",
        unsafe_allow_html=True
    )

with col_opex:
    st.subheader("🔧 OPEX (Gasto mensual)")
    for k, v in opex_mostrar.items():
        st.markdown(f"- **{k}**: {v}")
    st.markdown(
        f"**Total OPEX mensual:** <span style='color:red'>{opex_total:,.2f} €</span>",
        unsafe_allow_html=True
    )



# ================================================================
# Cálculos financieros base (CAPEX y OPEX) reutilizando variables
# ================================================================

# Viajes diarios totales estimados
viajes_diarios_totales = round(viajes_por_estacion_y_dia["mean_day"].sum(), 2)

# ===========================
# Cálculos a X años
# ===========================

ingresos_mes = ingresos_mensuales(
    MESES,
    GANANCIA_POR_VIAJE,
    viajes_diarios_totales,
    crecer=INGRESOS_CRECIENTE,
    tasa=TASA_CRECIMIENTO_INGRESOS_MENSUAL
)

opex_mes = serie_creciente_mensual(
    opex_mensual_total,
    MESES,
    TASA_CRECIMIENTO_OPEX_MENSUAL
)

ingresos_acum = np.round(ingresos_mes.cumsum(), 2)
opex_acum = np.round(opex_mes.cumsum(), 2)
coste_total_acum = np.round(capex_total_neto + opex_acum, 2)
rentabilidad_acum = np.round(ingresos_acum - coste_total_acum, 2)

ingreso_mensual_base = ingresos_mes[0]   # primer mes
ingreso_anual_base = ingreso_mensual_base * 12
beneficio_mensual = ingreso_mensual_base - opex_total


st.subheader("💵 Ingresos")
st.markdown(f"- **Ingreso mensual estimado:** {ingreso_mensual_base:,.2f} €")
st.markdown(f"- **Ingreso anual estimado:** {ingreso_anual_base:,.2f} €")
color = "green" if beneficio_mensual >= 0 else "red"
st.markdown(f"- **Beneficio mensual:** <span style='color:{color}'>{beneficio_mensual:,.2f} €</span>", unsafe_allow_html=True)

# ================================================================
# Gráfico de beneficio acumulado SOLO para el tipo elegido
# ================================================================

registros = []
for tipo in TIPOS_ESTACION.keys():
    capex_neto = calcular_capex(tipo, df_estaciones)
    opex_mensual = calcular_opex_mensual(tipo, df_estaciones)
    ingresos_mes = ingresos_mensuales(
        MESES,
        GANANCIA_POR_VIAJE,
        viajes_diarios_totales,
        crecer=INGRESOS_CRECIENTE,
        tasa=TASA_CRECIMIENTO_INGRESOS_MENSUAL
    )
    opex_mes = serie_creciente_mensual(opex_mensual, MESES, TASA_CRECIMIENTO_OPEX_MENSUAL)

    # Beneficio mensual acumulado
    rentabilidad_acum = ingresos_mes.cumsum() - (capex_neto + opex_mes.cumsum())

    # Tomamos el valor al final de cada año
    for año in range(1, AÑOS+1):
        idx = año*12 - 1
        registros.append({
            "Año": año,
            "Tipo": tipo.capitalize(),
            "Beneficio": float(rentabilidad_acum[idx])
        })

# Convertir a DataFrame
df_bar = pd.DataFrame(registros)

# Pivotar para que cada tipo sea una columna y los años el índice
df_bar_pivot = df_bar.pivot(index="Año", columns="Tipo", values="Beneficio")

# Ordenar por año explícitamente
df_bar_pivot = df_bar_pivot.sort_index()

# Filtrar SOLO el tipo de estación elegido
tipo_seleccionado = TIPO_ESTACION_USADA.capitalize()
df_bar_filtrado = df_bar_pivot[[tipo_seleccionado]]

# Mostrar gráfico en Streamlit
st.subheader(f"📊 Beneficio neto acumulado por año ({tipo_seleccionado})")
st.bar_chart(df_bar_filtrado)

# ================================================================
# Gráfico financiero acumulado con Streamlit
# ================================================================

# Crear DataFrame con columna Mes
df_finanzas = pd.DataFrame({
    "Mes": np.arange(1, MESES+1),
    "Ingresos acumulados": ingresos_acum,
    "Coste acumulado": coste_total_acum,
    "Rentabilidad acumulada": rentabilidad_acum
})

# Pasar a formato largo para que Altair pueda pintar varias series
df_long = df_finanzas.melt(id_vars="Mes", var_name="Serie", value_name="Euros")

# Gráfico con líneas (sin relleno)
chart = alt.Chart(df_long).mark_line().encode(
    x=alt.X("Mes:O", title="Mes"),
    y=alt.Y("Euros:Q", title="Euros (€)"),
    color=alt.Color("Serie:N", title="Concepto"),
    tooltip=["Mes", "Serie", alt.Tooltip("Euros:Q", format=",.2f")]
).properties(
    title=f"📈 Rentabilidad del sistema a {AÑOS} años (Estación {TIPO_ESTACION_USADA})"
)

# Mostrar en Streamlit
st.subheader(f"📈 Rentabilidad del sistema a {AÑOS} años (Estación {TIPO_ESTACION_USADA})")
st.altair_chart(chart, use_container_width=True)

# Calcular y mostrar el punto de equilibrio
mes_equilibrio = np.argmax(rentabilidad_acum > 0) + 1
st.write(f"📍 El sistema alcanza rentabilidad positiva en el mes {mes_equilibrio}.")

# ================================================================
# Gráfico mensual con etiquetas claras en el hover
# ================================================================

# Crear DataFrame con columna Mes
df_mensual = pd.DataFrame({
    "Mes": np.arange(1, MESES+1),
    "Ingresos mensuales": ingresos_mes,
    "Costes mensuales": opex_mes
})

# Pasar a formato largo para que Altair pueda pintar varias series
df_long = df_mensual.melt(id_vars="Mes", var_name="Concepto", value_name="Euros")

# Gráfico de líneas
chart = alt.Chart(df_long).mark_line(point=True).encode(
    x=alt.X("Mes:O", title="Mes"),
    y=alt.Y("Euros:Q", title="Euros (€)"),
    color=alt.Color("Concepto:N", title="Concepto"),
    tooltip=["Mes", "Concepto", alt.Tooltip("Euros:Q", format=",.2f")]
).properties(
    title="📊 Ingresos vs Costes mensuales"
)

# Mostrar en Streamlit
st.subheader("📊 Ingresos vs Costes mensuales")
st.altair_chart(chart, use_container_width=True)

