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
import os

ruta_csv = 'rome_u_journeys.csv'
df = pd.read_csv(ruta_csv)
df.head()

# -------------------------------
# Trip_id
# -------------------------------
# Asegurar tipo datetime
df['tsO'] = pd.to_datetime(df['tsO'], format='%d/%m/%Y %H:%M:%S')

# Ordenar por scooter y fecha
df = df.sort_values(by=['idS', 'tsO'])

# Contador por scooter y día
df['viaje_dia'] = df.groupby(['idS', df['tsO'].dt.date]).cumcount() + 1

# Crear id único (formato: idS + YYMMDD + contador)
df['id_viaje'] = (
    df['idS']
    + df['tsO'].dt.strftime('%y%m%d')
    + df['viaje_dia'].astype(str)
)

# ------------------------------------------------------------
# Función batch para consultar altitudes
# ------------------------------------------------------------
def obtener_altitudes_batch(lista_coordenadas, batch_size=300, reintentos=3):
    """
    Recibe una lista de coordenadas [(lat, lon), (lat, lon), ...]
    Devuelve una lista de altitudes en el mismo orden.
    """
    altitudes = []

    # Recorrer en lotes
    for i in tqdm(range(0, len(lista_coordenadas), batch_size), desc="Obteniendo altitudes"):
        batch = lista_coordenadas[i : i + batch_size]

        # Crear string para API
        loc_str = "|".join([f"{lat},{lon}" for lat, lon in batch])
        url = f"https://api.open-elevation.com/api/v1/lookup?locations={loc_str}"

        # Sistema de reintentos
        for intento in range(reintentos):
            try:
                response = requests.get(url, timeout=10)
                data = response.json()

                if "results" in data:
                    altitudes.extend([res["elevation"] for res in data["results"]])
                else:
                    altitudes.extend([None] * len(batch))

                time.sleep(0.2)  # para no saturar la API
                break

            except Exception:
                print(f"Error en batch {i} intento {intento+1}. Reintentando...")
                time.sleep(1)

        else:
            # Si fallan todos los reintentos
            altitudes.extend([None] * len(batch))

    return altitudes

# ------------------------------------------------------------
# Función principal diff_altura
# ------------------------------------------------------------
def diff_altura(df):
    """
    Añade únicamente la columna diff_altura al DataFrame usando consultas batch.
    """
    print("Preparando coordenadas...")

    coords_origen = list(zip(df["latO"], df["lonO"]))
    coords_destino = list(zip(df["latD"], df["lonD"]))

    print("Obteniendo altitudes de ORIGEN...")
    altO = obtener_altitudes_batch(coords_origen)

    print("Obteniendo altitudes de DESTINO...")
    altD = obtener_altitudes_batch(coords_destino)

    # Solo añadimos diff_altura, sin guardar altO ni altD
    df["diff_altura"] = [d - o for d, o in zip(altD, altO)]

    print("Columna diff_altura añadida correctamente.")

    return df

df = diff_altura(df)

# ------------------------------------------------------------
# Pendiente media (%)
# ------------------------------------------------------------
df['pendiente'] = (df['diff_altura'] / df['dis']) * 100

# 1. Velocidades > 25 km/h
out_vel = df[df["vel"] > 25]
print(f"Registros con velocidad > 25 km/h: {len(out_vel)}")

# 2. Viajes con tt > 3 horas  (3h = 10800 s)
out_tt_muy_largo = df[df["tt"] > 10800]
print(f"Registros con tt > 3 horas: {len(out_tt_muy_largo)}")

# 3. Viajes con tt < 30 s
out_tt_muy_corto = df[df["tt"] < 30]
print(f"Registros con tt < 30 segundos: {len(out_tt_muy_corto)}")

# 4. Viajes fuera de Roma
out_coords = df[
    (df["latO"] < 41.78) | (df["latO"] > 41.99) |
    (df["latD"] < 41.78) | (df["latD"] > 41.99) |
    (df["lonO"] < 12.38) | (df["lonO"] > 12.63) |
    (df["lonD"] < 12.38) | (df["lonD"] > 12.63)
]

print(f"Trayectos fuera de Roma: {len(out_coords)}")

# 5. Viajes con distancia <5m y tiempo >90s
out_dist_cero = df[(df["dis"] < 5) & (df["tt"] > 90)]
print(f"Distancia casi 0 pero tiempo alto: {len(out_dist_cero)}")

# 6. Imprimir TOTAL
total_outliers = len(out_vel) + len(out_tt_muy_largo) + len(out_tt_muy_corto) + len(out_coords) + len(out_dist_cero)
print(f"\nTotal de registros marcados como outliers: {total_outliers}")


# Creamos una copia limpia
df_clean = df.copy()

# 1. Velocidad válida (<= 25 km/h)
df_clean = df_clean[df_clean["vel"] <= 25]

# 2. Tiempo válido (>= 30s y <= 3h)
df_clean = df_clean[(df_clean["tt"] >= 30) & (df_clean["tt"] <= 10800)]

# 3. Coordenadas dentro del bounding box de Roma
df_clean = df_clean[
    (df_clean["latO"] >= 41.78) & (df_clean["latO"] <= 41.99) &
    (df_clean["latD"] >= 41.78) & (df_clean["latD"] <= 41.99) &
    (df_clean["lonO"] >= 12.38) & (df_clean["lonO"] <= 12.63) &
    (df_clean["lonD"] >= 12.38) & (df_clean["lonD"] <= 12.63)
]

# 4. Distancia mínima lógica (>= 5 m) si el viaje dura más de 90s
df_clean = df_clean[~((df_clean["dis"] < 5) & (df_clean["tt"] > 90))]

# Mostrar resultados
print("Datos eliminados correctamente.")
print(f"Filas originales: {len(df)}")
print(f"Filas después de limpiar: {len(df_clean)}")
print(f"Filas eliminadas: {len(df) - len(df_clean)}")


# ------------------------------------------------------------
# Guardar df_clean en CSV
# ------------------------------------------------------------

# Nombre del archivo de salida
nombre_salida = "df_clean.csv"

# Ruta absoluta de la carpeta donde está este script
ruta_actual = os.path.dirname(os.path.abspath(__file__))

# Construir la ruta completa al archivo de salida
ruta_salida = os.path.join(ruta_actual, nombre_salida)

# Guardar el DataFrame limpio
df_clean.to_csv(ruta_salida, index=False)

print(f"Archivo guardado correctamente en: {ruta_salida}")

