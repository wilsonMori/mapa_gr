import pandas as pd
import re
import streamlit as st

def procesar_coordenadas(coordenada_str):
    if pd.isna(coordenada_str):
        return None, None
    coordenada_str = str(coordenada_str).strip()
    if not coordenada_str:
        return None, None

    coordenada_str = re.sub(r'\s*,\s*', ' ', coordenada_str)
    coordenada_str = re.sub(r'\s+', ' ', coordenada_str)
    partes = coordenada_str.split()

    if len(partes) >= 2:
        try:
            lat = float(partes[0])
            lon = float(partes[1])
            return lat, lon
        except ValueError:
            return None, None
    return None, None

def extraer_coordenadas(df):
    st.info("Procesando coordenadas...")
    columnas_lower = {col.lower(): col for col in df.columns}

    if 'coordenadas' in columnas_lower:
        col_real = columnas_lower['coordenadas']
        coords = df[col_real].apply(procesar_coordenadas)
        df['Latitud'] = [c[0] for c in coords]
        df['Longitud'] = [c[1] for c in coords]
    elif 'ubicación' in columnas_lower:
        col_real = columnas_lower['ubicación']
        coords = df[col_real].apply(procesar_coordenadas)
        df['Latitud'] = [c[0] for c in coords]
        df['Longitud'] = [c[1] for c in coords]
    elif 'latitud' in columnas_lower and 'longitud' in columnas_lower:
        lat_col = columnas_lower['latitud']
        lon_col = columnas_lower['longitud']
        df['Latitud'] = pd.to_numeric(df[lat_col], errors='coerce')
        df['Longitud'] = pd.to_numeric(df[lon_col], errors='coerce')
    else:
        raise ValueError("❌ No se encontraron columnas de coordenadas válidas")

    original_count = len(df)
    df = df.dropna(subset=['Latitud', 'Longitud'])
    validos = len(df)

    st.success(f"✅ Puntos válidos: {validos}/{original_count}")
    if original_count - validos > 0:
        st.warning(f"⚠️ Se eliminaron {original_count - validos} registros inválidos")

    return df.reset_index(drop=True)
