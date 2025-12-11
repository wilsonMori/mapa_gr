import folium
from folium.plugins import Draw, Fullscreen
from streamlit_folium import st_folium
import pandas as pd
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import streamlit as st

def render_map(df):
    """
    Mapa inicial con todos los puntos en azul.
    Al hacer clic en un punto se muestra el contrato (sea 'Número de Contrato de Suministro' o 'Contrato').
    """

    m = folium.Map(location=[df['Latitud'].mean(), df['Longitud'].mean()], zoom_start=12)
    Fullscreen().add_to(m)  # Pantalla completa

    # 👉 Detectar columna de contrato
    normalized_cols = {c.lower().replace(" ", ""): c for c in df.columns}
    col_contrato = None
    for norm, original in normalized_cols.items():
        if "contrato" in norm:
            col_contrato = original
            break

    for _, row in df.iterrows():
        if col_contrato and pd.notna(row[col_contrato]) and str(row[col_contrato]).strip() != "":
            contrato_text = f"Contrato: {row[col_contrato]}"
        else:
            contrato_text = "Contrato: Sin dato"

        folium.CircleMarker(
            [row['Latitud'], row['Longitud']],
            radius=5,
            color="blue",
            fill=True,
            popup=contrato_text
        ).add_to(m)

    Draw(export=True).add_to(m)
    return st_folium(m, width=700, height=500)

def render_colored_map(df, color_by="Dia", key=None):
    """
    Mapa coloreado por una columna (por defecto 'Dia').
    Cada categoría tiene su color, se muestra el total de puntos asignados
    y permite dibujar polígonos para edición.
    """

    if color_by not in df.columns:
        st.warning(f"⚠️ La columna '{color_by}' no existe en el DataFrame.")
        return None

    # 👉 Crear mapa centrado en el promedio de coordenadas
    m = folium.Map(location=[df['Latitud'].mean(), df['Longitud'].mean()], zoom_start=12)
    Fullscreen().add_to(m)

    categorias_unicas = sorted(df[color_by].dropna().unique())

    # 👉 Paleta dinámica (más viva: Set1)
    cmap = cm.get_cmap('Set1', len(categorias_unicas))
    colores_map = {cat: mcolors.to_hex(cmap(i)) for i, cat in enumerate(categorias_unicas)}

    # 👉 Detectar columna de contrato
    normalized_cols = {c.lower().replace(" ", ""): c for c in df.columns}
    col_contrato = None
    for norm, original in normalized_cols.items():
        if "contrato" in norm:
            col_contrato = original
            break

    # 👉 Dibujar puntos
    for _, row in df.iterrows():
        valor = row[color_by]
        if pd.isna(valor):
            color = "#FF00FF"  # magenta para nulos
        else:
            color = colores_map.get(valor, "#FF4500")  # naranja fuerte fallback

        if col_contrato and pd.notna(row[col_contrato]) and str(row[col_contrato]).strip() != "":
            contrato_text = f"Contrato: {row[col_contrato]}"
        else:
            contrato_text = "Contrato: Sin dato"

        popup_text = f"{contrato_text} | {color_by}: {valor}"

        folium.CircleMarker(
            [row['Latitud'], row['Longitud']],
            radius=6,
            color=color,
            fill=True,
            popup=popup_text
        ).add_to(m)

    # 👉 Conteo por categoría
    conteo = df[color_by].value_counts().sort_index()

    # 👉 Leyenda scrollable (texto siempre en negro)
    leyenda_html = f"""
    <div style='position: fixed; 
                bottom: 20px; left: 20px; width: 190px; max-height: 300px; 
                overflow-y: auto;
                background-color: #f9f9f9; z-index:9999; 
                padding: 12px; border:2px solid #444; font-size:14px;
                box-shadow: 2px 2px 6px rgba(0,0,0,0.3);'>
        <b style='color:#000000;'>🗂️ Puntos por {color_by}</b><br>
    """
    for cat in categorias_unicas:
        color = colores_map[cat]
        total = conteo[cat]
        leyenda_html += f"""
        <div style='margin-top:6px; font-weight:bold; color:#000000;'>
            <span style='background:{color};width:14px;height:14px;display:inline-block;
                         border-radius:50%;margin-right:6px;'></span>
            {color_by} {cat} — {total} puntos
        </div>
        """

    leyenda_html += "</div>"
    m.get_root().html.add_child(folium.Element(leyenda_html))

    Draw(export=True).add_to(m)
    return st_folium(m, width=700, height=500, key=key)
