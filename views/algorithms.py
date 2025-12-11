import pandas as pd
from sklearn.cluster import KMeans
import streamlit as st
import numpy as np
 
from sklearn.metrics import pairwise_distances
from geopy.distance import geodesic

def aplicar_algoritmo(df, algoritmo, n_clusters, columna="Dia"):
    """
    Aplica un algoritmo de asignación sobre df.
    - algoritmo: nombre del algoritmo (Zona, Proximidad, Preciso, Capacitado, Sweep, Secuencial, KMeans)
    - n_clusters: número de días o técnicos
    - columna: 'Dia' o 'Tecnico'
    """
    df = df.copy()

    if algoritmo == "Por zona":
        df = asignar_por_zona(df, n_clusters)
    elif algoritmo == "Por proximidad":
        df = distribucion_por_proximidad(df, n_clusters)
    elif algoritmo == "Balanceado Preciso":
        df = asignar_balanceado_preciso(df, n_clusters)
    elif algoritmo == "Capacitado":
        df = asignar_capacitado(df, n_clusters)
    elif algoritmo == "Sweep":
        df = asignar_sweep(df, n_clusters)
    elif algoritmo == "kms":
        cantidades = [len(df)//n_clusters] * n_clusters
        df = asignar_por_kmeans(df, cantidades)

    # 👉 Si la columna destino no es 'Dia', copiar resultado
    if columna != "Dia":
        df[columna] = df["Dia"]

    return df

def asignar_por_zona(df, n_dias, random_state=42):
    """
    Asigna puntos por zona geográfica usando KMeans.
    Cada zona se interpreta como un día.
    """
    df = df.copy()

    if len(df) < n_dias:
        df["Dia"] = 0
        return df

    coords = df[["Latitud", "Longitud"]].values
    kmeans = KMeans(n_clusters=n_dias, random_state=random_state)
    labels = kmeans.fit_predict(coords)

    df["Dia"] = labels.astype(int)
    return df

def distribucion_por_proximidad(df, n_dias):
    """
    Distribuye puntos por proximidad geográfica usando KMeans.
    Cada cluster se interpreta como un día.
    """
    df = df.copy()
    coords = df[['Latitud', 'Longitud']].values
    kmeans = KMeans(n_clusters=n_dias, random_state=42, n_init=10)
    df['Dia'] = kmeans.fit_predict(coords)

    # Ordenar clusters para que los días sean consistentes
    orden_clusters = (
        df.groupby("Dia")[["Latitud", "Longitud"]]
        .mean()
        .sort_values(by=["Latitud", "Longitud"])
        .index.tolist()
    )
    mapping = {cluster: i for i, cluster in enumerate(orden_clusters)}
    df['Dia'] = df['Dia'].map(mapping)

    df["Dia"] = df["Dia"].astype(int)
    return df

def asignar_balanceado_preciso(df, n_dias, max_iter=100):
    """
    Balanced KMeans con reasignación de sobrantes.
    - Cada día recibe casi la misma cantidad de puntos.
    - Los sobrantes se distribuyen entre los días existentes.
    """
    df = df.copy()
    coords = df[['Latitud', 'Longitud']].values
    n_points = len(coords)
    target_size = n_points // n_dias
    extra = n_points % n_dias  # puntos sobrantes

    # Inicializar centroides aleatorios
    centroides = coords[np.random.choice(n_points, n_dias, replace=False)]
    asignaciones = np.full(n_points, -1)

    for _ in range(max_iter):
        dist = pairwise_distances(coords, centroides)
        asignaciones[:] = -1
        usados = set()

        for dia in range(n_dias):
            candidatos = np.argsort(dist[:, dia])
            count = 0
            limite = target_size + (1 if dia < extra else 0)  # distribuir sobrantes
            for idx in candidatos:
                if asignaciones[idx] == -1 and count < limite:
                    asignaciones[idx] = dia
                    usados.add(idx)
                    count += 1

        # Recalcular centroides
        for dia in range(n_dias):
            centroides[dia] = coords[asignaciones == dia].mean(axis=0)

    # Si aún quedan puntos sin asignar (-1), ponerlos en el cluster más cercano
    for idx in np.where(asignaciones == -1)[0]:
        distancias = pairwise_distances([coords[idx]], centroides)[0]
        asignaciones[idx] = np.argmin(distancias)

    df['Dia'] = asignaciones.astype(int)
    return df

def asignar_capacitado(df, n_dias):
    """
    Opción 2: Capacitated Clustering (Capacitated Voronoi).
    - Cada cluster tiene una capacidad máxima (puntos por día).
    - Los puntos se asignan al centroide más cercano, pero respetando la capacidad.
    """
    df = df.copy()
    coords = df[['Latitud', 'Longitud']].values
    n_points = len(coords)
    capacidad = n_points // n_dias

    # Inicializar centroides aleatorios
    centroides = coords[np.random.choice(n_points, n_dias, replace=False)]
    asignaciones = np.full(n_points, -1)

    for idx, punto in enumerate(coords):
        distancias = [geodesic(punto, c).meters for c in centroides]
        orden = np.argsort(distancias)
        for dia in orden:
            if (asignaciones == dia).sum() < capacidad:
                asignaciones[idx] = dia
                break

    df['Dia'] = asignaciones
    return df

def asignar_sweep(df, n_dias, esquina="NO"):
    """
    Sweep con agrupamiento espacial:
    - Recorre desde una esquina.
    - Agrupa por cercanía.
    - Asigna bloques sin cruces.
    """
    df = df.copy()
    n_points = len(df)
    target_size = n_points // n_dias
    extra = n_points % n_dias

    # Ordenar puntos según esquina
    if esquina == "NO":
        df = df.sort_values(by=["Latitud", "Longitud"], ascending=[False, True])
    elif esquina == "NE":
        df = df.sort_values(by=["Latitud", "Longitud"], ascending=[False, False])
    elif esquina == "SO":
        df = df.sort_values(by=["Latitud", "Longitud"], ascending=[True, True])
    elif esquina == "SE":
        df = df.sort_values(by=["Latitud", "Longitud"], ascending=[True, False])

    df["Dia"] = -1
    usados = set()
    coords = df[["Latitud", "Longitud"]].values

    for dia in range(n_dias):
        limite = target_size + (1 if dia < extra else 0)
        candidatos = df[~df.index.isin(usados)].copy()

        if len(candidatos) <= limite:
            df.loc[candidatos.index, "Dia"] = dia
            break

        # Tomar punto inicial del bloque
        punto_inicio = candidatos.iloc[0][["Latitud", "Longitud"]].values
        candidatos["distancia"] = candidatos.apply(
            lambda r: geodesic(punto_inicio, (r["Latitud"], r["Longitud"])).meters,
            axis=1
        )

        seleccionados = candidatos.nsmallest(limite, "distancia")
        df.loc[seleccionados.index, "Dia"] = dia
        usados.update(seleccionados.index)

    return df


def asignacion_manual(df, cantidades, n_sectores=None):
    """
    Asigna puntos por sectores verticales compactos y centrados, evitando dispersión.

    Parámetros:
    - df: DataFrame con columnas Latitud y Longitud
    - cantidades: lista con la cantidad esperada por día
    - n_sectores: número de sectores verticales (por defecto igual al número de días)

    Retorna:
    - DataFrame con columna 'Dia' asignada, sin puntos en -1
    """
    df = df.copy()
    df["Dia"] = -1
    usados = set()

    if n_sectores is None:
        n_sectores = len(cantidades)

    min_lon, max_lon = df["Longitud"].min(), df["Longitud"].max()
    cortes = np.linspace(min_lon, max_lon, n_sectores + 1)

    dia = 0
    for i in range(n_sectores):
        if dia >= len(cantidades):
            break

        sector = df[
            (df["Longitud"] >= cortes[i]) & (df["Longitud"] < cortes[i + 1]) & (~df.index.isin(usados))
        ]

        if sector.empty:
            continue

        # 👉 Calcular centroide del sector
        centro_lat = sector["Latitud"].mean()
        centro_lon = sector["Longitud"].mean()

        # 👉 Calcular distancia al centroide
        sector["distancia"] = np.sqrt(
            (sector["Latitud"] - centro_lat) ** 2 + (sector["Longitud"] - centro_lon) ** 2
        )

        # 👉 Seleccionar puntos más cercanos
        seleccionados = sector.sort_values(by="distancia").head(cantidades[dia])
        df.loc[seleccionados.index, "Dia"] = dia
        usados.update(seleccionados.index)
        dia += 1

    df.drop(columns=["distancia"], errors="ignore", inplace=True)

    # 👉 Redistribuir puntos no asignados
    df = redistribuir_sobrantes(df, cantidades)

    df["Dia"] = df["Dia"].astype(int)
    return df

def redistribuir_sobrantes(df, cantidades):
    """
    Redistribuye puntos no asignados (Dia = -1) a días con espacio disponible.

    Parámetros:
    - df: DataFrame con columna 'Dia'
    - cantidades: lista con la cantidad esperada por día

    Retorna:
    - DataFrame con todos los puntos asignados
    """
    df = df.copy()
    no_asignados = df[df["Dia"] == -1]
    asignados = df[df["Dia"] != -1]

    conteo_actual = asignados["Dia"].value_counts().to_dict()

    for dia, esperado in enumerate(cantidades):
        faltan = esperado - conteo_actual.get(dia, 0)
        if faltan > 0 and not no_asignados.empty:
            nuevos = no_asignados.head(faltan)
            df.loc[nuevos.index, "Dia"] = dia
            no_asignados = no_asignados.drop(nuevos.index)

    df["Dia"] = df["Dia"].astype(int)
    return df
#
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd

def asignar_por_kmeans(df, cantidades, max_iter=100):
    df = df.copy()
    df["Dia"] = -1
    n_dias = len(cantidades)

    # 👉 KMeans inicial
    coords = df[["Latitud", "Longitud"]].values
    kmeans = KMeans(n_clusters=n_dias, n_init=10, max_iter=max_iter, random_state=42)
    df["cluster"] = kmeans.fit_predict(coords)

    # 👉 Asignar por cercanía al centroide de cada cluster
    usados = set()
    for dia in range(n_dias):
        grupo = df[df["cluster"] == dia].copy()
        if grupo.empty:
            continue
        c_lat, c_lon = grupo["Latitud"].mean(), grupo["Longitud"].mean()
        grupo["dist"] = np.sqrt((grupo["Latitud"] - c_lat)**2 + (grupo["Longitud"] - c_lon)**2)
        seleccionados = grupo.sort_values("dist").head(cantidades[dia])
        df.loc[seleccionados.index, "Dia"] = dia
        usados.update(seleccionados.index)

    # 👉 Redistribuir sobrantes
    df = redistribuir_sobrantes(df, cantidades)
    return df


def redistribuir_sobrantes(df, cantidades):
    df = df.copy()
    no_asignados = df[df["Dia"] == -1]
    if no_asignados.empty:
        return df

    # Centroide por día
    centroides = {}
    for dia in range(len(cantidades)):
        g = df[df["Dia"] == dia]
        if g.empty:
            centroides[dia] = (df["Latitud"].mean(), df["Longitud"].mean())
        else:
            centroides[dia] = (g["Latitud"].mean(), g["Longitud"].mean())

    # Conteo actual
    counts = df[df["Dia"] != -1]["Dia"].value_counts().to_dict()

    # Asignar sobrantes por cercanía
    for dia, esperado in enumerate(cantidades):
        falta = esperado - counts.get(dia, 0)
        if falta <= 0 or no_asignados.empty:
            continue
        c_lat, c_lon = centroides[dia]
        tmp = no_asignados.copy()
        tmp["dist"] = np.sqrt((tmp["Latitud"] - c_lat)**2 + (tmp["Longitud"] - c_lon)**2)
        mov = tmp.sort_values("dist").head(falta)
        df.loc[mov.index, "Dia"] = dia
        no_asignados = no_asignados.drop(mov.index)

    return df
