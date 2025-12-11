import streamlit as st
import pandas as pd
import io
from models.points_model import PointsModel
from views.map_view import render_colored_map
from shapely.geometry import Point, Polygon
from controllers.dias_controller import DiasController
from controllers.tecnico_controller import TecnicosController
from views.algorithms import (
    asignar_por_zona,
    distribucion_por_proximidad,
    asignar_balanceado_preciso,   # ✅ nuevo
    asignar_capacitado,
    asignar_sweep,
    asignacion_manual,
    asignar_por_kmeans
)
from views.prueba import asignar_por_kmeans_evolutivo
class PointsController:
    def __init__(self, df):
        if "df" not in st.session_state:
            st.session_state["df"] = df
        if "cambios_guardados" not in st.session_state:
            st.session_state["cambios_guardados"] = False
        if "algoritmo_aplicado" not in st.session_state:
            st.session_state["algoritmo_aplicado"] = False
        self.model = PointsModel(st.session_state["df"])
    
    def run(self):
        st.title("Planificación de Técnicos GR")
        modo = st.radio(
            "Selecciona vista de planificación",
            ["Por días", "Por técnicos"],
            key="modo_selector"
        )

        if modo == "Por días":
            self.run_por_dias()
        elif modo == "Por técnicos":
            self.run_por_tecnicos()

    # ------------------------------
    # Vista por días (flujo completo)
    # ------------------------------
    def run_por_dias(self):
        st.title("Vista por días")

        dias_ctrl = DiasController(st.session_state["df"])
        n_dias = dias_ctrl.solicitar_numero_dias()

        if "n_dias_anterior" not in st.session_state or st.session_state["n_dias_anterior"] != n_dias:
            st.session_state["algoritmo_aplicado"] = False
            st.session_state["n_dias_anterior"] = n_dias

        if n_dias > 0:
            cantidades = dias_ctrl.asignar_puntos_por_dia()

            if cantidades is not None:
                st.success("Distribución de puntos validada correctamente ✅")

                algoritmo = st.selectbox(
                    "Seleccione algoritmo de asignación",
                    [
                        "Por zona",
                        "Por proximidad",
                        "Balanceado Preciso",
                        "Capacitado",
                        "Sweep",
                        "Manual",
                        "kms",
                        "kms-evolutivo"
                    ],
                    key="algoritmo_selector"
                )

                if "algoritmo_anterior" not in st.session_state or st.session_state["algoritmo_anterior"] != algoritmo:
                    st.session_state["algoritmo_aplicado"] = False
                    st.session_state["algoritmo_anterior"] = algoritmo

                if not st.session_state["algoritmo_aplicado"]:
                    from views.algorithms import aplicar_algoritmo

                    if algoritmo in ["Por zona","Por proximidad",
                                    "Balanceado Preciso","Capacitado","Sweep","kms"]:
                        st.session_state["df"] = aplicar_algoritmo(
                            st.session_state["df"], algoritmo, n_dias, columna="Dia"
                        )
                        st.success(f"✅ Asignación aplicada con algoritmo {algoritmo}")
                        st.session_state["algoritmo_aplicado"] = True

                    elif algoritmo == "kms-evolutivo":
                        df_opt, info = asignar_por_kmeans_evolutivo(
                            st.session_state["df"], cantidades,
                            n_generations=50, population_size=20,
                            alpha=1.0, beta=3.0, gamma=2.0
                        )
                        st.session_state["df"] = df_opt
                        st.success("✅ Asignación híbrida KMeans + Evolutivo aplicada")
                        st.session_state["algoritmo_aplicado"] = True

                        st.subheader("📍 Validación KMeans-Evolutivo")
                        st.write(f"Mejor costo: {info['mejor_costo']}")
                        st.write(f"Puntos sin asignar: {(df_opt['Dia'] == -1).sum()}")
                        st.table(df_opt.groupby("Dia").agg(Cantidad=("Dia","count")).reset_index())

                        import matplotlib.pyplot as plt
                        st.subheader("📈 Convergencia del algoritmo híbrido")
                        fig, ax = plt.subplots()
                        ax.plot(info["historial_costos"], marker="o", linestyle="-", color="green")
                        ax.set_title("Evolución del costo por generación")
                        ax.set_xlabel("Generación")
                        ax.set_ylabel("Costo")
                        st.pyplot(fig)

                    elif algoritmo == "Manual":
                        sugeridas = self.sugerir_distribucion(st.session_state["df"], n_dias)
                        st.info(f"📊 Distribución propuesta: {sugeridas} puntos por día")
                        if st.button("✅ Aceptar propuesta", key="aceptar_propuesta"):
                            st.session_state["df"] = asignacion_manual(st.session_state["df"], sugeridas)
                            st.success("Asignación aplicada con la distribución propuesta")
                            st.session_state["algoritmo_aplicado"] = True

                # 👉 Normalización y resumen
                if "Dia" in st.session_state["df"].columns:
                    st.session_state["df"]["Dia"] = st.session_state["df"]["Dia"].astype(int)

                dias_ctrl.data = st.session_state["df"]

                # 🗺️ Mapa automático
                st.subheader("🗺️ Distribución automática por días")
                render_colored_map(st.session_state["df"], color_by="Dia", key="map_auto")

                # 📊 Resumen por día
                st.subheader("📊 Resumen por día")
                resumen = (
                    st.session_state["df"]
                    .groupby("Dia")
                    .agg(Cantidad_puntos=("Dia", "count"))
                    .reset_index()
                )
                st.table(resumen)

                # ✏️ Mapa editable (siempre visible)
                st.subheader("✏️ Edición manual en el mapa")
                output = render_colored_map(st.session_state["df"], color_by="Dia", key="map_editable")

                if output and output.get("last_active_drawing"):
                    coords_poly = output["last_active_drawing"]["geometry"]["coordinates"][0]
                    polygon = Polygon(coords_poly)

                    seleccionados = st.session_state["df"][st.session_state["df"].apply(
                        lambda r: polygon.contains(Point(r['Longitud'], r['Latitud'])), axis=1
                    )]

                    st.success(f"Puntos seleccionados: {len(seleccionados)}")
                    st.write(seleccionados)

                    if len(seleccionados) > 0:
                        dia_manual = st.number_input(
                            "Asignar estos puntos al día:",
                            min_value=0, max_value=n_dias-1, step=1,
                            key=f"dia_manual_{len(seleccionados)}"
                        )

                        if st.button("💾 Guardar cambios en asignación", key=f"guardar_{len(seleccionados)}"):
                            st.session_state["df"].loc[seleccionados.index, "Dia"] = int(dia_manual)
                            dias_ctrl.data = st.session_state["df"]
                            dias_ctrl.mostrar_resumen_por_dia()
                            st.session_state["cambios_guardados"] = True
                            st.success("Cambios guardados correctamente ✅")

                # 🗺️ Mapa final (siempre visible)
                st.subheader("🗺️ Distribución final")
                render_colored_map(st.session_state["df"], color_by="Dia", key="map_final")

                # 👉 Botón de descarga de la distribución completa
                output_excel_completo = io.BytesIO()
                with pd.ExcelWriter(output_excel_completo, engine="openpyxl") as writer:
                    st.session_state["df"].to_excel(writer, index=False, sheet_name="Distribucion_Final")
                    resumen.to_excel(writer, index=False, sheet_name="Resumen")

                st.download_button(
                    label="📥 Descargar distribución completa (todos los días + resumen)",
                    data=output_excel_completo.getvalue(),
                    file_name="distribucion_completa.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

    def run_por_tecnicos(self):
        st.title("Vista por técnicos")

        # 👉 Seleccionar día
        dias_disponibles = sorted(st.session_state["df"]["Dia"].unique())
        dia_seleccionado = st.selectbox(
            "Seleccionar día para asignación de técnicos:",
            options=dias_disponibles,
            index=0,
            key="dia_tecnicos_selector"
        )

        puntos_dia = st.session_state["df"][st.session_state["df"]["Dia"] == dia_seleccionado]
        st.write(f"Puntos asignados al día {dia_seleccionado}: {len(puntos_dia)}")

        if len(puntos_dia) > 0:
            tecnicos_ctrl = TecnicosController(st.session_state["df"], dia_seleccionado=dia_seleccionado)
            tecnicos_ctrl.run()


    def sugerir_distribucion(df, n_dias):
        """
        Sugiere una distribución automática de puntos por día
        según el número de días elegido.
        """
        total = len(df)
        base = total // n_dias
        resto = total % n_dias

        cantidades = [base] * n_dias
        for i in range(resto):
            cantidades[i] += 1

        return cantidades
