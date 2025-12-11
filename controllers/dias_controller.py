import streamlit as st
import pandas as pd
import numpy as np

class DiasController:
    def __init__(self, data: pd.DataFrame):
        """
        Controlador para manejar la división de puntos en días.
        data: DataFrame con al menos columnas ['Latitud', 'Longitud'].
        """
        self.data = data.copy()
        self.n_dias = None
        self.total_puntos = len(data)

    def solicitar_numero_dias(self):
        """
        Solicita al usuario el número de días.
        """
        self.n_dias = st.number_input(
            "Ingrese número de días:",
            min_value=1,
            step=1,
            help="Número de grupos en que se dividirán los puntos."
        )
        return self.n_dias

    def asignar_puntos_por_dia(self):
        """
        Genera una propuesta automática balanceada de puntos por día.
        Ya no se edita en tabla, porque la edición será en el mapa.
        """
        if not self.n_dias or self.n_dias <= 0:
            st.warning("Debe ingresar un número de días válido.")
            return None

        # 👉 Propuesta automática balanceada
        sugerido = int(np.floor(self.total_puntos / self.n_dias))
        cantidades_auto = [sugerido] * self.n_dias
        # Ajustar el último día si sobra
        cantidades_auto[-1] += self.total_puntos - sum(cantidades_auto)

        st.info(
            f"Total de puntos: {self.total_puntos}. "
            f"Sugerencia inicial: {sugerido} puntos por día."
        )

        return cantidades_auto

    def mostrar_resumen_por_dia(self):
        """
        Muestra tabla con cantidad de puntos por día según la columna 'Dia'.
        """
        if 'Dia' not in self.data.columns:
            st.warning("⚠️ Aún no se ha asignado la columna 'Dia'.")
            return

        resumen = (
            self.data['Dia']
            .value_counts()
            .sort_index()
            .reset_index()
        )
        resumen.columns = ['Día', 'Cantidad de puntos']

        st.subheader("📊 Resumen por día")
        st.dataframe(resumen, use_container_width=True)

        # 👉 Validación extra: mostrar si faltan o sobran puntos
        suma = resumen['Cantidad de puntos'].sum()
        if suma < self.total_puntos:
            st.warning(f"Quedan {self.total_puntos - suma} puntos sin asignar.")
        elif suma > self.total_puntos:
            st.error(f"La suma ({suma}) supera el total de puntos ({self.total_puntos}).")
