import sys
import asyncio

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import truncnorm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from fpdf import FPDF
import os
import warnings
import heapq
import math
from typing import Tuple, List

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Simulador Bancario Profesional",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# UTILIDADES
# ==========================================
def trunc_normal_sample(mu: float, sigma: float, low: float, high: float) -> float:
    a, b = (low - mu) / sigma, (high - mu) / sigma
    return truncnorm.rvs(a, b, loc=mu, scale=sigma)

def ic_95(data: List[float]) -> Tuple[float, float, float]:
    arr = np.array(data)
    mean = arr.mean()
    se = arr.std(ddof=1) / math.sqrt(len(arr)) if len(arr) > 1 else 0.0
    z = 1.96
    return mean, mean - z * se, mean + z * se

# ==========================================
# 1. LÓGICA DE SIMULACIÓN (mejorada)
# ==========================================
def simular_banco_multicajero(
    num_clientes: int,
    num_cajeros: int,
    modo_demanda: int,
    seed: int = None,
    use_heap: bool = False,
    deposito_mu: float = 3.0,
    deposito_sigma: float = 0.2,
    deposito_min: float = 0.1,
    deposito_max: float = 10.0
) -> Tuple[pd.DataFrame, List[float]]:
    """
    Simula un sistema multicajero.
    Retorna DataFrame con registro por cliente y lista de ocio por cajero.
    Parámetros:
      - modo_demanda: 1 -> alta (media llegada 1.5), 2 -> baja (media llegada 20.0)
      - seed: reproducibilidad (None = aleatorio)
      - use_heap: si True usa heapq para eficiencia con muchos cajeros
    Unidades: minutos
    """
    if seed is not None:
        np.random.seed(seed)

    # <-- Ajuste solicitado: media de llegada en alta demanda = 1.5 minutos
    media_llegada = 1.5 if modo_demanda == 1 else 20.0
    datos = []
    h_llegada_ant = 0.0

    # tiempos_libres_cajeros[j] = instante (min) en que el cajero j queda libre
    tiempos_libres_cajeros = [0.0] * num_cajeros
    ocio_acumulado_cajeros = [0.0] * num_cajeros

    # heap opcional: (tiempo_libre, id_cajero)
    if use_heap:
        heap = [(0.0, j) for j in range(num_cajeros)]
        heapq.heapify(heap)

    for i in range(1, num_clientes + 1):
        # Generar tiempo entre llegadas exponencial (media = media_llegada)
        ri_llegadas = np.random.rand()
        t_entre_llegadas = -media_llegada * np.log(ri_llegadas)
        h_llegada = h_llegada_ant + t_entre_llegadas

        # ===== actualizar ocio correctamente para TODOS los cajeros libres hasta h_llegada
        for j in range(num_cajeros):
            if tiempos_libres_cajeros[j] < h_llegada:
                ocio_acumulado_cajeros[j] += (h_llegada - tiempos_libres_cajeros[j])
                tiempos_libres_cajeros[j] = h_llegada  # sincronizamos su "último tiempo libre" hasta llegada

        # ===== seleccionar cajero disponible (el que quede libre antes)
        if use_heap:
            # peek el menor tiempo libre
            h_inicio_posible, id_cajero = heap[0]
        else:
            h_inicio_posible = min(tiempos_libres_cajeros)
            id_cajero = tiempos_libres_cajeros.index(h_inicio_posible)

        # inicio de atención
        h_inicio = max(h_inicio_posible, h_llegada)

        # operación aleatoria
        ri_operacion = np.random.rand()
        if ri_operacion < 0.50:
            operacion = "Retiro"
        elif ri_operacion < 0.70:
            operacion = "Transferencia"
        else:
            operacion = "Depósito"

        # tiempo de servicio según operación
        ri_servicio = np.random.rand()
        if operacion in ["Retiro", "Transferencia"]:
            # servicio uniforme entre 1 y 2 minutos
            t_servicio = 1.0 + (2.0 - 1.0) * ri_servicio
        else:
            # Depósito: usar normal truncada para evitar valores extremos o negativos
            t_servicio = trunc_normal_sample(
                mu=deposito_mu,
                sigma=deposito_sigma,
                low=deposito_min,
                high=deposito_max
            )

        # asegurar no negativo y un mínimo razonable
        t_servicio = max(0.01, float(t_servicio))

        h_salida = h_inicio + t_servicio
        t_sistema = h_salida - h_llegada

        # actualizar tiempos libres y heap si aplica
        if use_heap:
            # reemplazar la tupla del cajero en el heap
            heapq.heapreplace(heap, (h_salida, id_cajero))
            # sincronizar arreglo auxiliar para trazabilidad
            tiempos_libres_cajeros[id_cajero] = h_salida
        else:
            tiempos_libres_cajeros[id_cajero] = h_salida

        datos.append([
            i, id_cajero + 1, t_entre_llegadas, h_llegada, h_inicio,
            None,  # Esperó_Fila se insertará después
            None,  # T.Espera se insertará después
            operacion, t_servicio, h_salida, t_sistema
        ])

        h_llegada_ant = h_llegada

    columnas = ["Cliente", "Cajero", "T.Entre", "H.Llegada", "H.Inicio",
                "Esperó_Fila", "T.Espera", "Operación", "T.Servicio", "H.Salida", "T.Sistema"]

    df = pd.DataFrame(datos, columns=columnas)

    # calcular Esperó_Fila y T.Espera
    espera_calculada = df['H.Inicio'] - df['H.Llegada']
    df['Esperó_Fila'] = np.where(espera_calculada > 1e-9, 'Sí', 'No')
    df['T.Espera'] = espera_calculada

    return df, ocio_acumulado_cajeros

# ==========================================
# 2. REPLICACIONES Y ESTADÍSTICAS
# ==========================================
def replicar_simulacion(
    R: int,
    num_clientes: int,
    num_cajeros: int,
    modo_demanda: int,
    use_heap: bool = False
):
    resultados_prom_ocio = []
    resultados_pct_tiempo = []
    for r in range(R):
        df, ocio = simular_banco_multicajero(
            num_clientes=num_clientes,
            num_cajeros=num_cajeros,
            modo_demanda=modo_demanda,
            seed=r,
            use_heap=use_heap
        )
        t_final = df['H.Salida'].max()
        pct_tiempo_sistema = (df['T.Sistema'].mean() / t_final) * 100 if t_final > 0 else 0.0
        prom_ocio = sum(ocio) / num_cajeros
        resultados_prom_ocio.append(prom_ocio)
        resultados_pct_tiempo.append(pct_tiempo_sistema)

    prom_ocio_mean, prom_ocio_lo, prom_ocio_hi = ic_95(resultados_prom_ocio)
    pct_mean, pct_lo, pct_hi = ic_95(resultados_pct_tiempo)

    resumen = {
        "prom_ocio_mean": prom_ocio_mean,
        "prom_ocio_ic": (prom_ocio_lo, prom_ocio_hi),
        "pct_tiempo_mean": pct_mean,
        "pct_tiempo_ic": (pct_lo, pct_hi),
        "raw_prom_ocio": resultados_prom_ocio,
        "raw_pct_tiempo": resultados_pct_tiempo
    }
    return resumen

# ==========================================
# 3. ANÁLISIS PROFESIONAL (sin cambios lógicos, mejor robustez)
# ==========================================
def generar_analisis_dinamico(df: pd.DataFrame, ocio_list: List[float], num_cajeros: int, num_clientes: int) -> str:
    t_final = df['H.Salida'].max()
    prom_ocio = sum(ocio_list) / num_cajeros if num_cajeros > 0 else 0.0
    pct_tiempo_sistema = (df['T.Sistema'].mean() / t_final) * 100 if t_final > 0 else 0.0

    servicios = df.groupby("Operación")["T.Servicio"].sum()
    total_serv = servicios.sum()
    pct_deposito = (servicios.get("Depósito", 0) / total_serv) * 100 if total_serv > 0 else 0

    analisis = "ANÁLISIS OPERATIVO Y RECOMENDACIONES:\n\n"
    analisis += "1. Distribución del Trabajo:\n"
    if pct_deposito > 40:
        analisis += f"Se detecta que los Depósitos consumen una gran parte del tiempo operativo ({pct_deposito:.1f}%). Esto representa un cuello de botella. Se recomienda derivar estas transacciones a Practicajas o cajeros automáticos inteligentes.\n\n"
    else:
        analisis += "Las transacciones fluyen de manera equilibrada sin que una sola operación sature el tiempo de los cajeros en ventanilla.\n\n"

    analisis += "2. Capacidad Instalada y Ocio:\n"
    if prom_ocio > 2.0:
        analisis += f"El tiempo de ocio promedio por cajero es alto ({prom_ocio:.2f} min). Esto indica una sobrecapacidad instalada. Para este volumen de {num_clientes} clientes, tener {num_cajeros} cajeros resulta financieramente ineficiente. Se sugiere reducir la plantilla activa.\n\n"
    elif prom_ocio < 0.5:
        analisis += f"El tiempo de ocio promedio es muy bajo ({prom_ocio:.2f} min). Los cajeros están trabajando a su máxima capacidad. El sistema es altamente productivo, pero vulnerable a saturación si llegan ráfagas de clientes.\n\n"
    else:
        analisis += f"El tiempo de ocio promedio ({prom_ocio:.2f} min) se encuentra en niveles aceptables, mostrando un equilibrio entre productividad y disponibilidad.\n\n"

    analisis += "3. Experiencia del Cliente:\n"
    analisis += f"El tiempo promedio de los clientes en el sistema representa el {pct_tiempo_sistema:.2f}% del tiempo total de la simulación. La fila única está funcionando correctamente para gestionar el flujo."

    return analisis

# ==========================================
# 4. PDF EJECUTIVO (sin cambios funcionales)
# ==========================================
class PDFReport(FPDF):
    def header(self):
        self.set_fill_color(41, 128, 185)
        self.rect(0, 0, 210, 22, 'F')
        self.set_y(6)
        self.set_font('Arial', 'B', 16)
        self.set_text_color(255, 255, 255)
        self.cell(0, 10, 'Reporte Ejecutivo: Simulacion Bancaria', 0, 1, 'C')

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f'Pagina {self.page_no()}', 0, 0, 'C')

    def seccion_titulo(self, texto):
        self.set_font("Arial", 'B', 12)
        self.set_text_color(41, 128, 185)
        self.cell(0, 8, txt=texto, ln=True)
        self.set_draw_color(41, 128, 185)
        self.set_line_width(0.5)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(3)
        self.set_text_color(60, 60, 60)

def crear_pdf(df: pd.DataFrame, analisis: str, prom_ocio: float, pct_tiempo_sistema: float, conteo_cajeros):
    pdf = PDFReport()
    pdf.add_page()
    pdf.set_y(30)

    pdf.seccion_titulo("Resumen General Operativo")
    pdf.set_font("Arial", 'B', 10)
    pdf.set_text_color(60, 60, 60)
    pdf.cell(0, 6, txt="Carga de trabajo por cajero (Muestra):", ln=True)
    pdf.set_font("Arial", '', 10)

    for cj, cant in list(conteo_cajeros.items())[:15]:
        pdf.cell(10)
        pdf.cell(0, 5, txt=f"- Cajero {cj}: {cant} clientes atendidos", ln=True)
    if len(conteo_cajeros) > 15:
        pdf.cell(10)
        pdf.cell(0, 5, txt=f"... y {len(conteo_cajeros)-15} cajeros mas.", ln=True)

    pdf.ln(3)
    pdf.set_fill_color(240, 248, 255)
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(0, 7, txt=f"  Tiempos Promedio de Clientes en el Sistema: {pct_tiempo_sistema:.2f}%", ln=True, fill=True)
    pdf.cell(0, 7, txt=f"  Promedio de Tiempo de Ocio en Cajas: {prom_ocio:.4f} min", ln=True, fill=True)
    pdf.ln(6)

    pdf.seccion_titulo("Analisis Especializado")
    pdf.set_font("Arial", '', 10)
    analisis_limpio = analisis.replace("ANÁLISIS OPERATIVO Y RECOMENDACIONES:\n\n", "")
    analisis_limpio = analisis_limpio.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 5, txt=analisis_limpio)

    pdf.add_page()
    pdf.set_y(28)

    if os.path.exists("graficas_pasteles.jpg"):
        pdf.seccion_titulo("Matriz de Graficas Operativas (Dashboard)")
        Y_PASTELES = pdf.get_y()
        ALTO_PASTELES = 68
        pdf.image("graficas_pasteles.jpg", x=10, y=Y_PASTELES, w=190, h=ALTO_PASTELES)
        pdf.set_y(Y_PASTELES + ALTO_PASTELES + 8)

    if os.path.exists("graficas_monitoreo.jpg"):
        pdf.seccion_titulo("Monitoreo en Tiempo Real")
        Y_MONITOREO = pdf.get_y()
        ALTO_MONITOREO = 68
        pdf.image("graficas_monitoreo.jpg", x=10, y=Y_MONITOREO, w=190, h=ALTO_MONITOREO)
        pdf.set_y(Y_MONITOREO + ALTO_MONITOREO + 8)

    if os.path.exists("grafica_fila_pdf.jpg"):
        pdf.seccion_titulo("Distribucion Avanzada de Tiempos de Espera en Fila")
        Y_FILA = pdf.get_y()
        ALTO_FILA = 68
        pdf.image("grafica_fila_pdf.jpg", x=10, y=Y_FILA, w=190, h=ALTO_FILA)

    pdf.output("Reporte_Simulacion.pdf")

# ==========================================
# 5. INTERFAZ STREAMLIT (adaptada)
# ==========================================
st.title("🏦 Sistema Bancario Multicajero")
st.markdown("---")

st.sidebar.header("⚙️ Parámetros de Control")
clientes  = st.sidebar.number_input("Número de clientes a simular:", min_value=1, max_value=5000, value=100)
cajeros   = st.sidebar.number_input("Número de cajeros activos:",    min_value=1, max_value=500,  value=6)
escenario = st.sidebar.selectbox("Escenario de Demanda:", [("1. ALTA DEMANDA", 1), ("2. BAJA DEMANDA", 2)])
usar_heap = st.sidebar.checkbox("Usar heap para asignación (recomendado si cajeros>50)", value=False)
replicas  = st.sidebar.number_input("Réplicas para IC (0 = desactivar):", min_value=0, max_value=500, value=0)
seed_opt  = st.sidebar.number_input("Semilla (seed) opcional (-1 = aleatorio):", min_value=-1, max_value=999999, value=-1)

if st.sidebar.button("▶️ Ejecutar Simulación", type="primary"):
    seed = None if seed_opt == -1 else int(seed_opt)

    df, ocio_list = simular_banco_multicajero(
        num_clientes=clientes,
        num_cajeros=cajeros,
        modo_demanda=escenario[1],
        seed=seed,
        use_heap=usar_heap
    )

    # limpiar imágenes previas
    for archivo in ["graficas_pasteles.jpg", "graficas_monitoreo.jpg", "grafica_fila_pdf.jpg"]:
        if os.path.exists(archivo):
            os.remove(archivo)

    t_final            = df['H.Salida'].max()
    pct_tiempo_sistema = (df['T.Sistema'].mean() / t_final) * 100 if t_final > 0 else 0.0
    prom_ocio          = sum(ocio_list) / cajeros if cajeros > 0 else 0.0
    conteo_cajeros     = df['Cajero'].value_counts().sort_index()
    analisis_texto     = generar_analisis_dinamico(df, ocio_list, cajeros, clientes)

    st.session_state['df']                 = df
    st.session_state['analisis']           = analisis_texto
    st.session_state['prom_ocio']          = prom_ocio
    st.session_state['pct_tiempo_sistema'] = pct_tiempo_sistema
    st.session_state['conteo']             = conteo_cajeros

    col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
    with col_kpi1:
        st.metric(label="Clientes Atendidos", value=f"{clientes}")
    with col_kpi2:
        st.metric(label="Promedio Tiempo en Sistema", value=f"{pct_tiempo_sistema:.2f}%")
    with col_kpi3:
        if prom_ocio > 59:
            st.metric(label="Ocio Promedio en Ventanilla", value=f"{(prom_ocio/60):.2f} hrs")
        else:
            st.metric(label="Ocio Promedio en Ventanilla", value=f"{prom_ocio:.4f} min")

    st.subheader("📋 Registro Operativo Detallado")
    st.dataframe(df.round(3), use_container_width=True)

    st.subheader("📝 Evaluación del Sistema y Diagnóstico")
    st.info(analisis_texto)

    st.subheader("📊 Visualización de Datos")
    mapa_colores = {"Depósito": '#f39c12', "Retiro": '#3498db', "Transferencia": '#e74c3c'}

    # ── Figura A: Pasteles ───────────────────────────────────────────────────
    fig_pasteles, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    fig_pasteles.patch.set_facecolor('white')
    fig_pasteles.suptitle("Dashboard Bancario", fontsize=18, fontweight='bold')

    servicios_agrupados = df.groupby("Operación")["T.Servicio"].sum()
    mapa_etiquetas_serv = {"Depósito": "Depósitos", "Retiro": "Retiros", "Transferencia": "Transferencias"}
    etiquetas_serv = [mapa_etiquetas_serv.get(op, op) for op in servicios_agrupados.index]
    colores_serv   = [mapa_colores.get(op, '#000000') for op in servicios_agrupados.index]
    if len(servicios_agrupados) > 0:
        ax1.pie(servicios_agrupados, labels=etiquetas_serv, autopct='%1.1f%%',
                startangle=140, colors=colores_serv,
                wedgeprops={'edgecolor': 'grey', 'linewidth': 1.5})
    ax1.set_title("Distribución de Tiempo de Servicio", fontweight='bold')

    try:
        ocio_series = pd.Series(ocio_list)
        if ocio_series.max() == 0 or ocio_series.max() == ocio_series.min():
            val_unico = ocio_series.max()
            etiqueta  = f"{(val_unico/60):.2f} HRS" if val_unico > 59 else f"{val_unico:.2f} min"
            conteo_ocio = pd.Series({etiqueta: len(ocio_series)})
            labels_ocio = conteo_ocio.index
        else:
            bins        = pd.cut(ocio_series, bins=4)
            conteo_ocio = bins.value_counts().sort_index()
            conteo_ocio = conteo_ocio[conteo_ocio > 0]
            labels_ocio = []
            for b in conteo_ocio.index:
                li, ls = max(0, b.left), b.right
                if ls > 59:
                    labels_ocio.append(f"De {(li/60):.2f} a {(ls/60):.2f} hrs")
                else:
                    labels_ocio.append(f"De {li:.2f} a {ls:.2f} min")

        colores_ocio = plt.cm.Pastel1(np.linspace(0, 1, len(conteo_ocio)))
        ax2.pie(conteo_ocio.values, labels=labels_ocio, autopct='%1.1f%%',
                startangle=140, colors=colores_ocio,
                wedgeprops={'edgecolor': 'gray'})
        ax2.set_title("Distribución de Tiempo de Ocio (Por Rangos)", fontweight="bold")
    except Exception as e:
        st.error(f"⚠️ Error en pastel de ocio: {e}")

    plt.tight_layout()
    plt.savefig("graficas_pasteles.jpg", bbox_inches='tight', dpi=120,
                facecolor='white', transparent=False)
    plt.close(fig_pasteles)

    # ── Figura B: Monitoreo (Gantt y/o Histograma) ──────────────────────────
    if cajeros <= 6:
        if clientes <= 10:
            fig_mon, (ax3, ax4) = plt.subplots(1, 2, figsize=(16, 5))
            fig_mon.patch.set_facecolor('white')

            ax3.set_title("Diagrama de Gantt - Monitoreo de Operaciones", fontweight='bold')
            for idx, row in df.iterrows():
                ax3.barh(row['Cajero'], row['T.Servicio'], left=row['H.Inicio'],
                         color=mapa_colores.get(row['Operación'], '#000000'),
                         edgecolor='grey', height=0.6)
            ax3.set_yticks(range(1, cajeros + 1))
            ax3.set_yticklabels([f"Cajero {i}" for i in range(1, cajeros + 1)], fontsize=9)
            handles = [mpatches.Patch(color=mapa_colores[k], label=k) for k in mapa_colores]
            ax3.legend(handles=handles, loc='lower right')
            ax3.grid(axis='x', linestyle='--', alpha=0.5)

            ax4.set_title("Distribución del Tiempo Total en Sistema", fontweight='bold')
            ax4.hist(df['T.Sistema'], bins=12, color="#3498db", edgecolor='white', alpha=0.85)
            ax4.axvline(df['T.Sistema'].mean(), color='red', linestyle='dashed', linewidth=2,
                        label=f"Media: {df['T.Sistema'].mean():.2f} min")
            ax4.legend()
            ax4.grid(axis='y', linestyle='--', alpha=0.5)

        else:
            fig_mon, ax4 = plt.subplots(1, 1, figsize=(16, 5))
            fig_mon.patch.set_facecolor('white')

            ax4.set_title("Distribución del Tiempo Total en Sistema", fontweight='bold')
            ax4.hist(df['T.Sistema'], bins=12, color="#3498db", edgecolor='white', alpha=0.85)
            ax4.axvline(df['T.Sistema'].mean(), color='red', linestyle='dashed', linewidth=2,
                        label=f"Media: {df['T.Sistema'].mean():.2f} min")
            ax4.legend()
            ax4.grid(axis='y', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig("graficas_monitoreo.jpg", bbox_inches='tight', dpi=120,
                    facecolor='white', transparent=False)
        plt.close(fig_mon)

    st.markdown("##### Distribución de Servicios y Tiempos de Ocio")
    st.image("graficas_pasteles.jpg", use_container_width=True)
    if cajeros <= 6 and os.path.exists("graficas_monitoreo.jpg"):
        st.markdown("##### Monitoreo en Tiempo Real")
        st.image("graficas_monitoreo.jpg", use_container_width=True)

    # ── Gráfica de fila para el PDF ────────────────────────────────────────
    fig_pdf, ax_pdf = plt.subplots(figsize=(10, 2.8))
    ax_pdf.hist(df['T.Espera'], bins=10, color="#2ecc71", edgecolor='black', alpha=0.8)
    ax_pdf.set_title("Distribución de Tiempos de Espera en Fila",
                     fontweight='bold', fontsize=11)
    ax_pdf.set_xlabel("Tiempo de espera (minutos)", fontsize=9)
    ax_pdf.set_ylabel("Cantidad de Clientes", fontsize=9)
    ax_pdf.tick_params(axis='both', which='major', labelsize=9)
    ax_pdf.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("grafica_fila_pdf.jpg", bbox_inches='tight', dpi=130, facecolor='white')
    plt.close(fig_pdf)

    # Si el usuario pidió réplicas, ejecutar y mostrar IC
    if replicas and replicas > 0:
        resumen = replicar_simulacion(
            R=int(replicas),
            num_clientes=clientes,
            num_cajeros=cajeros,
            modo_demanda=escenario[1],
            use_heap=usar_heap
        )
        st.subheader("📈 Resultados de Réplicas (IC 95%)")
        st.write(f"Promedio Ocio por cajero: {resumen['prom_ocio_mean']:.4f} min")
        st.write(f"IC 95% Ocio: [{resumen['prom_ocio_ic'][0]:.4f}, {resumen['prom_ocio_ic'][1]:.4f}]")
        st.write(f"Promedio % Tiempo en Sistema: {resumen['pct_tiempo_mean']:.4f} %")
        st.write(f"IC 95% % Tiempo en Sistema: [{resumen['pct_tiempo_ic'][0]:.4f}, {resumen['pct_tiempo_ic'][1]:.4f}]")

# ==========================================
# 6. EXPORTACIÓN (igual que antes)
# ==========================================
if 'df' in st.session_state:
    st.markdown("---")
    st.subheader("📥 Exportación de Entregables Profesionales")

    col_btn1, col_btn2, col_btn3 = st.columns(3)

    with col_btn1:
        df_export = st.session_state['df'].copy()
        df_export['Eficiencia_Atención_%'] = (
            (df_export['T.Servicio'] /
             (df_export['T.Espera'] + df_export['T.Servicio'])) * 100
        ).round(2)

        hora_apertura = pd.to_datetime('2026-06-05 09:00:00')
        df_export['Hora_Llegada_Reloj'] = (hora_apertura +
            pd.to_timedelta(df_export['H.Llegada'], unit='m')).dt.strftime('%H:%M:%S')
        df_export['Hora_Salida_Reloj'] = (hora_apertura +
            pd.to_timedelta(df_export['H.Salida'],  unit='m')).dt.strftime('%H:%M:%S')

        cols_tiempo = ['T.Entre', 'H.Llegada', 'H.Inicio', 'T.Espera',
                       'T.Servicio', 'H.Salida', 'T.Sistema']
        df_export[cols_tiempo] = df_export[cols_tiempo].round(2)

        df_export = df_export.rename(columns={
            'Cliente'   : 'ID_Cliente',
            'Cajero'    : 'Num_Cajero_Asignado',
            'T.Entre'   : 'Tiempo_Entre_Llegadas_min',
            'H.Llegada' : 'Cronómetro_Llegada',
            'H.Inicio'  : 'Cronómetro_Atención',
            'T.Espera'  : 'Minutos_Esperando_Fila',
            'Operación' : 'Tipo_Operación',
            'T.Servicio': 'Tiempo_Transacción_min',
            'H.Salida'  : 'Cronómetro_Salida',
            'T.Sistema' : 'Total_Tiempo_Sucursal_min'
        })

        columnas_finales = [
            'ID_Cliente', 'Hora_Llegada_Reloj', 'Hora_Salida_Reloj',
            'Num_Cajero_Asignado', 'Esperó_Fila', 'Minutos_Esperando_Fila',
            'Tipo_Operación', 'Tiempo_Transacción_min',
            'Eficiencia_Atención_%', 'Total_Tiempo_Sucursal_min'
        ]
        df_export = df_export[columnas_finales]

        csv_data = df_export.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
        st.download_button(
            label="📊 Descargar CSV",
            data=csv_data,
            file_name="Resultados_Bancarios_Auditados.csv",
            mime="text/csv",
            use_container_width=True
        )

    with col_btn2:
        crear_pdf(
            st.session_state['df'],
            st.session_state['analisis'],
            st.session_state['prom_ocio'],
            st.session_state['pct_tiempo_sistema'],
            st.session_state['conteo']
        )
        with open("Reporte_Simulacion.pdf", "rb") as f:
            pdf_data = f.read()

        st.download_button(
            label="📄 Descargar PDF",
            data=pdf_data,
            file_name="Reporte_Ejecutivo_Bancario.pdf",
            mime="application/pdf",
            use_container_width=True
        )

    with col_btn3:
        with st.popover("📧 Enviar por Correo", use_container_width=True):
            st.markdown("**Datos de Envío (Usa Gmail con Contraseña de Aplicación):**")
            remitente = st.text_input("Tu Email:",  placeholder="ejemplo@gmail.com")
            password  = st.text_input("App Pass:", type="password",
                                       placeholder="Contraseña de aplicación")
            destino   = st.text_input("Destino:",  placeholder="destino@correo.com")

            if st.button("Enviar Ahora", type="primary", use_container_width=True):
                if not all([remitente, password, destino]):
                    st.error("⚠️ Faltan datos. Llena todos los campos.")
                else:
                    try:
                        import smtplib
                        from email.mime.multipart import MIMEMultipart
                        from email.mime.base import MIMEBase
                        from email.mime.text import MIMEText
                        from email import encoders

                        msg = MIMEMultipart()
                        msg['From']    = remitente
                        msg['To']      = destino
                        msg['Subject'] = "Reporte Ejecutivo - Simulador Bancario"

                        msg.attach(MIMEText(
                            "Hola. Adjunto encontrarás el reporte generado "
                            "automáticamente por el Simulador Bancario.", 'plain'))

                        with open("Reporte_Simulacion.pdf", "rb") as adjunto:
                            parte = MIMEBase("application", "octet-stream")
                            parte.set_payload(adjunto.read())
                        encoders.encode_base64(parte)
                        parte.add_header("Content-Disposition",
                                         "attachment; filename=Reporte_Simulacion.pdf")
                        msg.attach(parte)

                        with smtplib.SMTP('smtp.gmail.com', 587) as servidor:
                            servidor.starttls()
                            servidor.login(remitente, password)
                            servidor.sendmail(remitente, destino, msg.as_string())

                        st.success(f"¡PDF enviado con éxito a {destino}!")
                        st.balloons()

                    except smtplib.SMTPAuthenticationError:
                        st.error("❌ Error de Autenticación: Verifica tus datos.")
                    except Exception as e:
                        st.error(f"❌ Error inesperado: {e}")
