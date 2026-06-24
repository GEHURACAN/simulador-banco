import sys
import asyncio

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from fpdf import FPDF
import os
import warnings
from typing import Tuple, List
import functools
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders

warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURACIÓN DE PÁGINA
# ==========================================
st.set_page_config(
    page_title="Simulador Bancario Profesional",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CONSTANTES
# ==========================================
PROB_RETIRO = 0.50
PROB_TRANSFERENCIA = 0.20
MEDIA_LLEGADA_ALTA = 1.5  # Cambiado a 1.5 como recomendaste
MEDIA_LLEGADA_BAJA = 20.0
TIEMPO_SERVICIO_MIN = 1.0
TIEMPO_SERVICIO_MAX = 2.0
DEPOSITO_MEDIA = 3.0
DEPOSITO_DESVIACION = 0.2
DEPOSITO_MIN = 2.0
DEPOSITO_MAX = 4.0
PRECISION_NUMERICA = 1e-9

# ==========================================
# 1. LÓGICA DE SIMULACIÓN MEJORADA
# ==========================================
def simular_banco_multicajero(
    num_clientes: int, 
    num_cajeros: int, 
    modo_demanda: int,
    progress_callback=None
) -> Tuple[pd.DataFrame, List[float]]:
    """
    Simula un sistema bancario con múltiples cajeros.
    
    Args:
        num_clientes: Número de clientes a simular
        num_cajeros: Número de cajeros activos
        modo_demanda: 1 = Alta demanda, 2 = Baja demanda
        progress_callback: Función para actualizar progreso
    
    Returns:
        DataFrame con resultados y lista de tiempos de ocio
    """
    media_llegada = MEDIA_LLEGADA_ALTA if modo_demanda == 1 else MEDIA_LLEGADA_BAJA
    datos = []
    h_llegada_ant = 0
    tiempos_libres_cajeros = [0.0] * num_cajeros
    ocio_acumulado_cajeros = [0.0] * num_cajeros
    
    # Pre-generar arrays aleatorios para mejorar rendimiento
    llegadas_ri = np.random.rand(num_clientes)
    operacion_ri = np.random.rand(num_clientes)
    servicio_ri = np.random.rand(num_clientes)

    for i in range(1, num_clientes + 1):
        # Tiempo entre llegadas
        t_entre_llegadas = -media_llegada * np.log(llegadas_ri[i-1])
        h_llegada = h_llegada_ant + t_entre_llegadas

        # Encontrar cajero más libre
        h_inicio_posible = min(tiempos_libres_cajeros)
        id_cajero = tiempos_libres_cajeros.index(h_inicio_posible)

        # Calcular ocio
        if h_llegada > h_inicio_posible:
            ocio_acumulado_cajeros[id_cajero] += (h_llegada - h_inicio_posible)

        h_inicio = max(h_inicio_posible, h_llegada)

        # Determinar operación
        ri_operacion = operacion_ri[i-1]
        if ri_operacion < PROB_RETIRO:
            operacion = "Retiro"
        elif ri_operacion < PROB_RETIRO + PROB_TRANSFERENCIA:
            operacion = "Transferencia"
        else:
            operacion = "Depósito"

        # Tiempo de servicio con validación
        ri_servicio = servicio_ri[i-1]
        if operacion in ["Retiro", "Transferencia"]:
            t_servicio = TIEMPO_SERVICIO_MIN + (TIEMPO_SERVICIO_MAX - TIEMPO_SERVICIO_MIN) * ri_servicio
        else:  # Depósito
            # Usar distribución normal truncada para evitar valores atípicos
            t_servicio = norm.ppf(ri_servicio, loc=DEPOSITO_MEDIA, scale=DEPOSITO_DESVIACION)
            t_servicio = max(DEPOSITO_MIN, min(DEPOSITO_MAX, t_servicio))

        h_salida = h_inicio + t_servicio
        t_sistema = h_salida - h_llegada
        tiempos_libres_cajeros[id_cajero] = h_salida

        datos.append([
            i, id_cajero + 1, t_entre_llegadas, h_llegada, h_inicio,
            operacion, t_servicio, h_salida, t_sistema
        ])
        h_llegada_ant = h_llegada
        
        # Actualizar progreso si se proporciona callback
        if progress_callback and i % 100 == 0:
            progress_callback(i / num_clientes)

    columnas = ["Cliente", "Cajero", "T.Entre", "H.Llegada", "H.Inicio",
                "Operación", "T.Servicio", "H.Salida", "T.Sistema"]
    df = pd.DataFrame(datos, columns=columnas)
    
    # Calcular espera con precisión numérica
    espera_calculada = (df['H.Inicio'] - df['H.Llegada']).round(6)
    df.insert(5, 'Esperó_Fila', np.where(espera_calculada > PRECISION_NUMERICA, 'Sí', 'No'))
    df.insert(6, 'T.Espera', espera_calculada)
    
    return df, ocio_acumulado_cajeros

# ==========================================
# 2. ANÁLISIS PROFESIONAL MEJORADO
# ==========================================
def generar_analisis_dinamico(
    df: pd.DataFrame, 
    ocio_list: List[float], 
    num_cajeros: int, 
    num_clientes: int
) -> str:
    """Genera un análisis detallado del sistema bancario."""
    t_final = df['H.Salida'].max()
    prom_ocio = sum(ocio_list) / num_cajeros
    pct_tiempo_sistema = (df['T.Sistema'].mean() / t_final) * 100
    
    # Métricas de espera
    clientes_esperaron = (df['Esperó_Fila'] == 'Sí').sum()
    porcentaje_espera = (clientes_esperaron / len(df)) * 100
    espera_promedio = df[df['Esperó_Fila'] == 'Sí']['T.Espera'].mean() if clientes_esperaron > 0 else 0
    espera_maxima = df['T.Espera'].max()

    servicios = df.groupby("Operación")["T.Servicio"].sum()
    total_serv = servicios.sum()
    pct_deposito = (servicios.get("Depósito", 0) / total_serv) * 100 if total_serv > 0 else 0

    # Tasa de utilización
    tasa_utilizacion = (1 - (prom_ocio / t_final)) * 100 if t_final > 0 else 0

    analisis = "ANÁLISIS OPERATIVO Y RECOMENDACIONES:\n\n"
    
    # 1. Distribución del trabajo
    analisis += "1. Distribución del Trabajo:\n"
    if pct_deposito > 40:
        analisis += f"   - Se detecta que los Depósitos consumen una gran parte del tiempo operativo ({pct_deposito:.1f}%). "
        analisis += "Esto representa un cuello de botella. Se recomienda derivar estas transacciones a Practicajas o cajeros automáticos inteligentes.\n\n"
    else:
        analisis += "   - Las transacciones fluyen de manera equilibrada sin que una sola operación sature el tiempo de los cajeros en ventanilla.\n\n"

    # 2. Capacidad instalada
    analisis += "2. Capacidad Instalada y Ocio:\n"
    if prom_ocio > 2.0:
        analisis += f"   - El tiempo de ocio promedio por cajero es alto ({prom_ocio:.2f} min). "
        analisis += f"Esto indica una sobrecapacidad instalada. Para este volumen de {num_clientes} clientes, "
        analisis += f"tener {num_cajeros} cajeros resulta financieramente ineficiente. Se sugiere reducir la plantilla activa.\n\n"
    elif prom_ocio < 0.5:
        analisis += f"   - El tiempo de ocio promedio es muy bajo ({prom_ocio:.2f} min). "
        analisis += "Los cajeros están trabajando a su máxima capacidad. El sistema es altamente productivo, "
        analisis += "pero vulnerable a saturación si llegan ráfagas de clientes.\n\n"
    else:
        analisis += f"   - El tiempo de ocio promedio ({prom_ocio:.2f} min) se encuentra en niveles aceptables, "
        analisis += "mostrando un equilibrio entre productividad y disponibilidad.\n\n"

    # 3. Experiencia del cliente
    analisis += "3. Experiencia del Cliente:\n"
    analisis += f"   - El tiempo promedio de los clientes en el sistema representa el {pct_tiempo_sistema:.2f}% del tiempo total de la simulación.\n"
    analisis += f"   - La fila única está funcionando correctamente para gestionar el flujo.\n\n"

    # 4. Análisis de filas (NUEVO)
    analisis += "4. Análisis de Filas y Espera:\n"
    analisis += f"   - Clientes que esperaron en fila: {clientes_esperaron} de {len(df)} ({porcentaje_espera:.1f}%)\n"
    if clientes_esperaron > 0:
        analisis += f"   - Tiempo promedio de espera en fila: {espera_promedio:.2f} minutos\n"
        analisis += f"   - Tiempo máximo de espera en fila: {espera_maxima:.2f} minutos\n"
        if espera_promedio > 1.0:
            analisis += "   - ⚠️ El tiempo de espera en fila supera 1 minuto. Considerar abrir más cajeros en horas pico.\n"
    else:
        analisis += "   - ✅ ¡Excelente! Ningún cliente esperó en fila. El sistema tiene capacidad suficiente.\n"
    analisis += "\n"

    # 5. Eficiencia del sistema (NUEVO)
    analisis += "5. Eficiencia del Sistema:\n"
    analisis += f"   - Tasa de utilización de cajeros: {tasa_utilizacion:.1f}%\n"
    if tasa_utilizacion > 85:
        analisis += "   - ⚠️ Alta utilización. El sistema opera cerca de su capacidad máxima.\n"
    elif tasa_utilizacion < 40:
        analisis += "   - 📉 Baja utilización. El sistema está subutilizado.\n"
    else:
        analisis += "   - ✅ Nivel de utilización óptimo para operaciones bancarias.\n"

    return analisis

# ==========================================
# 3. PDF EJECUTIVO MEJORADO
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


def crear_pdf(
    df: pd.DataFrame, 
    analisis: str, 
    prom_ocio: float, 
    pct_tiempo_sistema: float, 
    conteo_cajeros: pd.Series
):
    """Genera el PDF ejecutivo con gráficas."""
    pdf = PDFReport()
    pdf.add_page()
    pdf.set_y(30)

    # ── Página 1: Resumen operativo ──────────────────────────────────────────
    pdf.seccion_titulo("Resumen General Operativo")

    # Métricas de espera
    clientes_esperaron = (df['Esperó_Fila'] == 'Sí').sum()
    porcentaje_espera = (clientes_esperaron / len(df)) * 100
    espera_promedio = df[df['Esperó_Fila'] == 'Sí']['T.Espera'].mean() if clientes_esperaron > 0 else 0

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
    pdf.cell(0, 7, txt=f"  Clientes que Esperaron en Fila: {clientes_esperaron} ({porcentaje_espera:.1f}%)", ln=True, fill=True)
    if clientes_esperaron > 0:
        pdf.cell(0, 7, txt=f"  Tiempo Promedio de Espera: {espera_promedio:.2f} min", ln=True, fill=True)
    pdf.ln(6)

    pdf.seccion_titulo("Analisis Especializado")
    pdf.set_font("Arial", '', 10)
    analisis_limpio = analisis.replace("ANÁLISIS OPERATIVO Y RECOMENDACIONES:\n\n", "")
    analisis_limpio = analisis_limpio.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 5, txt=analisis_limpio)

    # ── Página 2: Gráficas ───────────────────────────────────────────────
    pdf.add_page()
    pdf.set_y(28)

    # Sección 1: Dashboard (pasteles)
    if os.path.exists("graficas_pasteles.jpg") and os.path.getsize("graficas_pasteles.jpg") > 0:
        pdf.seccion_titulo("Matriz de Graficas Operativas (Dashboard)")
        Y_PASTELES = pdf.get_y()
        ALTO_PASTELES = 68
        pdf.image("graficas_pasteles.jpg", x=10, y=Y_PASTELES, w=190, h=ALTO_PASTELES)
        pdf.set_y(Y_PASTELES + ALTO_PASTELES + 8)

    # Sección 2: Monitoreo (solo si cajeros <= 6)
    if os.path.exists("graficas_monitoreo.jpg") and os.path.getsize("graficas_monitoreo.jpg") > 0:
        pdf.seccion_titulo("Monitoreo en Tiempo Real")
        Y_MONITOREO = pdf.get_y()
        ALTO_MONITOREO = 68
        pdf.image("graficas_monitoreo.jpg", x=10, y=Y_MONITOREO, w=190, h=ALTO_MONITOREO)
        pdf.set_y(Y_MONITOREO + ALTO_MONITOREO + 8)

    # Sección 3: Tiempos de espera
    if os.path.exists("grafica_fila_pdf.jpg") and os.path.getsize("grafica_fila_pdf.jpg") > 0:
        pdf.seccion_titulo("Distribucion Avanzada de Tiempos de Espera en Fila")
        Y_FILA = pdf.get_y()
        ALTO_FILA = 68
        pdf.image("grafica_fila_pdf.jpg", x=10, y=Y_FILA, w=190, h=ALTO_FILA)

    pdf.output("Reporte_Simulacion.pdf")

# ==========================================
# 4. INTERFAZ STREAMLIT
# ==========================================
st.title("🏦 Sistema Bancario Multicajero")
st.markdown("---")

st.sidebar.header("⚙️ Parámetros de Control")
st.sidebar.info(f"📊 Media de llegadas (Alta demanda): {MEDIA_LLEGADA_ALTA} min")

# ── Número de clientes ──────────────────────────────────────────────
clientes = st.sidebar.number_input(
    "Número de clientes a simular:", 
    min_value=1, 
    max_value=5000, 
    value=100
)

# ── Número de cajeros (VERSIÓN ELEGANTE CON OPCIONES RECOMENDADAS) ──
st.sidebar.markdown("**Número de cajeros activos:**")

# Crear columnas para el selector
col1, col2, col3 = st.sidebar.columns([1, 2, 1])

with col1:
    if st.button("➖", key="menos_cajeros", use_container_width=True):
        if 'cajeros_temp' not in st.session_state:
            st.session_state.cajeros_temp = 4
        # Disminuir entre las opciones: 6→5→4→3
        opciones = [3, 4, 5, 6]
        idx = opciones.index(st.session_state.cajeros_temp) if st.session_state.cajeros_temp in opciones else 0
        st.session_state.cajeros_temp = opciones[max(0, idx - 1)]

with col2:
    # Opciones recomendadas
    opciones_cajeros = [3, 4, 5, 6]
    etiquetas_cajeros = {
        3: "⚠️ 3 - Espera moderada",
        4: "✅ 4 - Punto óptimo",
        5: "👍 5 - Poca espera",
        6: "📊 6 - Sin espera"
    }
    
    if 'cajeros_temp' not in st.session_state:
        st.session_state.cajeros_temp = 4
    
    cajeros_seleccionado = st.selectbox(
        "Seleccionar:",
        options=opciones_cajeros,
        index=opciones_cajeros.index(st.session_state.cajeros_temp) if st.session_state.cajeros_temp in opciones_cajeros else 1,
        format_func=lambda x: etiquetas_cajeros.get(x, f"{x} cajeros"),
        key="select_cajeros",
        label_visibility="collapsed"
    )
    st.session_state.cajeros_temp = cajeros_seleccionado

with col3:
    if st.button("➕", key="mas_cajeros", use_container_width=True):
        if 'cajeros_temp' not in st.session_state:
            st.session_state.cajeros_temp = 4
        # Aumentar entre las opciones: 3→4→5→6
        opciones = [3, 4, 5, 6]
        idx = opciones.index(st.session_state.cajeros_temp) if st.session_state.cajeros_temp in opciones else 0
        st.session_state.cajeros_temp = opciones[min(3, idx + 1)]

# Asignar el valor seleccionado
cajeros = st.session_state.cajeros_temp
st.sidebar.markdown(f"**✅ Cajeros seleccionados: {cajeros}**")

# ── Escenario de Demanda ────────────────────────────────────────────
escenario = st.sidebar.selectbox(
    "Escenario de Demanda:", 
    [("1. ALTA DEMANDA", 1), ("2. BAJA DEMANDA", 2)]
)

# ── Hora de apertura ────────────────────────────────────────────────
hora_apertura = st.sidebar.time_input(
    "Hora de apertura:", 
    value=pd.Timestamp('09:00:00').time()
)

# ── Botón Ejecutar ──────────────────────────────────────────────────
if st.sidebar.button("▶️ Ejecutar Simulación", type="primary"):
    
    # Limpiar imágenes anteriores
    for archivo in ["graficas_pasteles.jpg", "graficas_monitoreo.jpg", "grafica_fila_pdf.jpg"]:
        if os.path.exists(archivo):
            os.remove(archivo)
    
    # Ejecutar simulación con barra de progreso
    if clientes > 500:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(progress):
            progress_bar.progress(progress)
            status_text.text(f"Simulando cliente {int(progress * clientes)} de {clientes}...")
        
        df, ocio_list = simular_banco_multicajero(clientes, cajeros, escenario[1], update_progress)
        progress_bar.empty()
        status_text.empty()
    else:
        df, ocio_list = simular_banco_multicajero(clientes, cajeros, escenario[1])

    # Calcular métricas
    t_final = df['H.Salida'].max()
    pct_tiempo_sistema = (df['T.Sistema'].mean() / t_final) * 100 if t_final > 0 else 0
    prom_ocio = sum(ocio_list) / cajeros
    conteo_cajeros = df['Cajero'].value_counts().sort_index()
    analisis_texto = generar_analisis_dinamico(df, ocio_list, cajeros, clientes)
    
    # Métricas adicionales
    clientes_esperaron = (df['Esperó_Fila'] == 'Sí').sum()
    porcentaje_espera = (clientes_esperaron / len(df)) * 100
    espera_promedio = df[df['Esperó_Fila'] == 'Sí']['T.Espera'].mean() if clientes_esperaron > 0 else 0
    tasa_utilizacion = (1 - (prom_ocio / t_final)) * 100 if t_final > 0 else 0

    st.session_state['df'] = df
    st.session_state['analisis'] = analisis_texto
    st.session_state['prom_ocio'] = prom_ocio
    st.session_state['pct_tiempo_sistema'] = pct_tiempo_sistema
    st.session_state['conteo'] = conteo_cajeros

    # ── KPIs ────────────────────────────────────────────────────────────────
    col_kpi1, col_kpi2, col_kpi3, col_kpi4, col_kpi5 = st.columns(5)
    with col_kpi1:
        st.metric(label="👥 Clientes Atendidos", value=f"{clientes}")
    with col_kpi2:
        st.metric(label="⏱️ Tiempo Promedio en Sistema", value=f"{pct_tiempo_sistema:.2f}%")
    with col_kpi3:
        if prom_ocio > 59:
            st.metric(label="🕐 Ocio Promedio", value=f"{(prom_ocio/60):.2f} hrs")
        else:
            st.metric(label="🕐 Ocio Promedio", value=f"{prom_ocio:.4f} min")
    with col_kpi4:
        st.metric(label="📊 Utilización", value=f"{tasa_utilizacion:.1f}%")
    with col_kpi5:
        st.metric(label="⏳ % Clientes que Esperaron", value=f"{porcentaje_espera:.1f}%")

    # ── Tabla resumen por operación ──────────────────────────────────────
    st.subheader("📊 Resumen por Tipo de Operación")
    resumen_operaciones = df.groupby('Operación').agg({
        'Cliente': 'count',
        'T.Servicio': ['mean', 'std', 'sum'],
        'T.Espera': 'mean'
    }).round(2)
    resumen_operaciones.columns = ['Cantidad', 'Servicio Promedio', 'Desv. Servicio', 'Total Servicio', 'Espera Promedio']
    st.dataframe(resumen_operaciones, use_container_width=True)

    # ── Registro Detallado ─────────────────────────────────────────────────
    st.subheader("📋 Registro Operativo Detallado")
    if len(df) > 100:
        st.warning(f"Mostrando primeros 100 de {len(df)} registros")
        st.dataframe(df.head(100).round(3), use_container_width=True)
    else:
        st.dataframe(df.round(3), use_container_width=True)

    # ── Análisis ────────────────────────────────────────────────────────────
    st.subheader("📝 Evaluación del Sistema y Diagnóstico")
    st.info(analisis_texto)

    # ── Visualizaciones ────────────────────────────────────────────────────
    st.subheader("📊 Visualización de Datos")

    mapa_colores = {"Depósito": '#f39c12', "Retiro": '#3498db', "Transferencia": '#e74c3c'}

    # ── Figura A: Pasteles ────────────────────────────────────────────────
    fig_pasteles, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    fig_pasteles.patch.set_facecolor('white')
    fig_pasteles.suptitle("Dashboard Bancario", fontsize=18, fontweight='bold')

    servicios_agrupados = df.groupby("Operación")["T.Servicio"].sum()
    mapa_etiquetas_serv = {"Depósito": "Depósitos", "Retiro": "Retiros", "Transferencia": "Transferencias"}
    etiquetas_serv = [mapa_etiquetas_serv.get(op, op) for op in servicios_agrupados.index]
    colores_serv = [mapa_colores.get(op, '#000000') for op in servicios_agrupados.index]
    ax1.pie(servicios_agrupados, labels=etiquetas_serv, autopct='%1.1f%%',
            startangle=140, colors=colores_serv,
            wedgeprops={'edgecolor': 'grey', 'linewidth': 1.5})
    ax1.set_title("Distribución de Tiempo de Servicio", fontweight='bold')

    try:
        ocio_series = pd.Series(ocio_list)
        if ocio_series.max() == 0 or ocio_series.max() == ocio_series.min():
            val_unico = ocio_series.max()
            etiqueta = f"{(val_unico/60):.2f} HRS" if val_unico > 59 else f"{val_unico:.2f} min"
            conteo_ocio = pd.Series({etiqueta: len(ocio_series)})
            labels_ocio = conteo_ocio.index
        else:
            bins = pd.cut(ocio_series, bins=4)
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

    # ── Figura B: Monitoreo ───────────────────────────────────────────────
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

    # Mostrar en Streamlit
    st.markdown("##### Distribución de Servicios y Tiempos de Ocio")
    st.image("graficas_pasteles.jpg", use_container_width=True)
    if cajeros <= 6 and os.path.exists("graficas_monitoreo.jpg"):
        st.markdown("##### Monitoreo en Tiempo Real")
        st.image("graficas_monitoreo.jpg", use_container_width=True)

    # ── Gráfica de fila ────────────────────────────────────────────────────
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

    # ── Gráfica de evolución de la fila (NUEVO) ──────────────────────────
    st.markdown("##### Evolución de la Fila en el Tiempo")
    fig_fila, ax_fila = plt.subplots(figsize=(10, 4))
    df_ordenado = df.sort_values('H.Llegada')
    df_ordenado['Clientes_Acumulados'] = range(1, len(df_ordenado) + 1)
    df_ordenado['Clientes_Atendidos'] = df_ordenado['H.Salida'].rank()
    
    ax_fila.plot(df_ordenado['H.Llegada'], df_ordenado['Clientes_Acumulados'], 
                 label='Clientes Llegados', color='blue', linewidth=2)
    ax_fila.plot(df_ordenado['H.Salida'], df_ordenado['Clientes_Atendidos'], 
                 label='Clientes Atendidos', color='green', linewidth=2)
    ax_fila.fill_between(df_ordenado['H.Llegada'], 
                         df_ordenado['Clientes_Acumulados'], 
                         df_ordenado['Clientes_Atendidos'], 
                         alpha=0.3, color='orange', label='Clientes en Fila')
    ax_fila.set_xlabel('Tiempo (minutos)', fontsize=10)
    ax_fila.set_ylabel('Número de Clientes', fontsize=10)
    ax_fila.legend()
    ax_fila.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig_fila)
    plt.close(fig_fila)

# ==========================================
# 5. EXPORTACIÓN MEJORADA
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

        hora_apertura_dt = pd.to_datetime(f"2026-06-05 {hora_apertura.strftime('%H:%M:%S')}")
        df_export['Hora_Llegada_Reloj'] = (hora_apertura_dt +
            pd.to_timedelta(df_export['H.Llegada'], unit='m')).dt.strftime('%H:%M:%S')
        df_export['Hora_Salida_Reloj'] = (hora_apertura_dt +
            pd.to_timedelta(df_export['H.Salida'], unit='m')).dt.strftime('%H:%M:%S')

        cols_tiempo = ['T.Entre', 'H.Llegada', 'H.Inicio', 'T.Espera',
                       'T.Servicio', 'H.Salida', 'T.Sistema']
        df_export[cols_tiempo] = df_export[cols_tiempo].round(2)

        df_export = df_export.rename(columns={
            'Cliente': 'ID_Cliente',
            'Cajero': 'Num_Cajero_Asignado',
            'T.Entre': 'Tiempo_Entre_Llegadas_min',
            'H.Llegada': 'Cronómetro_Llegada',
            'H.Inicio': 'Cronómetro_Atención',
            'T.Espera': 'Minutos_Esperando_Fila',
            'Operación': 'Tipo_Operación',
            'T.Servicio': 'Tiempo_Transacción_min',
            'H.Salida': 'Cronómetro_Salida',
            'T.Sistema': 'Total_Tiempo_Sucursal_min'
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
        try:
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
        except Exception as e:
            st.error(f"❌ Error al generar PDF: {e}")

    with col_btn3:
        with st.popover("📧 Enviar por Correo", use_container_width=True):
            st.markdown("**Datos de Envío (Usa Gmail con Contraseña de Aplicación):**")
            remitente = st.text_input("Tu Email:", placeholder="ejemplo@gmail.com")
            password = st.text_input("App Pass:", type="password",
                                     placeholder="Contraseña de aplicación")
            destino = st.text_input("Destino:", placeholder="destino@correo.com")

            if st.button("Enviar Ahora", type="primary", use_container_width=True):
                if not all([remitente, password, destino]):
                    st.error("⚠️ Faltan datos. Llena todos los campos.")
                else:
                    try:
                        msg = MIMEMultipart()
                        msg['From'] = remitente
                        msg['To'] = destino
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
