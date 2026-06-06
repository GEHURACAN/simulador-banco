import sys
import asyncio

# --- ESTE ES EL PARCHE PARA EL ERROR DE PANTALLA NEGRA EN WINDOWS ---
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
# -------------------------------------------------------------------

import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from fpdf import FPDF
import os
import warnings

warnings.filterwarnings("ignore")

# Configuración de la página web
st.set_page_config(
    page_title="Simulador Bancario Profesional",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 1. LÓGICA DE SIMULACIÓN
# ==========================================
def simular_banco_multicajero(num_clientes, num_cajeros, modo_demanda):
    media_llegada = 1.0 if modo_demanda == 1 else 20.0
    datos = []
    h_llegada_ant = 0
    tiempos_libres_cajeros = [0.0] * num_cajeros
    ocio_acumulado_cajeros = [0.0] * num_cajeros

    for i in range(1, num_clientes + 1):
        ri_llegadas = np.random.rand()
        t_entre_llegadas = -media_llegada * np.log(ri_llegadas)
        h_llegada = h_llegada_ant + t_entre_llegadas

        h_inicio_posible = min(tiempos_libres_cajeros)
        id_cajero = tiempos_libres_cajeros.index(h_inicio_posible)

        if h_llegada > h_inicio_posible:
            ocio_acumulado_cajeros[id_cajero] += (h_llegada - h_inicio_posible)

        h_inicio = max(h_inicio_posible, h_llegada)

        ri_operacion = np.random.rand()
        if ri_operacion < 0.50:
            operacion = "Retiro"
        elif ri_operacion < 0.70:
            operacion = "Transferencia"
        else:
            operacion = "Depósito"

        ri_servicio = np.random.rand()
        if operacion in ["Retiro", "Transferencia"]:
            t_servicio = 1 + (2 - 1) * ri_servicio
        else:
            t_servicio = norm.ppf(ri_servicio, loc=3.0, scale=0.2)

        h_salida = h_inicio + t_servicio
        t_sistema = h_salida - h_llegada
        tiempos_libres_cajeros[id_cajero] = h_salida

        datos.append([
            i, id_cajero + 1, t_entre_llegadas, h_llegada, h_inicio,
            operacion, t_servicio, h_salida, t_sistema
        ])
        h_llegada_ant = h_llegada

    columnas = ["Cliente", "Cajero", "T.Entre", "H.Llegada", "H.Inicio",
                "Operación", "T.Servicio", "H.Salida", "T.Sistema"]

    return pd.DataFrame(datos, columns=columnas), ocio_acumulado_cajeros

# ==========================================
# 2. GENERADOR DE ANÁLISIS PROFESIONAL
# ==========================================
def generar_analisis_dinamico(df, ocio_list, num_cajeros, num_clientes):
    t_final = df['H.Salida'].max()
    prom_ocio = sum(ocio_list) / num_cajeros
    pct_tiempo_sistema = (df['T.Sistema'].mean() / t_final) * 100

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
# 3. CLASE Y GENERADOR DE PDF EJECUTIVO
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

def crear_pdf(df, analisis, prom_ocio, pct_tiempo_sistema, conteo_cajeros):
    pdf = PDFReport()
    pdf.add_page()

    azul_titulos = (41, 128, 185)
    gris_texto = (60, 60, 60)

    pdf.set_y(30)
    pdf.set_font("Arial", 'B', 12)
    pdf.set_text_color(*azul_titulos)
    pdf.cell(0, 8, txt="Resumen General Operativo", ln=True)

    pdf.set_draw_color(*azul_titulos)
    pdf.set_line_width(0.5)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)

    pdf.set_font("Arial", 'B', 10)
    pdf.set_text_color(*gris_texto)
    pdf.cell(0, 6, txt="Carga de trabajo por cajero (Muestra):", ln=True)
    pdf.set_font("Arial", '', 10)

    items_to_show = list(conteo_cajeros.items())[:15]
    for cj, cant in items_to_show:
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

    pdf.set_font("Arial", 'B', 12)
    pdf.set_text_color(*azul_titulos)
    pdf.cell(0, 8, txt="Analisis Especializado", ln=True)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)

    pdf.set_font("Arial", '', 10)
    pdf.set_text_color(*gris_texto)

    analisis_limpio = analisis.replace("ANÁLISIS OPERATIVO Y RECOMENDACIONES:\n\n", "")
    analisis_limpio = analisis_limpio.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 5, txt=analisis_limpio)

    # --- MAQUETADO: DOS GRÁFICAS EN UNA SOLA HOJA ---
    if os.path.exists("graficas_simulacion.jpg"):
        pdf.add_page()
        pdf.set_y(20)
        pdf.set_font("Arial", 'B', 12)
        pdf.set_text_color(*azul_titulos)
        pdf.cell(0, 8, txt="Matriz de Graficas Operativas (Dashboard)", ln=True)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(3)

        y_dashboard = pdf.get_y()
        pdf.image("graficas_simulacion.jpg", x=10, y=y_dashboard, w=190)
        pdf.set_y(y_dashboard + 115)

    if os.path.exists("grafica_fila_pdf.jpg"):
        pdf.set_font("Arial", 'B', 12)
        pdf.set_text_color(*azul_titulos)
        pdf.cell(0, 8, txt="Distribución Avanzada de Tiempos de Espera en Fila", ln=True)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(3)

        pdf.image("grafica_fila_pdf.jpg", x=45, y=pdf.get_y(), w=120)

    pdf.output("Reporte_Simulacion.pdf")

# ==========================================
# 4. INTERFAZ WEB RESPONSIVA (STREAMLIT)
# ==========================================

st.title("🏦 Sistema Bancario Multicajero")
st.markdown("---")

st.sidebar.header("⚙️ Parámetros de Control")
clientes = st.sidebar.number_input("Número de clientes a simular:", min_value=1, max_value=5000, value=10)
cajeros = st.sidebar.number_input("Número de cajeros activos:", min_value=1, max_value=500, value=6)
escenario = st.sidebar.selectbox("Escenario de Demanda:", [("1. ALTA DEMANDA", 1), ("2. BAJA DEMANDA", 2)])

if st.sidebar.button("▶️ Ejecutar Simulación", type="primary"):

    df, ocio_list = simular_banco_multicajero(clientes, cajeros, escenario[1])

    # --- COLUMNAS DE FILA (CAMBIO: Sufrió_Fila → Esperó_Fila) ---
    espera_calculada = df['H.Inicio'] - df['H.Llegada']
    df.insert(5, 'Esperó_Fila', np.where(espera_calculada > 0, 'Sí', 'No'))
    df.insert(6, 'T.Espera', espera_calculada)

    t_final = df['H.Salida'].max()
    pct_tiempo_sistema = (df['T.Sistema'].mean() / t_final) * 100
    prom_ocio = sum(ocio_list) / cajeros
    conteo_cajeros = df['Cajero'].value_counts().sort_index()
    analisis_texto = generar_analisis_dinamico(df, ocio_list, cajeros, clientes)

    st.session_state['df'] = df
    st.session_state['analisis'] = analisis_texto
    st.session_state['prom_ocio'] = prom_ocio
    st.session_state['pct_tiempo_sistema'] = pct_tiempo_sistema
    st.session_state['conteo'] = conteo_cajeros

    col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
    with col_kpi1:
        st.metric(label="Clientes Atendidos", value=f"{clientes}")
    with col_kpi2:
        st.metric(label="Promedio Tiempo en Sistema", value=f"{pct_tiempo_sistema:.2f}%")
    with col_kpi3:
        if prom_ocio > 59:
            st.metric(label="Ocio Promedio en Ventanilla", value=f"{(prom_ocio / 60):.2f} hrs")
        else:
            st.metric(label="Ocio Promedio en Ventanilla", value=f"{prom_ocio:.4f} min")

    st.subheader("📋 Registro Operativo Detallado")
    st.dataframe(df.round(3), use_container_width=True)

    st.subheader("📝 Evaluación del Sistema y Diagnóstico")
    st.info(analisis_texto)

    st.subheader("📊 Visualización de Datos")

    if cajeros <= 6:
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        fig.patch.set_facecolor('white')
        fig.suptitle("Dashboard Bancario", fontsize=18, fontweight='bold', y=0.98)
        fig.text(0.5, 0.51, "(MONITOREO EN TIEMPO REAL)", ha='center', va='center', fontsize=15, fontweight='bold', color='darkblue')
        plt.subplots_adjust(hspace=0.3)
        ax1, ax2, ax3, ax4 = axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]
    else:
        fig, axs = plt.subplots(1, 2, figsize=(16, 6))
        fig.patch.set_facecolor('white')
        fig.suptitle("Dashboard Bancario", fontsize=18, fontweight='bold')
        ax1, ax2 = axs[0], axs[1]

    mapa_colores = {"Depósito": '#f39c12', "Retiro": '#3498db', "Transferencia": '#e74c3c'}

    servicios_agrupados = df.groupby("Operación")["T.Servicio"].sum()
    mapa_etiquetas_serv = {"Depósito": "Depósitos", "Retiro": "Retiros", "Transferencia": "Transferencias"}
    etiquetas_serv = [mapa_etiquetas_serv.get(op, op) for op in servicios_agrupados.index]
    colores_serv = [mapa_colores.get(op, '#000000') for op in servicios_agrupados.index]
    ax1.pie(servicios_agrupados, labels=etiquetas_serv, autopct='%1.1f%%', startangle=140, colors=colores_serv, wedgeprops={'edgecolor': 'grey', 'linewidth': 1.5})
    ax1.set_title("Distribución de Tiempo de Servicio", fontweight='bold')

    try:
        ocio_series = pd.Series(ocio_list)

        if ocio_series.max() == 0 or ocio_series.max() == ocio_series.min():
            val_unico = ocio_series.max()
            if val_unico > 59:
                etiqueta = f"{(val_unico/60):.2f} HRS"
            else:
                etiqueta = f"{val_unico:.2f} min"
            conteo_ocio = pd.Series({etiqueta: len(ocio_series)})
            labels_ocio = conteo_ocio.index
        else:
            bins = pd.cut(ocio_series, bins=4)
            conteo_ocio = bins.value_counts().sort_index()
            conteo_ocio = conteo_ocio[conteo_ocio > 0]

            labels_ocio = []
            for b in conteo_ocio.index:
                limite_inf = max(0, b.left)
                limite_sup = b.right
                if limite_sup > 59:
                    labels_ocio.append(f"De {(limite_inf/60):.2f} a {(limite_sup/60):.2f} hrs")
                else:
                    labels_ocio.append(f"De {limite_inf:.2f} a {limite_sup:.2f} min")

        colores_ocio = plt.cm.Pastel1(np.linspace(0, 1, len(conteo_ocio)))
        ax2.pie(conteo_ocio.values, labels=labels_ocio, autopct='%1.1f%%', startangle=140, colors=colores_ocio, wedgeprops={'edgecolor': 'gray'})
        ax2.set_title("Distribución de Tiempo de Ocio (Por Rangos)", fontweight="bold")

    except Exception as e:
        st.error(f"⚠️ Hubo un error interno al dibujar el pastel: {e}")

    if cajeros <= 6:
        ax3.set_title("Diagrama de Gantt - Monitoreo de Operaciones", fontweight='bold')
        for idx, row in df.iterrows():
            ax3.barh(row['Cajero'], row['T.Servicio'], left=row['H.Inicio'], color=mapa_colores.get(row['Operación'], '#000000'), edgecolor='grey', height=0.6)
        ax3.set_yticks(range(1, cajeros + 1))
        ax3.set_yticklabels([f"Cajero {i}" for i in range(1, cajeros + 1)], fontsize=9)
        handles = [mpatches.Patch(color=mapa_colores[k], label=k) for k in mapa_colores]
        ax3.legend(handles=handles, loc='lower right')
        ax3.grid(axis='x', linestyle='--', alpha=0.5)

        ax4.set_title("Distribución del Tiempo Total en Sistema", fontweight='bold')
        ax4.hist(df['T.Sistema'], bins=12, color="#3498db", edgecolor='white', alpha=0.85)
        ax4.axvline(df['T.Sistema'].mean(), color='red', linestyle='dashed', linewidth=2, label=f"Media: {df['T.Sistema'].mean():.2f} min")
        ax4.legend()
        ax4.grid(axis='y', linestyle='--', alpha=0.5)

        if clientes > 10:
            axs[1, 0].remove()
            gs = axs[1, 1].get_gridspec()
            axs[1, 1].set_subplotspec(gs[1, :])

    plt.tight_layout()
    plt.savefig("graficas_simulacion.jpg", bbox_inches='tight', dpi=120, facecolor='white', transparent=False)
    st.pyplot(fig)

    fig_pdf, ax_pdf = plt.subplots(figsize=(6, 3))
    ax_pdf.hist(df['T.Espera'], bins=10, color="#2ecc71", edgecolor='black', alpha=0.8)
    ax_pdf.set_title("Distribución de Tiempos de Espera en Fila", fontweight='bold', fontsize=10)
    ax_pdf.set_xlabel("Tiempo de espera (minutos)", fontsize=8)
    ax_pdf.set_ylabel("Cantidad de Clientes", fontsize=8)
    ax_pdf.tick_params(axis='both', which='major', labelsize=8)
    ax_pdf.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("grafica_fila_pdf.jpg", bbox_inches='tight', dpi=120, facecolor='white')
    plt.close(fig_pdf)

# ==========================================
# 5. BOTONES DE EXPORTACIÓN NATIVA
# ==========================================
if 'df' in st.session_state:
    st.markdown("---")
    st.subheader("📥 Exportación de Entregables Profesionales")

    col_btn1, col_btn2, col_btn3 = st.columns(3)

    with col_btn1:
        df_export = st.session_state['df'].copy()
        df_export['Eficiencia_Atención_%'] = ((df_export['T.Servicio'] / (df_export['T.Espera'] + df_export['T.Servicio'])) * 100).round(2)

        hora_apertura = pd.to_datetime('2026-06-05 09:00:00')
        df_export['Hora_Llegada_Reloj'] = hora_apertura + pd.to_timedelta(df_export['H.Llegada'], unit='m')
        df_export['Hora_Salida_Reloj'] = hora_apertura + pd.to_timedelta(df_export['H.Salida'], unit='m')

        df_export['Hora_Llegada_Reloj'] = df_export['Hora_Llegada_Reloj'].dt.strftime('%H:%M:%S')
        df_export['Hora_Salida_Reloj'] = df_export['Hora_Salida_Reloj'].dt.strftime('%H:%M:%S')

        cols_tiempo = ['T.Entre', 'H.Llegada', 'H.Inicio', 'T.Espera', 'T.Servicio', 'H.Salida', 'T.Sistema']
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

        # CAMBIO: Sufrió_Fila → Esperó_Fila en columnas finales del CSV
        columnas_finales = [
            'ID_Cliente', 'Hora_Llegada_Reloj', 'Hora_Salida_Reloj', 'Num_Cajero_Asignado',
            'Esperó_Fila', 'Minutos_Esperando_Fila', 'Tipo_Operación', 'Tiempo_Transacción_min',
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

            remitente = st.text_input("Tu Email:", placeholder="ejemplo@gmail.com")
            password = st.text_input("App Pass:", type="password", placeholder="Contraseña de aplicación")
            destino = st.text_input("Destino:", placeholder="destino@correo.com")

            if st.button("Enviar Ahora", type="primary", use_container_width=True):
                if remitente == "" or password == "" or destino == "":
                    st.error("⚠️ Faltan datos. Llena todos los campos.")
                else:
                    try:
                        import smtplib
                        from email.mime.multipart import MIMEMultipart
                        from email.mime.base import MIMEBase
                        from email.mime.text import MIMEText
                        from email import encoders

                        msg = MIMEMultipart()
                        msg['From'] = remitente
                        msg['To'] = destino
                        msg['Subject'] = "Reporte Ejecutivo - Simulador Bancario"

                        cuerpo = "Hola. Adjunto encontrarás el reporte generado automáticamente por el Simulador Bancario."
                        msg.attach(MIMEText(cuerpo, 'plain'))

                        nombre_archivo = "Reporte_Simulacion.pdf"
                        with open(nombre_archivo, "rb") as adjunto:
                            parte = MIMEBase("application", "octet-stream")
                            parte.set_payload(adjunto.read())

                        encoders.encode_base64(parte)
                        parte.add_header("Content-Disposition", f"attachment; filename= {nombre_archivo}")
                        msg.attach(parte)

                        servidor = smtplib.SMTP('smtp.gmail.com', 587)
                        servidor.starttls()
                        servidor.login(remitente, password)
                        texto_final = msg.as_string()
                        servidor.sendmail(remitente, destino, texto_final)
                        servidor.quit()

                        st.success(f"¡El PDF ha sido enviado con éxito a {destino}!")
                        st.balloons()

                    except smtplib.SMTPAuthenticationError:
                        st.error("❌ Error de Autenticación: Verifica tus datos.")
                    except Exception as e:
                        st.error(f"❌ Ocurrió un error inesperado: {e}")
