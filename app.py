    key=f"cajeros_input_{st.session_state.cajeros_valor}"
)
# Sincronizar si el usuario lo cambia manualmente
if cajeros != st.session_state.cajeros_valor:
    st.session_state.cajeros_valor = cajeros

escenario = st.sidebar.selectbox(
    "Escenario de Demanda:",
    [("1. ALTA DEMANDA", 1), ("2. BAJA DEMANDA", 2)]
)

hora_apertura = st.sidebar.time_input(
    "Hora de apertura:",
    value=pd.Timestamp('09:00:00').time()
)

# ── Botones de sugerencia ── al hacer clic simulan directamente ─────────────
st.sidebar.markdown("---")
st.sidebar.markdown("**💡 Sugerencias de configuración:**")

SUGERENCIAS = [
    ("⚠️ 3", 3, "Espera moderada"),
    ("✅ 4", 4, "Punto óptimo"),
    ("👍 5", 5, "Poca espera"),
    ("📊 6", 6, "Sin espera"),
]

cols_sug = st.sidebar.columns(len(SUGERENCIAS))
for col, (etiqueta, n_cajeros, _) in zip(cols_sug, SUGERENCIAS):
    with col:
        if st.button(etiqueta, key=f"sug_{n_cajeros}", use_container_width=True):
            st.session_state.cajeros_valor = n_cajeros
            st.session_state['trigger_sim'] = {
                'clientes' : clientes,
                'cajeros'  : n_cajeros,
                'escenario': escenario,
                'hora'     : hora_apertura,
            }
            st.rerun()

st.sidebar.caption("  |  ".join([f"{n}: {desc}" for _, n, desc in SUGERENCIAS]))

# ── Botón principal ────────────────────────────────────────────────────────
if st.sidebar.button("▶️ Ejecutar Simulación", type="primary"):
    st.session_state['trigger_sim'] = {
        'clientes' : clientes,
        'cajeros'  : cajeros,
        'escenario': escenario,
        'hora'     : hora_apertura,
    }

# ── Ejecutar si hay trigger (botón principal o sugerencia) ─────────────────
if 'trigger_sim' in st.session_state and st.session_state['trigger_sim']:
    params = st.session_state.pop('trigger_sim')
    ejecutar_simulacion(
        params['clientes'],
        params['cajeros'],
        params['escenario'],
        params['hora'],
    )

# ==========================================
# 6. EXPORTACIÓN
# ==========================================
if 'df' in st.session_state:
    st.markdown("---")
    st.subheader("📥 Exportación de Entregables Profesionales")

    col_btn1, col_btn2, col_btn3 = st.columns(3)

    with col_btn1:
        df_exp = st.session_state['df'].copy()
        df_exp['Eficiencia_Atención_%'] = (
            (df_exp['T.Servicio'] / (df_exp['T.Espera'] + df_exp['T.Servicio'])) * 100
        ).round(2)

        ha = st.session_state.get('hora_apertura', pd.Timestamp('09:00:00').time())
        ha_dt = pd.to_datetime(f"2026-06-05 {ha.strftime('%H:%M:%S')}")
        df_exp['Hora_Llegada_Reloj'] = (ha_dt + pd.to_timedelta(df_exp['H.Llegada'], unit='m')).dt.strftime('%H:%M:%S')
        df_exp['Hora_Salida_Reloj']  = (ha_dt + pd.to_timedelta(df_exp['H.Salida'],  unit='m')).dt.strftime('%H:%M:%S')

        for col in ['T.Entre','H.Llegada','H.Inicio','T.Espera','T.Servicio','H.Salida','T.Sistema']:
            df_exp[col] = df_exp[col].round(2)

        df_exp = df_exp.rename(columns={
            'Cliente':'ID_Cliente', 'Cajero':'Num_Cajero_Asignado',
            'T.Entre':'Tiempo_Entre_Llegadas_min', 'H.Llegada':'Cronómetro_Llegada',
            'H.Inicio':'Cronómetro_Atención', 'T.Espera':'Minutos_Esperando_Fila',
            'Operación':'Tipo_Operación', 'T.Servicio':'Tiempo_Transacción_min',
            'H.Salida':'Cronómetro_Salida', 'T.Sistema':'Total_Tiempo_Sucursal_min'
        })
        df_exp = df_exp[[
            'ID_Cliente','Hora_Llegada_Reloj','Hora_Salida_Reloj','Num_Cajero_Asignado',
            'Esperó_Fila','Minutos_Esperando_Fila','Tipo_Operación','Tiempo_Transacción_min',
            'Eficiencia_Atención_%','Total_Tiempo_Sucursal_min'
        ]]
        csv_data = df_exp.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
        st.download_button("📊 Descargar CSV", data=csv_data,
                           file_name="Resultados_Bancarios_Auditados.csv",
                           mime="text/csv", use_container_width=True)

    with col_btn2:
        try:
            crear_pdf(st.session_state['df'], st.session_state['analisis'],
                      st.session_state['prom_ocio'], st.session_state['pct_tiempo_sistema'],
                      st.session_state['conteo'])
            with open("Reporte_Simulacion.pdf", "rb") as f:
                pdf_data = f.read()
            st.download_button("📄 Descargar PDF", data=pdf_data,
                               file_name="Reporte_Ejecutivo_Bancario.pdf",
                               mime="application/pdf", use_container_width=True)
        except Exception as e:
            st.error(f"❌ Error al generar PDF: {e}")

    with col_btn3:
        with st.popover("📧 Enviar por Correo", use_container_width=True):
            st.markdown("**Datos de Envío (Usa Gmail con Contraseña de Aplicación):**")
            remitente = st.text_input("Tu Email:",  placeholder="ejemplo@gmail.com")
            password  = st.text_input("App Pass:", type="password", placeholder="Contraseña de aplicación")
            destino   = st.text_input("Destino:",  placeholder="destino@correo.com")

            if st.button("Enviar Ahora", type="primary", use_container_width=True):
                if not all([remitente, password, destino]):
                    st.error("⚠️ Faltan datos. Llena todos los campos.")
                else:
                    try:
                        msg = MIMEMultipart()
                        msg['From']    = remitente
                        msg['To']      = destino
                        msg['Subject'] = "Reporte Ejecutivo - Simulador Bancario"
                        msg.attach(MIMEText("Hola. Adjunto encontrarás el reporte generado automáticamente por el Simulador Bancario.", 'plain'))
                        with open("Reporte_Simulacion.pdf", "rb") as adj:
                            parte = MIMEBase("application", "octet-stream")
                            parte.set_payload(adj.read())
                        encoders.encode_base64(parte)
                        parte.add_header("Content-Disposition", "attachment; filename=Reporte_Simulacion.pdf")
                        msg.attach(parte)
                        with smtplib.SMTP('smtp.gmail.com', 587) as srv:
                            srv.starttls()
                            srv.login(remitente, password)
                            srv.sendmail(remitente, destino, msg.as_string())
                        st.success(f"¡PDF enviado con éxito a {destino}!")
                        st.balloons()
                    except smtplib.SMTPAuthenticationError:
                        st.error("❌ Error de Autenticación: Verifica tus datos.")
                    except Exception as e:
                        st.error(f"❌ Error inesperado: {e}")
