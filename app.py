import requests
from bs4 import BeautifulSoup
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from data import ETFs_Data, descargar_datos_historicos

def calcular_valor_futuro(inversion_inicial, rendimiento, periodos):
    """
    Calcula el valor futuro de una inversión utilizando la fórmula del interés compuesto.
    
    Args:
    inversion_inicial (float): Monto de la inversión inicial.
    rendimiento (float): Tasa de rendimiento por periodo (en formato decimal).
    periodos (float): Número de periodos de inversión.

    Returns:
    float: Valor futuro de la inversión.
    """
    return inversion_inicial * ((1 + rendimiento) ** periodos)

# Función para obtener noticias de Finviz
def get_finviz_news(etf_ticker, limit=3):
    url = f'https://finviz.com/quote.ashx?t={etf_ticker}'
    response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
    soup = BeautifulSoup(response.content, 'html.parser')
    
    news_table = soup.find('table', class_='fullview-news-outer')
    headlines = []

    for index, row in enumerate(news_table.find_all('tr')):
        if index >= limit:
            break
        news_item = row.find_all('td')
        date_time = news_item[0].text.strip()
        title = news_item[1].a.text.strip()
        link = news_item[1].a['href']

        headlines.append({
            'date_time': date_time, 
            'title': title, 
            'link': link
        })
    
    return headlines

# Función para formatear etiquetas
def formato_etiqueta(titulo, valor):
    return f"<strong style='font-size: 18px;'>{titulo}:</strong> {valor}"

# Establecer un tema
st.set_page_config(page_title="Análisis de ETFs", layout="wide")

# Título de la aplicación
st.markdown("<h1 style='color: darkblue;'>Análisis de ETFs 📈</h1>", unsafe_allow_html=True)
st.markdown("Explora el rendimiento y los detalles de los ETFs más relevantes de Allianz Patrimonial.")
st.markdown("---")

# Estilo de la barra lateral
st.sidebar.markdown(
    "<h3 style='color: darkred;'>Selecciona uno o más ETFs:</h3>",
    unsafe_allow_html=True
)
etfs_seleccionados = st.sidebar.multiselect(
    "",  # Deja el campo de etiqueta vacío
    options=[etf['nombre'] for etf in ETFs_Data],
    default=[]
)

# Verificar si hay algún ETF seleccionado
if etfs_seleccionados:
    # Descargar precios históricos para los ETFs seleccionados
    tickers_seleccionados = [etf_info['simbolo'] for etf_info in ETFs_Data if etf_info['nombre'] in etfs_seleccionados]
    precios_historicos_todos = descargar_datos_historicos(tickers_seleccionados)

    for etf_name in etfs_seleccionados:
        etf_info = next((etf for etf in ETFs_Data if etf['nombre'] == etf_name), None)
        if etf_info:
            # Extraer variables reutilizables
            nombre = etf_info['nombre']
            simbolo = etf_info['simbolo']
            nombre_corto = etf_info['nombre_corto']
            precio_actual = etf_info['precio_actual']  # Obtener el precio actual

            # Mostrar el nombre del ETF como subheader
            st.markdown(f"<h3 style='color: #1E3A8A;'>{etf_info['nombre']}</h3>", unsafe_allow_html=True)

            # Mostrar la información del ETF seleccionado con columnas ajustadas
            col1, col2 = st.columns([2, 1])  
            with col1:
                st.markdown(formato_etiqueta("Símbolo", simbolo), unsafe_allow_html=True)
                st.markdown(f"<div style='text-align: justify;'>{formato_etiqueta('Descripción', etf_info['descripcion_larga'])}</div>", unsafe_allow_html=True)
                # Precio actual
                st.write("")
                if precio_actual is not None:
                    st.markdown(formato_etiqueta("Precio Actual", f"${precio_actual:.2f}"), unsafe_allow_html=True)
                else:
                    st.markdown(formato_etiqueta("Precio Actual", "No disponible"), unsafe_allow_html=True)

            with col2:
                # Rendimiento
                rendimiento = etf_info['rendimiento_log_geom']
                if rendimiento is not None:
                    st.markdown(formato_etiqueta("Rendimiento Anualizado", f"{rendimiento:.2%}"), unsafe_allow_html=True)
                else:
                    st.markdown(formato_etiqueta("Rendimiento Anualizado", "No disponible"), unsafe_allow_html=True)

                # Riesgo promedio
                riesgo_promedio = etf_info['riesgo_promedio']
                if riesgo_promedio is not None:
                    st.markdown(formato_etiqueta("Riesgo Promedio", f"{riesgo_promedio:.2%}"), unsafe_allow_html=True)
                else:
                    st.markdown(formato_etiqueta("Riesgo Promedio", "No disponible"), unsafe_allow_html=True)

                # Ratio riesgo-rendimiento
                ratio_riesgo_rendimiento = etf_info['ratio_riesgo_rendimiento']
                if ratio_riesgo_rendimiento is not None:
                    st.markdown(formato_etiqueta("Ratio Riesgo-Rendimiento", f"{ratio_riesgo_rendimiento:.2f}"), unsafe_allow_html=True)
                else:
                    st.markdown(formato_etiqueta("Ratio Riesgo-Rendimiento", "No disponible"), unsafe_allow_html=True)

            # Espacio en blanco antes de la gráfica
            st.write("")

            # Obtener los precios históricos del ticker actual
            precios_historicos = precios_historicos_todos.get(simbolo)
            if precios_historicos is not None and not precios_historicos.empty:
                st.markdown("<h4 style='color: #1E3A8A;'>Desempeño Histórico</h4>", unsafe_allow_html=True)  
                # Graficar precios históricos
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.lineplot(x=precios_historicos.index, y=precios_historicos['Close'], ax=ax)
                ax.set_title(f"{nombre_corto} ({simbolo})", fontsize=16)
                ax.set_xlabel('Fecha', fontsize=12)
                ax.set_ylabel('Precio de Cierre', fontsize=12)
                ax.tick_params(axis='x', rotation=45)
                
                st.pyplot(fig)
            else:
                st.markdown(formato_etiqueta("Precios históricos", "No disponibles"), unsafe_allow_html=True)

            # Espacio después de la gráfica
            st.write("")

            # Mostrar el rendimiento y riesgo en diferentes periodos
            st.markdown("<h4 style='color: #1E3A8A; font-size: 21px;'>Rendimiento y Riesgo por Periodo</h4>", unsafe_allow_html=True)

            # Crear un DataFrame para los rendimientos y riesgos
            periodos = ['1m', '3m', '6m', '1y', '3y', '5y', '10y']
            rendimiento_riesgo_data = {
                "Rendimiento": [etf_info['rendimientos'].get(periodo, None) for periodo in periodos],
                "Riesgo": [etf_info['riesgos'].get(periodo, None) for periodo in periodos]
            }
            df_rendimiento_riesgo = pd.DataFrame(rendimiento_riesgo_data, index=periodos)

            # Inicializar variable para visualizar como tabla o gráfica
            visualizar_como_tabla = st.radio("Selecciona la vista:", ("Tabla", "Gráfica"), key=f"radio_{simbolo}")

            # Mostrar la tabla o la gráfica según la opción seleccionada
            if visualizar_como_tabla == "Tabla":
                df_rendimiento_riesgo.index.name = 'Periodo'  # Nombrar el índice como 'Periodo'
                df_rendimiento_riesgo['Rendimiento'] = df_rendimiento_riesgo['Rendimiento'].apply(lambda x: f"{x:.2%}" if x is not None else "No disponible")
                df_rendimiento_riesgo['Riesgo'] = df_rendimiento_riesgo['Riesgo'].apply(lambda x: f"{x:.2%}" if x is not None else "No disponible")

                # Mostrar la tabla en la app con formato
                st.markdown("<style>div.stDataframe > div > div > div > div:nth-child(1) { font-weight: bold; }</style>", unsafe_allow_html=True)
                st.dataframe(df_rendimiento_riesgo.T.style.set_table_attributes('style="background-color: #B0C4DE;"').set_table_styles(
                    [{'selector': 'th', 'props': [('font-weight', 'bold')]}]
                ))  # Transponer la tabla para que sea horizontal

            else:
                # Graficar la comparación de rendimiento y riesgo
                df_rendimiento_riesgo = df_rendimiento_riesgo.reset_index()
                df_rendimiento_riesgo_melted = pd.melt(df_rendimiento_riesgo, id_vars='index', var_name='Tipo', value_name='Valor')
                plt.figure(figsize=(12, 6))
                sns.barplot(data=df_rendimiento_riesgo_melted, x='index', y='Valor', hue='Tipo')
                plt.title(f"Rendimiento y Riesgo por Periodo: {nombre}", fontsize=16)
                plt.xlabel('Periodo', fontsize=12)
                plt.ylabel('Valor', fontsize=12)
                plt.xticks(rotation=45)
                st.pyplot(plt)

            # Espacio después de la gráfica
            st.write("")

            # Botón para mostrar las noticias
            if st.button(f'Mostrar Noticias de {nombre}', key=f'noticias_{simbolo}'):
                with st.expander("Últimas Noticias"):
                    noticias = get_finviz_news(simbolo)
                    for noticia in noticias:
                        st.markdown(f"<strong>{noticia['date_time']}</strong>: [{noticia['title']}]({noticia['link']})", unsafe_allow_html=True)
                        st.markdown("---")

            st.markdown("---")  # Línea de separación entre ETFs

    # Comparación de riesgo y rendimiento de todos los ETFs seleccionados
    if len(etfs_seleccionados) > 1:
        st.markdown("<h3 style='color: #1E3A8A;'>Comparación de ETFs seleccionados</h3>", unsafe_allow_html=True)
        
        for ticker in tickers_seleccionados:
            precios_historicos_ticker = precios_historicos_todos.get(ticker)
            if precios_historicos_ticker is not None and not precios_historicos_ticker.empty:
                plt.plot(precios_historicos_ticker.index, precios_historicos_ticker['Close'], label=ticker)

        plt.title("Comparación de Precios Históricos")
        plt.xlabel("Fecha")
        plt.ylabel("Precio de Cierre")
        plt.legend()
        st.pyplot(plt.gcf())  # Mostrar la gráfica

        # Seleccionar periodo
        periodos = ['1m', '3m', '6m', '1y', '3y', '5y', '10y']
        periodo_seleccionado = st.selectbox("Selecciona un periodo:", options=periodos)

        # Crear un DataFrame para almacenar los rendimientos, riesgos y valores futuros
        comparacion_data = {
            "ETF": etfs_seleccionados,
            "Rendimiento": [
                next((etf['rendimientos'].get(periodo_seleccionado, None) for etf in ETFs_Data if etf['nombre'] == etf_name), None) for etf_name in etfs_seleccionados
            ],
            "Riesgo": [
                next((etf['riesgos'].get(periodo_seleccionado, None) for etf in ETFs_Data if etf['nombre'] == etf_name), None) for etf_name in etfs_seleccionados
            ],
            "Valor Futuro": []  # Nueva columna para el valor futuro
        }

        # Solicitar al usuario el monto de inversión inicial en formato de dinero
        inversion_inicial = st.number_input("Ingresa el monto de tu inversión inicial:", 
                                            min_value=0.0, 
                                            format="%.2f", 
                                            step=100.0, 
                                            help="Ingrese la cantidad en USD.")

        # Calcular el valor futuro para cada ETF y agregarlo a la nueva columna
        for etf_name in etfs_seleccionados:
            rendimiento_promedio = next(
                (etf['rendimientos'].get(periodo_seleccionado, None) for etf in ETFs_Data if etf['nombre'] == etf_name),
                None
            )
            if rendimiento_promedio is not None:
                rendimiento_decimal = rendimiento_promedio
                numero_periodos = {
                    '1m': 1/12,
                    '3m': 3/12,
                    '6m': 6/12,
                    '1y': 1,
                    '3y': 3,
                    '5y': 5,
                    '10y': 10
                }[periodo_seleccionado]

                # Calcular el valor futuro
                valor_futuro = calcular_valor_futuro(inversion_inicial, rendimiento_decimal, numero_periodos)
                comparacion_data["Valor Futuro"].append(valor_futuro)
            else:
                comparacion_data["Valor Futuro"].append("No disponible")

        df_comparacion = pd.DataFrame(comparacion_data)

        # Formatear los valores de la tabla como porcentajes
        df_comparacion["Rendimiento"] = df_comparacion["Rendimiento"].apply(lambda x: f"{x:.2%}" if x is not None else "No disponible")
        df_comparacion["Riesgo"] = df_comparacion["Riesgo"].apply(lambda x: f"{x:.2%}" if x is not None else "No disponible")
        df_comparacion["Valor Futuro"] = df_comparacion["Valor Futuro"].apply(lambda x: f"${x:,.2f}" if isinstance(x, (int, float)) else x)

        st.markdown("<style>div.stDataframe > div > div > div > div:nth-child(1) { font-weight: bold; }</style>", unsafe_allow_html=True)
        st.dataframe(df_comparacion.set_index("ETF").style.set_table_attributes('style="background-color: #B0C4DE;"').set_table_styles(
            [{'selector': 'th', 'props': [('font-weight', 'bold')]}]
        ))
        
else:
    st.markdown("Por favor, selecciona al menos un ETF para ver los detalles.")