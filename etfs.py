import requests
from bs4 import BeautifulSoup
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from googletrans import Translator
import numpy as np

# Inicializar el traductor de Google
translator = Translator()

# Lista de nombres de ETFs y sus símbolos
etf_nombres = [
    "AZ QQQ NASDAQ 100",
    "AZ SPDR S&P 500 ETF TRUST",
    "AZ SPDR DJIA TRUST",
    "AZ VANGUARD EMERGING MARKET ETF",
    "AZ FINANCIAL SELECT SECTOR SPDR",
    "AZ HEALTH CARE SELECT SECTOR",
    "AZ DJ US HOME CONSTRUCT",
    "AZ SILVER TRUST",
    "AZ MSCI TAIWAN INDEX FD",
    "AZ MSCI UNITED KINGDOM",
    "AZ MSCI SOUTH KOREA IND",
    "AZ MSCI EMU",
    "AZ MSCI JAPAN INDEX FD",
    "AZ MSCI CANADA",
    "AZ MSCI GERMANY INDEX",
    "AZ MSCI AUSTRALIA INDEX",
    "AZ BARCLAYS AGGREGATE"
]

# Tickers correspondientes a los ETFs
etf_tickers = [
    "QQQ",
    "SPY",
    "DIA",
    "VWO",
    "XLF",
    "XLV",
    "ITB",
    "SLV",
    "EWT",
    "EWU",
    "EWY",
    "EZU",
    "EWJ",
    "EWC",
    "EWG",
    "EWA",
    "AGG"
]

def obtener_fechas_ultimos_diez_anos():
    """Obtiene las fechas de inicio y fin para los últimos 10 años."""
    fecha_fin = datetime.now()
    fecha_inicio = fecha_fin - timedelta(days=365 * 10)  # 10 años
    return fecha_inicio.strftime("%Y-%m-%d"), fecha_fin.strftime("%Y-%m-%d")

def descargar_datos_historicos(tickers):
    """Descarga los precios históricos de los últimos 10 años para una lista de tickers."""
    fecha_inicio, fecha_fin = obtener_fechas_ultimos_diez_anos()
    precios_historicos = {}
    
    for ticker in tickers:
        try:
            accion = yf.Ticker(ticker)
            datos = accion.history(start=fecha_inicio, end=fecha_fin)
            precios_historicos[ticker] = datos
        except Exception as e:
            print(f"Error al descargar datos para {ticker}: {e}")
            precios_historicos[ticker] = None
    
    return precios_historicos

def obtener_data(ticker):
    """Obtiene el nombre corto y la descripción larga de un ETF dado su ticker."""
    try:
        accion = yf.Ticker(ticker)
        info = accion.info
        nombre_corto = info.get('shortName', 'No disponible')
        descripcion_larga = info.get('longBusinessSummary', 'Descripción no disponible')
        descripcion_traducida = traducir_texto(descripcion_larga)  # Traducir la descripción
        return nombre_corto, descripcion_traducida
    except Exception as e:
        print(f"Error al obtener datos para {ticker}: {e}")
        return 'No disponible', 'Descripción no disponible'

def traducir_texto(texto):
    """Traduce un texto al español utilizando Google Translate."""
    try:
        traduccion = translator.translate(texto, dest='es')
        return traduccion.text
    except Exception as e:
        print(f"Error al traducir el texto: {e}")
        return 'Traducción no disponible'

def obtener_precio_actual(ticker):
    """Obtiene el precio actual de un ETF dado su ticker."""
    try:
        accion = yf.Ticker(ticker)
        precio_actual = accion.history(period='1d')['Close'].iloc[-1]  # Obtener el precio de cierre más reciente
        return precio_actual
    except Exception as e:
        print(f"Error al obtener el precio actual para {ticker}: {e}")
        return None

def rendimiento_logaritmico(precios_historicos):
    """Calcula el rendimiento logarítmico anualizado a partir de los precios históricos."""
    precios = precios_historicos['Close']
    primer_precio = precios.iloc[0]
    ultimo_precio = precios.iloc[-1]
    
    rendimiento_log = np.log(ultimo_precio / primer_precio)
    rendimiento_log_anualizado = rendimiento_log / 10  # Dividir entre 10 años
    
    return rendimiento_log_anualizado

def calcular_riesgo_promedio(precios_historicos):
    """Calcula el riesgo promedio (desviación estándar anualizada) basado en precios de cierre históricos."""
    precios = precios_historicos['Close']
    rendimientos_diarios = np.log(precios / precios.shift(1)).dropna()
    desviacion_diaria = rendimientos_diarios.std()
    riesgo_promedio_anualizado = desviacion_diaria * np.sqrt(252)  # 252 días de negociación en un año
    return riesgo_promedio_anualizado

def calcular_ratio_riesgo_rendimiento(rendimiento_anualizado, riesgo_promedio):
    """Calcula el ratio riesgo-rendimiento."""
    if riesgo_promedio > 0:
        return rendimiento_anualizado / riesgo_promedio
    else:
        return None

def rendimiento_y_riesgo_por_periodo(precios_historicos, periodo):
    """Calcula el rendimiento y riesgo para un periodo específico."""
    try:
        if periodo == '1m':
            datos_periodo = precios_historicos.last('1M')
        elif periodo == '3m':
            datos_periodo = precios_historicos.last('3M')
        elif periodo == '6m':
            datos_periodo = precios_historicos.last('6M')
        elif periodo == '1y':
            datos_periodo = precios_historicos.last('1Y')
        elif periodo == 'YTD':
            datos_periodo = precios_historicos[precios_historicos.index >= datetime.now().replace(month=1, day=1)]
        elif periodo == '3y':
            datos_periodo = precios_historicos.last('3Y')
        elif periodo == '5y':
            datos_periodo = precios_historicos.last('5Y')
        elif periodo == '10y':
            datos_periodo = precios_historicos.last('10Y')
        else:
            raise ValueError("Periodo no reconocido.")

        # Calcular el rendimiento logarítmico
        rendimiento_log = np.log(datos_periodo['Close'].iloc[-1] / datos_periodo['Close'].iloc[0])
        rendimiento_anualizado = rendimiento_log / (datos_periodo.shape[0] / 252)  # Ajustar por días de negociación

        # Calcular el riesgo
        rendimientos_diarios = np.log(datos_periodo['Close'] / datos_periodo['Close'].shift(1)).dropna()
        desviacion_diaria = rendimientos_diarios.std()
        riesgo_anualizado = desviacion_diaria * np.sqrt(252)

        return rendimiento_anualizado, riesgo_anualizado
    except Exception as e:
        print(f"Error al calcular rendimiento y riesgo para el periodo {periodo}: {e}")
        return None, None

# Variable para almacenar la información de los ETFs
ETFs_Data = []

# Descargar precios históricos para todos los tickers
precios_historicos_todos = descargar_datos_historicos(etf_tickers)

# Iterar sobre los ETFs y obtener la información
for nombre, ticker in zip(etf_nombres, etf_tickers):
    nombre_corto, descripcion_larga = obtener_data(ticker)
    
    # Obtener los precios históricos del ticker actual
    precios_historicos = precios_historicos_todos.get(ticker)
    
    # Obtener el precio actual
    precio_actual = obtener_precio_actual(ticker)

    # Calcular el rendimiento logarítmico anualizado
    if precios_historicos is not None and not precios_historicos.empty:
        rendimiento_log_geom = rendimiento_logaritmico(precios_historicos)
        riesgo_promedio = calcular_riesgo_promedio(precios_historicos)
        ratio_riesgo_rendimiento = calcular_ratio_riesgo_rendimiento(rendimiento_log_geom, riesgo_promedio)

        # Calcular rendimiento y riesgo para diferentes periodos
        periodos = ['1m', '3m', '6m', '1y', 'YTD', '3y', '5y', '10y']
        rendimientos = {}
        riesgos = {}
        
        for periodo in periodos:
            rendimiento, riesgo = rendimiento_y_riesgo_por_periodo(precios_historicos, periodo)
            rendimientos[periodo] = rendimiento
            riesgos[periodo] = riesgo

    else:
        rendimiento_log_geom = None
        riesgo_promedio = None
        ratio_riesgo_rendimiento = None
        rendimientos = {periodo: None for periodo in periodos}
        riesgos = {periodo: None for periodo in periodos}
    
    # Añadir la información a la lista de ETFs
    ETFs_Data.append({
        "nombre": nombre,
        "simbolo": ticker,
        "nombre_corto": nombre_corto,
        "descripcion_larga": descripcion_larga,
        "precios_historicos": precios_historicos,
        "precio_actual": precio_actual,
        "rendimiento_log_geom": rendimiento_log_geom,
        "riesgo_promedio": riesgo_promedio,
        "ratio_riesgo_rendimiento": ratio_riesgo_rendimiento,
        "rendimientos": rendimientos,
        "riesgos": riesgos
    })


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
        st.markdown("<h3 style='color: #1E3A8A;'>Comparación de Riesgo y Rendimiento</h3>", unsafe_allow_html=True)
        
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

        # Inicializar variable para visualizar como tabla o gráfica
        visualizar_como_tabla = st.radio("Selecciona la vista para la comparación:", ("Tabla", "Gráfica"), key="radio_comparacion")

        # Mostrar la tabla o la gráfica según la opción seleccionada
        if visualizar_como_tabla == "Tabla":
            st.markdown("<style>div.stDataframe > div > div > div > div:nth-child(1) { font-weight: bold; }</style>", unsafe_allow_html=True)
            st.dataframe(df_comparacion.set_index("ETF").style.set_table_attributes('style="background-color: #B0C4DE;"').set_table_styles(
                [{'selector': 'th', 'props': [('font-weight', 'bold')]}]
            ))

        else:
            # Graficar la comparación de rendimiento y riesgo
            # Primero, crear un nuevo DataFrame para la gráfica con valores numéricos
            comparacion_data_numeric = {
                "ETF": etfs_seleccionados,
                "Rendimiento": [
                    next((etf['rendimientos'].get(periodo_seleccionado, None) for etf in ETFs_Data if etf['nombre'] == etf_name), None) for etf_name in etfs_seleccionados
                ],
                "Riesgo": [
                    next((etf['riesgos'].get(periodo_seleccionado, None) for etf in ETFs_Data if etf['nombre'] == etf_name), None) for etf_name in etfs_seleccionados
                ]
            }
            
            df_comparacion_numeric = pd.DataFrame(comparacion_data_numeric)

            # Mantener los valores numéricos para la gráfica
            df_comparacion_numeric["Rendimiento"] = df_comparacion_numeric["Rendimiento"].replace({"No disponible": None}).astype(float)
            df_comparacion_numeric["Riesgo"] = df_comparacion_numeric["Riesgo"].replace({"No disponible": None}).astype(float)

            df_comparacion_melted = pd.melt(df_comparacion_numeric, id_vars='ETF', var_name='Tipo', value_name='Valor')

            # Verificar si hay valores para graficar
            if df_comparacion_melted['Valor'].notnull().any():
                # Graficar la comparación
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(data=df_comparacion_melted.dropna(), x='ETF', y='Valor', hue='Tipo', palette='Blues', ax=ax)
                ax.set_title('Comparación de Rendimiento y Riesgo', fontsize=16)
                ax.set_ylabel('Valor (%)', fontsize=12)
                ax.set_xlabel('ETF', fontsize=12)
                ax.legend(title='Tipo')
                st.pyplot(fig)
            else:
                st.markdown("No hay datos disponibles para graficar rendimiento y riesgo.")

else:
    st.markdown("Por favor, selecciona al menos un ETF para ver los detalles.")
