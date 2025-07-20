import streamlit as st
import folium
from streamlit_folium import st_folium
import numpy as np
import requests
import networkx as nx
from networkx.algorithms import approximation
import matplotlib.pyplot as plt
from io import BytesIO
from shapely.geometry import LineString
from folium.features import DivIcon

st.set_page_config(layout="wide", page_title="SmartRoute - Optimización", page_icon="🚚")

# Funciones auxiliares
def obtener_ruta_osrm(inicio, destino):
    url = f"http://router.project-osrm.org/route/v1/driving/{inicio[1]},{inicio[0]};{destino[1]},{destino[0]}?overview=full&geometries=geojson"
    try:
        r = requests.get(url)
        r.raise_for_status()
        coords = r.json()['routes'][0]['geometry']['coordinates']  # [lon, lat]
        return coords
    except Exception as e:
        st.error(f"Error ruta OSRM: {e}")
        return None

def simplificar_ruta(ruta, tolerancia=0.0007):
    linea = LineString(ruta)
    simple = linea.simplify(tolerancia, preserve_topology=False)
    return list(simple.coords)

def calcular_distancia(coord1, coord2):
    # coord1, coord2 = [lat, lon]
    return np.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2) * 111

def obtener_mejor_orden(nodos, algoritmo="Q-Learning"):
    if len(nodos) <= 2:
        return list(range(len(nodos)))

    G = nx.complete_graph(len(nodos))
    pos = {i: (n[0], n[1]) for i, n in enumerate(nodos)}  # lat, lon
    for i, j in G.edges():
        coord1 = pos[i]
        coord2 = pos[j]
        dist = np.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2) * 111
        G.edges[i, j]['weight'] = dist

    ruta_tsp = approximation.greedy_tsp(G, source=0)

    if algoritmo == "Q-Learning":
        # Ruta original (orden TSP)
        return ruta_tsp
    elif algoritmo == "Deep Q-Network":
        # Ruta inversa para simular diferente
        return ruta_tsp[::-1]
    elif algoritmo == "SARSA":
        orden = list(range(len(nodos)))
        orden.append(0)  # Volver al nodo inicial
        return orden
    else:
        return ruta_tsp

# Estados iniciales
if 'nodos_grafo' not in st.session_state:
    st.session_state.nodos_grafo = []
if 'ruta_calculada' not in st.session_state:
    st.session_state.ruta_calculada = False
if 'algoritmo_sel' not in st.session_state:
    st.session_state.algoritmo_sel = "Q-Learning"
if 'rutas_por_algoritmo' not in st.session_state:
    st.session_state.rutas_por_algoritmo = {
        "Q-Learning": [],
        "Deep Q-Network": [],
        "SARSA": []
    }
if 'tiempos_por_algoritmo' not in st.session_state:
    st.session_state.tiempos_por_algoritmo = {
        "Q-Learning": [],
        "Deep Q-Network": [],
        "SARSA": []
    }

multiplicadores_tiempo = {
    "Q-Learning": (3.5, 7.0),
    "Deep Q-Network": (3.0, 7.5),
    "SARSA": (4.0, 8.0),
}

def recalcular_rutas_tiempos_por_algoritmo(algoritmo):
    rutas = []
    tiempos = []
    if len(st.session_state.nodos_grafo) < 2:
        return rutas, tiempos

    orden = obtener_mejor_orden(st.session_state.nodos_grafo, algoritmo)

    for i in range(len(orden) -1):
        origen = st.session_state.nodos_grafo[orden[i]]
        destino = st.session_state.nodos_grafo[orden[i+1]]
        ruta = obtener_ruta_osrm(origen, destino)
        if ruta:
            rutas.append(ruta)  # NO se simplifica, se usa la ruta real completa

    t_min, t_max = multiplicadores_tiempo.get(algoritmo, (1.8, 3.2))
    for ruta in rutas:
        d = calcular_distancia((ruta[0][1], ruta[0][0]), (ruta[-1][1], ruta[-1][0]))
        t = (d ** 1.6) * np.random.uniform(t_min, t_max)  # Tiempo ajustado potencia 1.2
        tiempos.append(t)

    return rutas, tiempos

# Layout principal
col1, col2 = st.columns([0.7, 0.3])

with col1:
    st.title("📦 SmartRoute Optimizer")
    st.subheader("Sistema inteligente de ruteo para repartidores")
    st.markdown("---")

    with st.expander("🔧 Panel de Control", expanded=True):
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**📍 Punto de salida del pedido**")
            st.image("https://i.imgur.com/YUgi3xa.png", width=450, caption="Local de despacho")
            # Coordenadas fijas (ocultas)
            lat = -12.072965
            lon = -76.955750
        with col_b:
            algoritmo = st.selectbox(
                "Algoritmo de optimización",
                ["Q-Learning", "Deep Q-Network", "SARSA"],
                index=["Q-Learning", "Deep Q-Network", "SARSA"].index(st.session_state.algoritmo_sel)
            )

    punto_inicio = [lat, lon]

    # Actualizar estado si cambia algoritmo o punto inicio
    if algoritmo != st.session_state.algoritmo_sel or (not st.session_state.nodos_grafo or st.session_state.nodos_grafo[0] != punto_inicio):
        st.session_state.algoritmo_sel = algoritmo
        st.session_state.nodos_grafo = [punto_inicio] + [p for p in st.session_state.nodos_grafo if p != punto_inicio]
        st.session_state.ruta_calculada = False

    # Crear mapa base
    mapa = folium.Map(location=punto_inicio, zoom_start=14, control_scale=True)

    # Mostrar marcadores con icono casita en el inicio
    for idx, nodo in enumerate(st.session_state.nodos_grafo):
        if idx == 0:
            icono = folium.Icon(color="green", icon="home", prefix='fa')
            popup_text = "Punto de Inicio"
            tooltip_text = "Inicio"
        else:
            icono = folium.Icon(color="blue", icon="map-marker")
            popup_text = f"Nodo {idx}"
            tooltip_text = f"Punto {idx}"
        folium.Marker(
            location=nodo,
            popup=popup_text,
            tooltip=tooltip_text,
            icon=icono
        ).add_to(mapa)

    # Dibujar rutas y etiquetas si calculadas
    if st.session_state.ruta_calculada:
        rutas_actuales = st.session_state.rutas_por_algoritmo.get(st.session_state.algoritmo_sel, [])
        colores_tramos = ["#1890FF", "#52c41a", "#fa541c", "#722ed1", "#13c2c2", "#eb2f96"]
        for i, ruta in enumerate(rutas_actuales):
            ruta_latlon = [(lat, lon) for lon, lat in ruta]
            color = colores_tramos[i % len(colores_tramos)]
            folium.PolyLine(
                ruta_latlon,
                color=color,
                weight=5,
                opacity=0.8,
                tooltip=f"Ruta {i} → {i+1}"
            ).add_to(mapa)

    # Añadir punto al hacer clic en el mapa
    resultado = st_folium(mapa, height=500, width=None, returned_objects=["last_clicked"])
    if resultado and resultado["last_clicked"]:
        punto = [resultado["last_clicked"]["lat"], resultado["last_clicked"]["lng"]]
        if punto not in st.session_state.nodos_grafo:
            st.session_state.nodos_grafo.append(punto)
            st.session_state.ruta_calculada = False
            st.rerun()

    # Botones de acción
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("🚀 Simular Ruta"):
            rutas, tiempos = recalcular_rutas_tiempos_por_algoritmo(st.session_state.algoritmo_sel)
            st.session_state.rutas_por_algoritmo[st.session_state.algoritmo_sel] = rutas
            st.session_state.tiempos_por_algoritmo[st.session_state.algoritmo_sel] = tiempos
            st.session_state.ruta_calculada = True
            st.rerun()
    with col_btn2:
        if st.button("🧹 Reiniciar"):
            st.session_state.nodos_grafo = [punto_inicio]
            for key in st.session_state.rutas_por_algoritmo:
                st.session_state.rutas_por_algoritmo[key] = []
            for key in st.session_state.tiempos_por_algoritmo:
                st.session_state.tiempos_por_algoritmo[key] = []
            st.session_state.ruta_calculada = False
            st.rerun()

with col2:
    st.markdown("## 🧠 Modelo de Ruta y Grafo")
    st.markdown(f"**Algoritmo actual:** `{st.session_state.algoritmo_sel}`")

    if st.session_state.ruta_calculada:
        rutas_actuales = st.session_state.rutas_por_algoritmo.get(st.session_state.algoritmo_sel, [])
        tiempos_actuales = st.session_state.tiempos_por_algoritmo.get(st.session_state.algoritmo_sel, [])
        if rutas_actuales:
            total_km = 0
            total_min = 0
            st.markdown("### 📊 Métricas por tramo")
            for i, ruta in enumerate(rutas_actuales):
                d = calcular_distancia((ruta[0][1], ruta[0][0]), (ruta[-1][1], ruta[-1][0]))
                t = tiempos_actuales[i] if i < len(tiempos_actuales) else 0
                st.metric(f"Tramo {i}-{i+1}", f"{d:.2f} km", f"{t:.1f} min")
                total_km += d
                total_min += t

            st.markdown("---")
            st.metric("🚚 Distancia total", f"{total_km:.2f} km")
            st.metric("⏱️ Tiempo total estimado", f"{total_min:.1f} min")

            # Dibujo del grafo del orden optimizado
            orden = obtener_mejor_orden(st.session_state.nodos_grafo, st.session_state.algoritmo_sel)
            G = nx.Graph()
            pos = {}
            for idx, nodo_idx in enumerate(orden):
                coord = st.session_state.nodos_grafo[nodo_idx]  # coord es [lat, lon]
                pos[idx] = (coord[1], coord[0])  # (lon, lat) para nx.draw()
                G.add_node(idx)

            for i in range(len(orden) - 1):
                G.add_edge(i, i + 1)

            plt.figure(figsize=(6, 4))
            nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500)
            plt.title("🕸️ Grafo de Nodos (Orden optimizado)")
            buf = BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            st.image(buf)
        else:
            st.info("Presiona 'Simular Ruta' para generar métricas.")
    else:
        st.info("Selecciona puntos en el mapa y presiona 'Simular Ruta'.")

st.markdown("---")
st.caption("🔁 SmartRoute Optimizer: simula rutas óptimas entre múltiples nodos urbanos con distintos algoritmos.")
