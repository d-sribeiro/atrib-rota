import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import unicodedata

# Função robusta para achar coluna por nome aproximado (ignora acento, case e espaços)
def coluna_aproximada(df, nome_desejado):
    def normalize(s):
        return ''.join(
            c for c in unicodedata.normalize('NFKD', s)
            if not unicodedata.combining(c)
        ).strip().lower().replace(" ", "")
    nome_norm = normalize(nome_desejado)
    for col in df.columns:
        if normalize(col) == nome_norm:
            return col
    raise KeyError(f"Coluna '{nome_desejado}' não encontrada! Colunas disponíveis: {df.columns.tolist()}")

st.set_page_config(page_title="Atribuição de Rotas por Geolocalização", layout="wide")
st.title("Atribuição de rotas por geolocalização para transportadoras")

st.markdown("""
Faça upload dos dois arquivos necessários e insira os percentuais das transportadoras.
""")

# UPLOAD DOS ARQUIVOS
col1, col2 = st.columns(2)
with col1:
    plan_file = st.file_uploader("Base de IDs roteirizados com coordenadas (arquivo do planification)", type="csv")
with col2:
    rotas_file = st.file_uploader("Base de rotas (roadmaps agrupados)", type="csv")

# INSERÇÃO DE PERCENTUAIS
with st.expander("Inserir share de rotas de cada transportadora"):
    transportadoras = st.text_input(
        "Nomes separados por vírgula (ex: AC2 Logistica,Log Serviços,MSR,BRJTransportes,WLS CARGO)",
        value="AC2 Logistica,Log Serviços,MSR,BRJTransportes,WLS CARGO"
    )
    transportadoras = [x.strip() for x in transportadoras.split(",") if x.strip()]
    percentuais = {}
    for t in transportadoras:
        percentuais[t] = st.number_input(
            f"Share de {t} (Ex: 0.3 para 30%)",
            min_value=0.0, max_value=1.0, step=0.01, format="%.2f", value=0.0
        )

if st.button("Executar atribuição automática"):
    try:
        df_plan = pd.read_csv(plan_file, dtype=str)
        df_rotas = pd.read_csv(rotas_file, dtype=str)
        df_hist = pd.read_csv("treino.csv", dtype=str)
    except Exception as e:
        st.error(f"Erro ao carregar arquivos: {e}")
        st.stop()

    # --- AJUSTE DE COLUNAS ----
    # Para planification
    col_shipment = coluna_aproximada(df_plan, "ID de Unidad")
    col_rua = coluna_aproximada(df_plan, "Rua")
    col_numero = coluna_aproximada(df_plan, "Número")
    col_bairro = coluna_aproximada(df_plan, "Bairro")
    col_cidade = coluna_aproximada(df_plan, "Cidade")
    col_estado = coluna_aproximada(df_plan, "Estado")
    col_cep = coluna_aproximada(df_plan, "CEP")
    col_lat = coluna_aproximada(df_plan, "Latitude")
    col_lon = coluna_aproximada(df_plan, "Longitude")

    plan_cols = [col_shipment, col_rua, col_numero, col_bairro, col_cidade, col_estado, col_cep, col_lat, col_lon]
    df_plan = df_plan[plan_cols].copy()
    df_plan.rename(columns={
        col_shipment: "Shipment",
        col_rua: "Rua",
        col_numero: "Número",
        col_bairro: "Bairro",
        col_cidade: "Cidade",
        col_estado: "Estado",
        col_cep: "CEP",
        col_lat: "Latitude",
        col_lon: "Longitude"
    }, inplace=True)

    df_plan['Latitude'] = pd.to_numeric(df_plan['Latitude'], errors='coerce')
    df_plan['Longitude'] = pd.to_numeric(df_plan['Longitude'], errors='coerce')

    # Para rotas
    col_ship_comb = coluna_aproximada(df_rotas, "Shipment")
    col_rota_comb = coluna_aproximada(df_rotas, "Rota")
    df_rotas = df_rotas[[col_ship_comb, col_rota_comb]].copy()
    df_rotas.rename(columns={col_ship_comb: "Shipment", col_rota_comb: "Rota"}, inplace=True)

    # Para treino
    for col in ["Latitude", "Longitude"]:
        df_hist[col] = pd.to_numeric(df_hist[col], errors='coerce')

    # 1. Selecione as coordenadas e remova NaNs
    coords_hist = df_hist[['Latitude', 'Longitude']].dropna()

    # 2. Converta para radianos
    coords_hist_rad = np.radians(coords_hist[['Latitude', 'Longitude']].values)

    # 3. Ajuste do parâmetro eps (em radianos)
    kms_per_radian = 6371.0088
    eps_km = 4
    eps = eps_km / kms_per_radian

    # 4. Clusterize usando DBSCAN com métrica Haversine
    db = DBSCAN(eps=eps, min_samples=7, algorithm='ball_tree', metric='haversine').fit(coords_hist_rad)
    coords_hist = coords_hist.copy()
    coords_hist['cluster'] = db.labels_

    # 5. Junte os resultados
    hist_map = df_hist[["Shipment"]].merge(coords_hist[["Latitude", "Longitude", "cluster"]],
                                           left_index=True, right_index=True, how='left')
    df_hist = pd.concat([df_hist, hist_map[['cluster']]], axis=1)

    # 6. Afinidade por cluster do histórico
    cluster_affinity = (
        df_hist.groupby('cluster')['Transportadora']
        .agg(lambda x: x.value_counts().idxmax())
        .to_dict()
    )

    # ----------  NOVOS DADOS PARA ATRIBUIÇÃO -----------
    df_comb = df_rotas.merge(df_plan, on="Shipment", how="left")
    df_comb = df_comb.dropna(subset=['Latitude', 'Longitude'])

    def geo_centroid(latitudes, longitudes):
        lat_r = np.radians(latitudes)
        lon_r = np.radians(longitudes)
        x = np.cos(lat_r) * np.cos(lon_r)
        y = np.cos(lat_r) * np.sin(lon_r)
        z = np.sin(lat_r)
        x_m = x.mean()
        y_m = y.mean()
        z_m = z.mean()
        lon_centroid = np.arctan2(y_m, x_m)
        hyp = np.sqrt(x_m ** 2 + y_m ** 2)
        lat_centroid = np.arctan2(z_m, hyp)
        return np.degrees(lat_centroid), np.degrees(lon_centroid)

    # Groupby para cada grupo de 'Rota'
    centros_rota = df_comb.groupby('Rota').apply(
        lambda g: pd.Series(geo_centroid(g['Latitude'].values, g['Longitude'].values),
                            index=['Latitude', 'Longitude'])
    )

    db_r = DBSCAN(eps=0.000628, min_samples=7).fit(centros_rota.values)
    centros_rota = centros_rota.copy()
    centros_rota['cluster'] = db_r.labels_
    rota_to_cluster = centros_rota['cluster'].to_dict()

    df_comb['cluster'] = df_comb['Rota'].map(rota_to_cluster)

    # Afinidade aprendida
    def get_affinity(cluster):
        return cluster_affinity.get(cluster, None)
    df_comb['Transportadora_afinidade'] = df_comb['cluster'].apply(get_affinity)

    # Cidades livres para atuação geral
    cidades_livres = {'ARACAJU', 'NOSSA SENHORA DO SOCORRO'}
    df_comb['Cidade_norm'] = df_comb['Cidade'].str.strip().str.upper()

    total_rotas = df_comb['Rota'].nunique()
    cotas = {t: 0 for t in percentuais}
    rotas_atribuidas = {}
    rotas_unicas = df_comb[['Rota', 'Cidade_norm', 'cluster', 'Transportadora_afinidade']].drop_duplicates()

    for _, row in rotas_unicas.iterrows():
        rota = row['Rota']
        cidade = row['Cidade_norm']
        affin = row['Transportadora_afinidade']
        if cidade in cidades_livres:
            difs = {t: (cotas[t] / total_rotas) - percentuais[t] for t in percentuais}
            transp = min(difs, key=lambda k: difs[k])
        elif affin in cotas and cotas[affin] / total_rotas < percentuais[affin]:
            transp = affin
        else:
            difs = {t: (cotas[t] / total_rotas) - percentuais[t] for t in percentuais}
            transp = min(difs, key=lambda k: difs[k])
        rotas_atribuidas[rota] = transp
        cotas[transp] += 1

    df_comb['Transportadora_Sugerida'] = df_comb['Rota'].map(rotas_atribuidas)

    st.success("Atribuição concluída com sucesso!")
    preview_cols = ["Rota", "Cidade", "Transportadora_afinidade", "Transportadora_Sugerida"]
    st.dataframe(df_comb[preview_cols].drop_duplicates().head(20))

    st.download_button(
        "Baixar atribuição CSV",
        df_comb[preview_cols].to_csv(index=False),
        "rotas_atribuidas.csv"
    )
