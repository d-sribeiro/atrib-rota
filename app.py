import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from collections import defaultdict

st.set_page_config(page_title="Atribuição de Rotas por Geolocalização", layout="wide")
st.title("Atribuição de Rotas por Geolocalização para Transportadoras")

st.markdown("""
Faça upload dos dois arquivos necessários e insira os percentuais das transportadoras.
O sistema usará automaticamente o arquivo `treino.csv` como base histórica de aprendizado de afinidades regionais.
""")

# UPLOAD DOS 2 ARQUIVOS
col1, col2 = st.columns(2)
with col1:
    plan_file = st.file_uploader("Base de IDs (planification)", type="csv")
with col2:
    rotas_file = st.file_uploader("Arquivo de Rotas (combined)", type="csv")

# INSERÇÃO DE PERCENTUAIS
with st.expander("Inserir percentuais de cada transportadora"):
    transportadoras = st.text_input(
        "Nomes separados por vírgula (ex: AC2 Logistica,Log Serviços,MSR,BRJTransportes,WLS Cargo)",
        value="AC2 Logistica,Log Serviços,MSR,BRJTransportes,WLS Cargo"
    )
    transportadoras = [x.strip() for x in transportadoras.split(",") if x.strip()]
    percentuais = {}
    for t in transportadoras:
        percentuais[t] = st.number_input(
            f"Percentual de {t} (Ex: 0.3 para 30%)",
            min_value=0.0, max_value=1.0, step=0.01, format="%.3f", value=0.0)

if st.button("Executar atribuição automática"):
    # ---- 1. LEITURA DOS ARQUIVOS ----
    try:
        df_plan = pd.read_csv(plan_file, dtype=str)
        df_rotas = pd.read_csv(rotas_file, dtype=str)
        df_hist = pd.read_csv("treino.csv", dtype=str)
        # Garante números/coords em float para clustering
        df_hist['Latitude'] = df_hist['Latitude'].astype(float)
        df_hist['Longitude'] = df_hist['Longitude'].astype(float)
    except Exception as e:
        st.error(f"Erro ao carregar arquivos: {e}")
        st.stop()

    # ---- 2. CRIAÇÃO DOS CLUSTERS GEOGRÁFICOS HISTÓRICOS ----
    coords_hist = df_hist[['Latitude', 'Longitude']].dropna().values
    db = DBSCAN(eps=0.06, min_samples=4).fit(coords_hist)  # ~6km
    df_hist['cluster'] = db.labels_

    # Afinidade de cada cluster: transportadora mais frequente naquela região
    cluster_affinity = (
        df_hist.groupby('cluster')['Transportadora']
        .agg(lambda x: x.value_counts().idxmax())
        .to_dict()
    )
    # Clusters = -1 são pontos outliers sem cluster, não usamos clustered affinity

    # ---- 3. PREPARAR DADOS DAS ROTAS NOVAS ----
    # Planificação prepara dados de shipment:ID~coords. Combined associa shipment-rotas.
    # Precisamos: shipment~coords, shipment~rota, shipment~outros_dados

    # padroniza nomes de colunas
    if 'ID de Unidad' in df_plan.columns:
        df_plan = df_plan.rename(columns={'ID de Unidad': 'Shipment'})
    plan_cols = ['Shipment', 'Rua', 'Numero', 'Bairro', 'Cidade', 'Estado', 'CEP', 'Latitude', 'Longitude']
    df_plan = df_plan[plan_cols].copy()
    df_plan['Latitude'] = df_plan['Latitude'].astype(float)
    df_plan['Longitude'] = df_plan['Longitude'].astype(float)

    # Em rotas: 'Shipment', 'Rota' (nome pode ser diferente, ajuste se necessário)
    if 'Rota' not in df_rotas.columns:
        st.error("Arquivo de rotas não possui coluna 'Rota'.")
        st.stop()
    if 'Shipment' not in df_rotas.columns:
        st.error("Arquivo de rotas não possui coluna 'Shipment'.")
        st.stop()

    rotas_merged = df_rotas.merge(df_plan, on='Shipment', how='left')

    # Por rota, pegar a média dos pontos de latitude/longitude dos Shipments (centroide aproximado daquele conjunto)
    centros_rota = rotas_merged.groupby('Rota')[['Latitude', 'Longitude']].mean().dropna()

    # ---- 4. CLUSTERIZAR AS ROTAS NOVAS USANDO MESMA LÓGICA ----
     # Calcula clusters dos centroides
    db_r = DBSCAN(eps=0.06, min_samples=1).fit(centros_rota.values)
    centros_rota['cluster'] = db_r.labels_

    # Cria dict: rota -> cluster
    rota_to_cluster = centros_rota['cluster'].to_dict()

    # Espalha nos dados de shipments
    rotas_merged['cluster'] = rotas_merged['Rota'].map(rota_to_cluster)

    # cluster -> transportadora (afinidade histórica)
    # cluster_affinity: dict {cluster_number: "Transportadora"}
    rotas_merged['Transportadora_afinidade'] = rotas_merged['cluster'].map(cluster_affinity)

    # ---- 6. ALOCAÇÃO DAS ROTAS NOVAS ----
    cidades_livres = {'ARACAJU', 'NOSSA SENHORA DO SOCORRO'}
    rotas_merged['Cidade_norm'] = rotas_merged['Cidade'].str.strip().str.upper()
    total_rotas = rotas_merged['Rota'].nunique()
    cotas = {t: 0 for t in percentuais}
    rotas_atribuidas = {}

    rotas_unicas = rotas_merged[['Rota', 'Cidade_norm', 'cluster', 'Transportadora_afinidade']].drop_duplicates()

    for _, row in rotas_unicas.iterrows():
        rota = row['Rota']
        cidade = row['Cidade_norm']
        affin = row['Transportadora_afinidade']
        if cidade in cidades_livres:
            difs = {t: (cotas[t] / total_rotas) - percentuais[t] for t in percentuais}
            transp = min(difs, key=lambda k: difs[k])
        elif affin in cotas and cotas[affin]/total_rotas < percentuais[affin]:
            transp = affin
        else:
            difs = {t: (cotas[t] / total_rotas) - percentuais[t] for t in percentuais}
            transp = min(difs, key=lambda k: difs[k])
        rotas_atribuidas[rota] = transp
        cotas[transp] += 1

    rotas_merged['Transportadora_Sugerida'] = rotas_merged['Rota'].map(rotas_atribuidas)

    st.success("Atribuição realizada com sucesso!")
    preview_cols = ["Rota", "Shipment", "Cidade", "Bairro", "Rua", "Numero",
                    "Latitude", "Longitude", "Transportadora_Sugerida"]
    st.dataframe(rotas_merged[preview_cols].drop_duplicates().head(100))

    st.download_button(
        "Baixar atribuição CSV",
        rotas_merged.to_csv(index=False),
        "rotas_atribuidas.csv"
    )
