import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from collections import defaultdict

st.set_page_config(page_title="Atribuição de Rotas por Geolocalização", layout="wide")

st.title("Atribuição de Rotas Geolocalizada para Transportadoras")

st.markdown("""
- Faça upload dos arquivos necessários.
- Insira o percentual de rotas a ser atribuído para cada transportadora (0.0 a 1.0).
""")

# Upload dos arquivos
col1, col2, col3 = st.columns(3)
with col1:
    rotas_file = st.file_uploader("Upload do arquivo de rotas ('combined')", type='csv', key="rotas")
with col2:
    plan_file = st.file_uploader("Upload da base de ids ('planification')", type='csv', key="shipments")
with col3:
    hist_file = st.file_uploader("Upload planejamento histórico ('planejamento-original')", type='csv', key="history")


# Obtenção de percentuais manualmente
with st.expander("Inserir percentuais de cada transportadora"):
    transportadoras = st.text_input("Nomes separados por vírgula (ex: AC2 Logistica,Log Serviços,MSR,BRJTransportes,WLS Cargo)", value="AC2 Logistica,Log Serviços,MSR,BRJTransportes,WLS Cargo")
    transportadoras = [x.strip() for x in transportadoras.split(",") if x.strip()]
    percentuais = {}
    for t in transportadoras:
        percentuais[t] = st.number_input(f"Percentual de {t} (Ex: 0.3 para 30%)", min_value=0.0, max_value=1.0, value=0.0, step=0.01, format="%.3f")

if st.button("Executar atribuição automática"):

    # ---------- 1. Carregar arquivos ----------
    try:
        df_rotas = pd.read_csv(rotas_file)
        df_plan = pd.read_csv(plan_file)
        df_hist = pd.read_csv(hist_file)
    except Exception as e:
        st.error(f"Erro ao carregar arquivos: {e}")
        st.stop()

    # ---------- 2. Processar histórico: extrair afinidade com base nos clusters ----------
    # Obter coordenadas e transportadora do histórico
    # Assumindo que df_hist tem colunas ['ID Planejado', 'Transportadora']
    # E que consegue mapear 'ID Planejado' para shipment/ID na base
    # É preciso no mundo real garantir essa correspondência
    
    # Passo 1: Construir base consolidada: ID + Transportadora + latitude + longitude
    # Usar df_plan para vincular ID Planejado ao endereço e coordenadas
    # Supondo 'ID de Unidad' ou 'Shipment' seja o campo em comum (ajuste conforme sua base):

    # Ajustar nomes se necessário:
    col_id_hist = 'ID Planejado'
    col_id_plan = 'ID de Unidad' if 'ID de Unidad' in df_plan.columns else 'Shipment'

    # Merge: Colocar Transportadora na base de histórico com coordenadas
    df_hist = df_hist.dropna(subset=[col_id_hist, 'Transportadora'])
    df_hist_plan = df_hist.merge(
        df_plan[[col_id_plan, 'Latitude', 'Longitude']],
        left_on=col_id_hist, right_on=col_id_plan, how='left'
    )
    df_hist_plan = df_hist_plan.dropna(subset=['Latitude', 'Longitude'])

    # Passo 2: Clusterização dos pontos históricos (DBSCAN)
    coords = df_hist_plan[['Latitude', 'Longitude']].to_numpy()
    clustering = DBSCAN(eps=0.06, min_samples=4).fit(coords)  # eps ajusta o raio do cluster (~6km se coordenadas em grados)

    df_hist_plan['cluster'] = clustering.labels_
    # Cada cluster recebe a 'dominante' Transportadora: afinidade geográfica histórica!
    cluster_affinity = df_hist_plan.groupby('cluster')['Transportadora'].agg(lambda x: x.value_counts().idxmax())
    cluster_affinity = cluster_affinity.to_dict()

    # ---------- 3. Agrupar novas rotas por região ----------
    # Mapeie um ID por rota (pode ser a média dos pontos da rota)
    rota_coords = df_rotas.groupby('Rota')[['Latitude', 'Longitude']].mean()
    rota_coords = rota_coords.dropna()

    # Clusterizar essas rotas com os clusters históricos
    rotas_clusters = DBSCAN(eps=0.06, min_samples=1).fit(rota_coords)
    rota_coords['cluster'] = rotas_clusters.labels_
    # Atribua cluster ao df_rotas
    df_rotas = df_rotas.merge(rota_coords[['cluster']], left_on='Rota', right_index=True, how='left')

    # ---------- 4. Afinidade histórica por cluster ----------
    def get_affinity(cluster):
        try:
            return cluster_affinity[cluster]
        except:
            return None

    df_rotas['Transportadora_afinidade'] = df_rotas['cluster'].apply(get_affinity)

    # Cidades livres para atuação geral
    cidades_livres = {'ARACAJU', 'NOSSA SENHORA DO SOCORRO'}
    # Padronize o campo Cidade
    df_rotas['Cidade_norm'] = df_rotas['Cidade'].str.strip().str.upper()

    # ---------- 5. Atribuição respeitando percentual ----------
    # Fazer por ROTA (não por shipment)
    total_rotas = df_rotas['Rota'].nunique()
    cotas = {t: 0 for t in percentuais}
    rotas_atribuidas = {}
    rotas_unicas = df_rotas[['Rota', 'Cidade_norm', 'cluster', 'Transportadora_afinidade']].drop_duplicates()

    for idx, row in rotas_unicas.iterrows():
        rota = row['Rota']
        cid = row['Cidade_norm']
        affin = row['Transportadora_afinidade']
        # Se cidade livre, escolha quem estiver mais longe da cota
        if cid in cidades_livres:
            difs = {t: (cotas[t] / total_rotas) - percentuais[t] for t in percentuais}
            transp = min(difs, key=lambda k: difs[k])
        elif affin in cotas and cotas[affin]/total_rotas < percentuais[affin]:
            transp = affin
        else:
            # Alocar ao mais defasado
            difs = {t: (cotas[t] / total_rotas) - percentuais[t] for t in percentuais}
            transp = min(difs, key=lambda k: difs[k])
        rotas_atribuidas[rota] = transp
        cotas[transp] += 1

    # Aplique ao DataFrame final
    df_rotas['Transportadora_Sugerida'] = df_rotas['Rota'].map(rotas_atribuidas)
    st.success("Atribuição realizada com sucesso!")

    # Exiba resultado
    st.write(df_rotas[['Rota', 'Cidade', 'Transportadora_Sugerida']].drop_duplicates().head(100))

    st.download_button("Baixar atribuição CSV", df_rotas.to_csv(index=False), "rodas_atribuidas.csv")