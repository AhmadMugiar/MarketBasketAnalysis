import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime as dt
import networkx as nx # Import networkx for graph visualization
import matplotlib.pyplot as plt # Import matplotlib for graph visualization
# Import functions from data.py
from data import (
    load_and_preprocess_data, 
    perform_rfm_clustering, # Renamed clustering to perform_rfm_clustering
    calculate_elbow_silhouette, # Added for K-Means optimization
    perform_mba, # Renamed Market to perform_mba
    get_gap_rules, # Added for GAP rules
    recommend_products_based_on_gap # Added for recommendations
)

# Page configuration
st.set_page_config(
    page_title="Kalbe Analytics Dashboard",
    page_icon=":basket:",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for modern styling (your CSS remains the same)
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
        padding: 1rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
        margin-bottom: 1rem;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    .tab-container {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 4px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        border-radius: 8px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: transparent;
        border: none;
        color: #555;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .cluster-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    
    .insight-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state for data loading
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame()
if 'rfm_data' not in st.session_state:
    st.session_state.rfm_data = pd.DataFrame()
if 'cluster_summary' not in st.session_state:
    st.session_state.cluster_summary = pd.DataFrame()

# Header
st.markdown('<div class="main-header">DASHBOARD REKOMENDASI PRODUK BERBASIS MARKET BASKET ANALYSIS DAN CLUSTERING K-MEANS</div>', unsafe_allow_html=True)

# Data Loading Section
if not st.session_state.data_loaded:
    with st.container():
        st.markdown("### 🔄 Loading Data...")
        
        # Load and preprocess data using st.cache_data for efficiency
        @st.cache_data
        def cached_load_and_process():
            df_loaded = load_and_preprocess_data("df_cleaning3.csv") 
            return df_loaded

        df_temp = cached_load_and_process()

        if not df_temp.empty:
            st.session_state.df = df_temp
            rfm_temp, _, _ = perform_rfm_clustering(st.session_state.df.copy(), n_clusters=3)
            if not rfm_temp.empty:
                st.session_state.rfm_data = rfm_temp
                st.session_state.cluster_summary = rfm_temp.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean().astype(int)
                st.session_state.data_loaded = True
                st.success("✅ Data loaded and customers clustered successfully!")
                st.rerun()
            else:
                st.error("❌ Failed to perform RFM clustering.")
                st.stop()
        else:
            st.error("❌ Failed to load data. Please check if 'df_cleaning.csv' exists and is correctly formatted.")
            st.stop()

if st.session_state.data_loaded:
    df = st.session_state.df
    rfm_data = st.session_state.rfm_data
    cluster_summary = st.session_state.cluster_summary
    
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "👥 Customer Segmentation", "🛒 Market Basket Analysis", "💡 GAP Analysis & Recommendations"])
    
    with tab1:  
        st.markdown("### 📊 Business Overview")
        
        # Key Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_customers = df['CustomerID'].nunique()
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{total_customers:,}</div>
                    <div class="metric-label">Total Customers</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            total_transactions = df['TransactionID'].nunique()
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{total_transactions}</div>
                    <div class="metric-label">Total Transactions</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            total_revenue = df['TotalPrice'].sum()
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">Rp {total_revenue:,.0f}</div>
                    <div class="metric-label">Total Revenue</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col4:
            avg_order_value = df.groupby('TransactionID')['TotalPrice'].sum().mean()
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">Rp {avg_order_value:,.0f}</div>
                    <div class="metric-label">Avg Order Value</div>
                </div>
            """, unsafe_allow_html=True)
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Sales Trend
            daily_sales = df.groupby(df['Date'].dt.date)['TotalPrice'].sum().reset_index()
            fig_sales = px.line(daily_sales, x='Date', y='TotalPrice', 
                                title='📈 Tren Penjualan Harian',
                                color_discrete_sequence=['#667eea'])
            fig_sales.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#333')
            )
            st.plotly_chart(fig_sales, use_container_width=True)
        with col2:
            if 'Product' in df.columns and 'Quantity' in df.columns:
                top_products = df.groupby('Product')['Quantity'].sum().nlargest(10).sort_values(ascending=True)
                fig_products = px.bar(x=top_products.values, y=top_products.index, 
                                    orientation='h',
                                    title='🏆 Top 10 Penjualan Produk',
                                    color_discrete_sequence=['#764ba2'])
                fig_products.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#333')
                )
                st.plotly_chart(fig_products, use_container_width=True)
            else:
                st.warning("Kolom 'Product' atau 'Quantity' tidak ditemukan untuk menampilkan Top Products.")
        
with tab2:  # Customer Segmentation Tab
    st.markdown("### 👥 Segmentasi Pelanggan dengan K-Means Clustering")
    st.markdown("### 💡Deskripsi Pelanggan")
    
    col_segmen1,col_segmen2,col_segmen3 = st.columns(3)
    with col_segmen1:
        
        st.markdown("#### 🎯Segmen Pelanggan Cluster 0" )
        st.markdown("""
            <div class="insight-box">
                    Cluster 0 adalah Kelompok pelanggan dengan potensi pertumbuhan tinggi berdasarkan frekuensi dan nilai transaksi. Mereka menunjukkan minat kuat terhadap produk dan berpotensi menjadi pelanggan setia. Dengan strategi yang tepat, segmen ini dapat ditingkatkan menjadi pelanggan utama yang loyal dan menguntungkan.
                <ul>
                    <li><strong>Recency :</strong> 58 Hari</li>
                    <li><strong>Frequency :</strong> 3 </li>
                    <li><strong>Monetary :</strong> Rp 256.300,00</li>
                    <li><strong>Kategori  :</strong> Grow Potential Customers</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    with col_segmen2:            
            st.markdown("#### 🎯Segmen Pelanggan Cluster 1" )
            st.markdown("""
                <div class="insight-box">
                        Cluster 1 adalah Kelompok pelanggan setia yang sering bertransaksi dan memiliki nilai pembelian tinggi. Segmen ini menunjukkan kepercayaan kuat terhadap brand dan menjadi fondasi penting bagi keberlangsungan bisnis. Memberikan penghargaan dan pelayanan eksklusif sangat efektif untuk mempertahankan dan meningkatkan loyalitas mereka.
                    <ul>
                        <li><strong>Recency :</strong> 55 hari</li>
                        <li><strong>Frequency :</strong> 5 </li>
                        <li><strong>Monetary :</strong> Rp 579.632,00</li>
                        <li><strong>Kategori  :</strong> Loyal Customers</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
    with col_segmen3:
            st.markdown("#### 🎯Segmen Pelanggan Cluster 2" )
            st.markdown("""
                <div class="insight-box">
                    Cluster 2 adalah Kelompok pelanggan yang sudah lama tidak melakukan pembelian atau menunjukkan penurunan aktivitas secara signifikan. Segmen ini berisiko tinggi meninggalkan brand dan memerlukan pendekatan khusus, seperti penawaran personal, reminder, atau kampanye retensi, untuk mengembalikan minat dan keterlibatan mereka.
                    <ul>
                        <li><strong>Recency :</strong> 213 hari</li>
                        <li><strong>Frequency :</strong> 1 </li>
                        <li><strong>Monetary :</strong> Rp 187.667,00</li>
                        <li><strong>Kategori  :</strong> Churn Customers</li>
                    </ul>
                </div>""", unsafe_allow_html=True)   

    # Option to choose K for KMeans or use Elbow/Silhouette
    st.sidebar.subheader("Pengaturan Clustering")
    run_elbow_silhouette = st.sidebar.checkbox("Tampilkan Elbow & Silhouette Plots", value=False)

    # Recalculate RFM for optimal K selection (no need to store in session state yet)
    rfm_data_for_optimal_k = df.groupby('CustomerID').agg(
        Recency=('Date', lambda date: (dt.datetime(2023, 1, 1) - date.max()).days),
        Frequency=('TransactionID', 'nunique'),
        Monetary=('TotalPrice', 'sum')
    ).reset_index()
        # Cluster Distribution
    cluster_counts = rfm_data['Cluster'].value_counts().sort_index()

    col1, col2 = st.columns([1, 2])

    with col1:
        fig_pie = px.pie(values=cluster_counts.values, 
                         names=[f'Cluster {i}' for i in cluster_counts.index],
                         title='Distribusi Segmentasi Pelanggan ',
                         color_discrete_sequence=['#667eea', '#764ba2', '#f093fb', '#ADD8E6', '#90EE90', '#FFD700', '#FF6347'])
        fig_pie.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        fig_rfm = go.Figure()
        clusters = cluster_summary.index
        colors_list = ['#667eea', '#764ba2', '#f093fb', '#ADD8E6', '#90EE90', '#FFD700', '#FF6347']

        fig_rfm = px.scatter_3d(
                rfm_data,
                x='Recency',
                y='Frequency',
                z='Monetary',
                color='Cluster',
                title='Visualisasi 3D Cluster yang Diperoleh dari K-Means',
                labels={'Recency': 'Recency', 'Frequency': 'Frequency', 'Monetary': 'Monetary'},
                opacity=0.7
            )

            # Berikan key unik di sini
        st.plotly_chart(fig_rfm, use_container_width=True, key="scatter3d_rfm")
    if run_elbow_silhouette:
        st.subheader("Optimalisasi Jumlah Cluster (K)")
        st.info("Gunakan plot di bawah ini untuk membantu menentukan jumlah cluster (K) yang optimal. Pilih K di sidebar.")

        elbow_silhouette_results = calculate_elbow_silhouette(rfm_data_for_optimal_k)

        col_elbow, col_silhouette = st.columns(2)

        with col_elbow:
            st.subheader("Elbow Method")
            fig_elbow = px.line(
                x=elbow_silhouette_results['k_range'],
                y=elbow_silhouette_results['wcss'], 
                markers=True, 
                title='Elbow Method for Optimal K',
                labels={'x': 'Number of Clusters (K)', 'y': 'WCSS'}
            )
            st.plotly_chart(fig_elbow, use_container_width=True)

        with col_silhouette:
            if len(elbow_silhouette_results['silhouette_scores']) > 0:
                st.subheader("Silhouette Score")
                fig_silhouette = px.line(
                    x=elbow_silhouette_results['k_range'],
                    y=elbow_silhouette_results['silhouette_scores'],
                    markers=True, 
                    title='Silhouette Score for Optimal K',
                    labels={'x': 'Number of Clusters (K)', 'y': 'Silhouette Score'}
                )
                st.plotly_chart(fig_silhouette, use_container_width=True)
            else:
                st.info("Silhouette score membutuhkan minimal 2 cluster.")

    selected_n_clusters = st.sidebar.slider("Pilih Jumlah Cluster (K) untuk RFM", min_value=2, max_value=7, value=st.session_state.rfm_data['Cluster'].nunique())

    # Re-run clustering if K changes
    if selected_n_clusters != st.session_state.rfm_data['Cluster'].nunique():
        st.session_state.rfm_data, _, _ = perform_rfm_clustering(st.session_state.df.copy(), n_clusters=selected_n_clusters)
        st.session_state.cluster_summary = st.session_state.rfm_data.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean().astype(int)
        st.rerun()  # Rerun to apply new clustering and update all charts

    rfm_data = st.session_state.rfm_data
    cluster_summary = st.session_state.cluster_summary

    st.info(f"Pelanggan berhasil dikelompokkan menjadi **{selected_n_clusters}** cluster.")
    st.subheader("Data Pelanggan dengan Cluster:")
    st.dataframe(rfm_data)

    # Display cluster characteristics
    st.subheader("Karakteristik Rata-rata Setiap Cluster")
    st.dataframe(cluster_summary)
    
    with tab3:  # Market Basket Analysis Tab
        st.markdown("### 🛒 Market Basket Analysis")
        
        # Segment Selection for MBA
        selected_mba_cluster = st.selectbox(
            "Pilih Customer Segment untuk Analisis:",
            options=sorted(rfm_data['Cluster'].unique()),
            format_func=lambda x: f"Cluster {x} (Avg. R:{cluster_summary.loc[x]['Recency']}, F:{cluster_summary.loc[x]['Frequency']}, M:{cluster_summary.loc[x]['Monetary']})" if x in cluster_summary.index else f"Cluster {x}"
        )
        
        st.sidebar.subheader("Pengaturan Market Basket Analysis")
        mba_min_support = st.sidebar.slider("Minimum Support", min_value=0.005, max_value=0.1, value=0.01, step=0.001, key='mba_support')
        mba_min_confidence = st.sidebar.slider("Minimum Confidence", min_value=0.01, max_value=1.0, value=0.2, step=0.01, key='mba_confidence')
        mba_min_lift = st.sidebar.slider("Minimum Lift", min_value=0.1, max_value=5.0, value=1.0, step=0.1, key='mba_lift')

        try:
            with st.spinner(f"Analyzing market basket patterns for Cluster {selected_mba_cluster}..."):
                # Call perform_mba with selected cluster and parameters
                mba_results = perform_mba(df.copy(), rfm_data.copy(), selected_mba_cluster, mba_min_support, mba_min_confidence, mba_min_lift)
                
            if mba_results['rules'].empty and mba_results['frequent_itemsets'].empty:
                st.info(f"Tidak ada hasil MBA untuk Cluster {selected_mba_cluster} dengan parameter yang diberikan. {mba_results['message']}")
            else:
                st.success(f"✅ Analysis completed for Cluster {selected_mba_cluster}")
                
                st.markdown("#### 🔝 Kumpulan item yang sering dibeli bersamaan")
                st.dataframe(mba_results['frequent_itemsets'])

                st.markdown("#### 🔗 Association Rules")
                st.dataframe(mba_results['rules'])
                st.info("Interpretasi Rule (A -> B): Jika pelanggan membeli A, kemungkinan mereka juga akan membeli B.")
                st.markdown(f"- **Support**: Seberapa sering item A dan B muncul bersama dalam semua transaksi ")
                st.markdown(f"- **Confidence**: Probabilitas B dibeli ketika A telah dibeli.")
                st.markdown(f"- **Lift**: Seberapa besar kemungkinan B dibeli ketika A telah dibeli, dibandingkan dengan B dibeli secara independen. (>1 menunjukkan asosiasi positif)")
                # Visualize Network Graph of rules for the selected cluster
                if not mba_results['rules'].empty:
                    st.markdown("#### 🌐 Jaringan Aturan Asosiasi (Top Rules)")
                    
                    G_mba = nx.DiGraph()
                    for _, rule in mba_results['rules'].iterrows():
                        antecedents = rule['antecedents'].split(', ')
                        consequents = rule['consequents'].split(', ')
                        for ant in antecedents:
                            for cons in consequents:
                                G_mba.add_edge(ant, cons, weight=rule['lift'])

                    if G_mba.number_of_edges() > 0:
                        fig_mba_graph, ax = plt.subplots(figsize=(15, 10))
                        pos = nx.spring_layout(G_mba, k=0.15, iterations=50, seed=42) 
                        
                        nx.draw_networkx_nodes(G_mba, pos, node_size=5000, node_color="skyblue", alpha=0.7, ax=ax)
                        nx.draw_networkx_edges(G_mba, pos, width=2, alpha=0.5, edge_color='gray', ax=ax)
                        nx.draw_networkx_labels(G_mba, pos, font_size=10, font_weight="bold", font_color="black", ax=ax)

                        edge_labels = {(u, v): f"{G_mba[u][v]['weight']:.2f}" for u, v in G_mba.edges()}
                        nx.draw_networkx_edge_labels(G_mba, pos, edge_labels=edge_labels, font_size=9, font_color='darkgreen', ax=ax)

                        ax.set_title(f'Jaringan Aturan Asosiasi (Top Rules) Cluster {selected_mba_cluster}', fontsize=12)
                        ax.axis('off')
                        st.pyplot(fig_mba_graph)
                    else:
                        st.info("Tidak ada edge untuk divisualisasikan dari aturan asosiasi yang ditemukan.")
        
        except Exception as e:
            st.error(f"Error in market basket analysis: {str(e)}")


with tab4:  # Insights & Recommendations Tab
    st.markdown("### 💡 Analisis GAP Rules")
    st.markdown("#### 🔗 GAP Rules Analysis & Product Recommendation")
    st.write("Identifikasi celah atau peluang promosi dengan membandingkan pola pembelian antar segmen.")
    st.info("""
    **Apa itu GAP Rule?**
    GAP Rule adalah aturan asosiasi yang kuat di **Cluster Sumber (Source)**, tetapi tidak atau jarang ditemukan di **Cluster Target**.
    Ini menunjukkan peluang untuk menerapkan strategi promosi/rekomendasi dari Cluster Sumber ke Cluster Target.
    """)

    col_gap1, col_gap2 = st.columns(2)

    with col_gap1:
        source_cluster_id = st.selectbox(
            "Pilih Cluster Sumber (Source Cluster) - Aturan Asosiasi akan Diambil dari Sini:",
            options=sorted(rfm_data['Cluster'].unique()),
            key='gap_source_cluster_id',
            format_func=lambda x: f"Cluster {x} (Avg. R:{cluster_summary.loc[x]['Recency']}, F:{cluster_summary.loc[x]['Frequency']}, M:{cluster_summary.loc[x]['Monetary']})" if x in cluster_summary.index else f"Cluster {x}"
        )

    with col_gap2:
        target_cluster_id = st.selectbox(
            "Pilih Cluster Target (Target Cluster) - Celah akan Ditemukan di Sini:",
            options=sorted(rfm_data['Cluster'].unique()),
            key='gap_target_cluster_id',
            format_func=lambda x: f"Cluster {x} (Avg. R:{cluster_summary.loc[x]['Recency']}, F:{cluster_summary.loc[x]['Frequency']}, M:{cluster_summary.loc[x]['Monetary']})" if x in cluster_summary.index else f"Cluster {x}"
        )

    if source_cluster_id == target_cluster_id:
        st.warning("Cluster Sumber dan Cluster Target tidak boleh sama untuk analisis GAP Rules.")
    else:
        with st.spinner(f"Menganalisis GAP Rules antara Cluster {source_cluster_id} dan {target_cluster_id}..."):
            # Get all rules for Source Cluster (min_support, min_confidence, min_lift from sidebar MBA settings)
            mba_source_results = perform_mba(df.copy(), rfm_data.copy(), source_cluster_id, mba_min_support, mba_min_confidence, mba_min_lift)
            rules_source = mba_source_results['all_rules'] # This is the key that holds ALL filtered rules

            # Get all rules for Target Cluster
            mba_target_results = perform_mba(df.copy(), rfm_data.copy(), target_cluster_id, mba_min_support, mba_min_confidence, mba_min_lift)
            rules_target = mba_target_results['all_rules']

            if rules_source.empty:
                st.warning(f"Tidak ada aturan asosiasi yang kuat ditemukan di Cluster Sumber {source_cluster_id}. Tidak dapat mencari GAP.")
            else:
                # Call get_gap_rules from data.py
                gap_rules_df = get_gap_rules(rules_source, rules_target)

                if not gap_rules_df.empty:
                    st.subheader(f"GAP Rules Ditemukan (Ada di Cluster {source_cluster_id}, Tidak Ada/Jarang di Cluster {target_cluster_id})")
                    st.dataframe(gap_rules_df[['antecedents', 'consequents', 'support', 'confidence', 'lift', 'rule', 'Description']])
                    st.info("Gunakan aturan ini untuk merekomendasikan produk di Cluster Target.")
                    
                    st.subheader(f"Jaringan GAP Rules (Dari Cluster {source_cluster_id} ke Cluster {target_cluster_id})")
                    G_gap = nx.DiGraph()
                    for _, row in gap_rules_df.iterrows():
                        # antecedents and consequents are already lists due to get_gap_rules's output
                        antecedents = row['antecedents'] 
                        consequents = row['consequents']
                        confidence = row['confidence'] # Use 'confidence' directly from gap_rules_df

                        for a in antecedents:
                            for c in consequents:
                                G_gap.add_edge(a, c, weight=confidence)

                    if G_gap.number_of_edges() > 0:
                        fig_gap_graph, ax = plt.subplots(figsize=(15, 10))
                        pos = nx.spring_layout(G_gap, k=0.15, iterations=50, seed=42)  
                        
                        nx.draw_networkx_nodes(G_gap, pos, node_size=3000, node_color="lightcoral", alpha=0.7, ax=ax)
                        nx.draw_networkx_edges(G_gap, pos, width=2, alpha=0.5, edge_color='darkred', ax=ax)
                        nx.draw_networkx_labels(G_gap, pos, font_size=10, font_weight="bold", font_color="black", ax=ax)
                        
                        edge_labels = {(u, v): f"{G_gap[u][v]['weight']:.2f}" for u, v in G_gap.edges()}
                        nx.draw_networkx_edge_labels(G_gap, pos, edge_labels=edge_labels, font_size=9, font_color='red', ax=ax)
                        
                        ax.set_title(f'Jaringan GAP Rules (Dari Cluster {source_cluster_id} ke Cluster {target_cluster_id})', fontsize=16)
                        ax.axis('off')
                        st.pyplot(fig_gap_graph)
                    else:
                        st.info("Tidak ada edge untuk divisualisasikan dari GAP Rules yang ditemukan.")
                else:
                    st.info(f"Tidak ada GAP Rules yang ditemukan antara Cluster {source_cluster_id} dan Cluster {target_cluster_id} dengan parameter yang diberikan.")

            # --- Product Recommendation based on GAP Rules ---
            st.subheader("Rekomendasi Produk Berdasarkan GAP Rules")
            st.write("Masukkan produk yang baru saja dibeli pelanggan dari Cluster Target untuk mendapatkan rekomendasi produk yang 'seharusnya' mereka beli.")

            all_products = sorted(df['Product'].unique())  

            # Allow multi-selection of products
            customer_recent_items = st.multiselect(
                "Produk dibeli pelanggan (dari Cluster Target ini):",
                options=all_products,
                help="Pilih produk yang baru saja dibeli oleh seorang pelanggan dari cluster target."
            )

            # Ensure gap_rules_df is defined and not empty before trying to use it
            if customer_recent_items and 'gap_rules_df' in locals() and not gap_rules_df.empty:
                # Call recommend_products_based_on_gap from data.py
                recommendations = recommend_products_based_on_gap(customer_recent_items, gap_rules_df)
                if recommendations:
                    st.success(f"**Rekomendasi Produk untuk Pelanggan Ini:**")
                    for item in recommendations:
                        st.write(f"- {item}")
                    st.info("Rekomendasi ini didasarkan pada pola pembelian yang kuat di Cluster Sumber, yang tidak terlihat di Cluster Target.")
                else:
                    st.info("Tidak ada rekomendasi yang ditemukan berdasarkan produk yang dipilih dan GAP Rules yang ada.")
            elif not customer_recent_items:
                st.info("Silakan pilih produk yang baru saja dibeli pelanggan untuk mendapatkan rekomendasi.")
            else:
                st.info("Tidak ada GAP Rules yang tersedia untuk membuat rekomendasi. Pastikan Anda telah menemukan GAP Rules di atas.")
        
        st.markdown("---")
        st.markdown("#### 📊 Performance & Strategy Summary")
        
        # Create a summary table
        summary_data = []
        for cluster in cluster_summary.index: # Use actual clusters found
            cluster_data = rfm_data[rfm_data['Cluster'] == cluster]
            summary_data.append({
                'Cluster': f'Cluster {cluster}',
                'Customer Count': len(cluster_data),
                'Avg Recency (days)': cluster_data['Recency'].mean(),
                'Avg Frequency': cluster_data['Frequency'].mean(),
                'Avg Monetary (Rp)': cluster_data['Monetary'].mean(),
                'Revenue Share (%)': (cluster_data['Monetary'].sum() / rfm_data['Monetary'].sum()) * 100
            })
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 1rem;'>
        📊 PROJECT INTEGRASI 1 | DATA SCIENCE 2023 | 
        <a href='#' style='color: #667eea; text-decoration: none;'>Ahmad Mugiar Sujana & Melvin Ariwati Hanek</a>
    </div>
    """, 
    unsafe_allow_html=True
)