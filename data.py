import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import association_rules
from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
import os
from sklearn.metrics import silhouette_score
import datetime as dt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import streamlit as st
warnings.filterwarnings('ignore')

# --- CACHED DATA LOADING AND PREPROCESSING ---
def load_and_preprocess_data(file_path="df_cleaning3.csv"):
    if not os.path.exists(file_path):
        st.error(f"File '{file_path}' tidak ditemukan. Pastikan file berada di direktori yang sama.")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df.dropna(subset=['Date'], inplace=True)

        # Calculate TotalPrice
        df['TotalPrice'] = df['Quantity'] * df['Price']
        df['Product'].astype(str) 
        
        return df
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        return pd.DataFrame()

# --- RFM SEGMENTATION AND CLUSTERING ---
def perform_rfm_clustering(df, n_clusters=3):
    try:
        # Using a fixed date for recency calculation, adjust if needed
        # Or you can use df['Date'].max() + timedelta(days=1) for dynamic latest date
        today_date = dt.datetime(2023, 1, 1) 
        
        # Calculate RFM metrics
        rfm_df = df.groupby('CustomerID').agg(
            Recency=('Date', lambda date: (today_date - date.max()).days),
            Frequency=('TransactionID', 'nunique'),
            Monetary=('TotalPrice', 'sum')
        ).reset_index()
        # Prepare data for clustering
        rfm_data_for_clustering = rfm_df[['Recency', 'Frequency', 'Monetary']]
        scaler = StandardScaler()
        rfm_scaled = scaler.fit_transform(rfm_data_for_clustering)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto') # Added n_init for newer KMeans versions
        rfm_df['Cluster'] = kmeans.fit_predict(rfm_scaled) # Use scaled data for clustering

        return rfm_df, kmeans.inertia_, silhouette_score(rfm_scaled, kmeans.labels_) if n_clusters > 1 else None
    
    except Exception as e:
        st.error(f"Error in RFM clustering: {str(e)}")
        return pd.DataFrame(), None, None

def calculate_elbow_silhouette(rfm_data, max_k=10):
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_data[['Recency', 'Frequency', 'Monetary']])
    
    wcss = []
    silhouette_scores = []
    k_range = range(2, max_k + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(rfm_scaled)
        wcss.append(kmeans.inertia_)
        
        # Hitung silhouette score untuk setiap k (bukan hanya 2 dan 11)
        score = silhouette_score(rfm_scaled, kmeans.labels_)
        silhouette_scores.append(score)
    
    return {
        'k_range': list(k_range),
        'wcss': wcss,
        'silhouette_scores': silhouette_scores
    }



# --- MARKET BASKET ANALYSIS (MBA) ---
def perform_mba(df_cleaned, rfm_df, segment_id, min_support=0.01, min_confidence=0.2, min_lift=1.0):

    try:
        # Merge RFM cluster info to transaction data
        df_merged = df_cleaned.merge(rfm_df[['CustomerID', 'Cluster']], on='CustomerID', how='left')
        
        # Filter data by segment
        df_seg = df_merged[df_merged['Cluster'] == segment_id]
        
        if df_seg.empty:
            return {
                'frequent_itemsets': pd.DataFrame(),
                'rules': pd.DataFrame(),
                'top_pairs': pd.DataFrame(),
                'message': f'No data found for segment {segment_id}.'
            }
        if 'Product' not in df_seg.columns:
            st.error("Kolom 'Product' tidak ditemukan dalam data. Pastikan data Anda memiliki kolom nama produk.")
            return {
                'frequent_itemsets': pd.DataFrame(),
                'rules': pd.DataFrame(),
                'top_pairs': pd.DataFrame(),
                'message': "Kolom 'Product' tidak ditemukan."
            }
        
        # Aggregate products by transaction ID
        keranjang = df_seg.groupby('TransactionID')['Product'].apply(list)
        
        if keranjang.empty:
            return {
                'frequent_itemsets': pd.DataFrame(),
                'rules': pd.DataFrame(),
                'top_pairs': pd.DataFrame(),
                'message': 'No transactions found for market basket analysis in this segment.'
            }
        
        # Transaction encoding
        te = TransactionEncoder()
        te_ary = te.fit(keranjang).transform(keranjang)
        basket_df = pd.DataFrame(te_ary, columns=te.columns_)
        
        # Run Apriori algorithm
        frequent_itemsets = apriori(basket_df, min_support=min_support, use_colnames=True)
        if frequent_itemsets.empty:
            return {
                'frequent_itemsets': pd.DataFrame(),
                'rules': pd.DataFrame(),
                'top_pairs': pd.DataFrame(),
                'message': f'No frequent itemsets found with current support threshold ({min_support}) for segment {segment_id}.'
            }
        
        # Generate association rules
        rules = pd.DataFrame()
        if len(frequent_itemsets) > 0: # Ensure there are itemsets to form rules
            try:
                rules = association_rules(frequent_itemsets, metric='lift', min_threshold=min_confidence)
                frequent_itemsets['itemsets'] = frequent_itemsets['itemsets'].apply(lambda x: ', '.join(list(x)))

                if not rules.empty:
                    # Filter and format rules
                    rules = rules[(rules['confidence'] >= min_confidence) & (rules['lift'] >= min_lift)]
                    rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x))) 
                    rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
                    rules = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
                    rules = rules.sort_values(by='lift', ascending=False).reset_index(drop=True)
                    
            except Exception as e:
                st.warning(f"Could not generate association rules for segment {segment_id}: {str(e)}")
        
        # Get top pairs (for visualization or separate analysis)
        top_pairs = frequent_itemsets[frequent_itemsets['itemsets'].apply(lambda x: len(x) == 2)] 
        top_pairs = top_pairs.sort_values(by='support', ascending=False)
        if not top_pairs.empty:
            top_pairs['Item 1'] = top_pairs['itemsets'].apply(lambda x: list(x)[0])
            top_pairs['Item 2'] = top_pairs['itemsets'].apply(lambda x: list(x)[1])
            top_pairs['pair'] = top_pairs['Item 1'] + ' & ' + top_pairs['Item 2']
            top_pairs = top_pairs[['pair', 'support']]      
        return {
            'frequent_itemsets': frequent_itemsets.sort_values(by='support', ascending=False),
            'rules': rules, 
            'all_rules': rules, 
            'top_pairs': top_pairs if not top_pairs.empty else pd.DataFrame(),
            'message': f'Analysis completed for segment {segment_id}.'
        }
        
    except Exception as e:
        st.error(f"Error in Market Basket Analysis for segment {segment_id}: {str(e)}")
        return {
            'frequent_itemsets': pd.DataFrame(),
            'rules': pd.DataFrame(),
            'top_pairs': pd.DataFrame(),
            'message': f'Error occurred: {str(e)}'
        }

# --- GAP RULES ANALYSIS ---
import pandas as pd

def get_gap_rules(rules_segment_A, rules_segment_B):
    """
    Mengidentifikasi aturan asosiasi yang ada di rules_segment_A tetapi tidak ada di rules_segment_B,
    menggunakan nama variabel asli.

    Args:
        rules_segment_A (pd.DataFrame): DataFrame aturan asosiasi untuk Segment A,
                                        dengan kolom 'antecedents' dan 'consequents' (bisa frozenset, list, atau string).
                                        Harus juga memiliki 'support', 'confidence', 'lift'.
        rules_segment_B (pd.DataFrame): DataFrame aturan asosiasi untuk Segment B,
                                        dengan kolom 'antecedents' dan 'consequents' (bisa frozenset, list, atau string).

    Returns:
        pd.DataFrame: DataFrame yang berisi 'gap rules' dengan metrik support, confidence, dan lift dari Segment A.
                      Kolom 'antecedents' dan 'consequents' akan dikonversi ke list untuk tampilan.
                      Nama kolom metrik tetap 'support', 'confidence', 'lift'.
    """
    if rules_segment_A.empty:
        return pd.DataFrame()

    # Fungsi pembantu untuk mengkonversi kolom antecedents/consequents ke frozenset dari list/string
    def _to_frozenset_col(df_col):
        if df_col.empty:
            return pd.Series([], dtype='object') # Return empty Series of object type
        first_val = df_col.iloc[0]
        if isinstance(first_val, frozenset):
            return df_col # Sudah frozenset
        elif isinstance(first_val, str):
            # Sortir item sebelum frozenset untuk konsistensi hash
            return df_col.apply(lambda x: frozenset(sorted(x.split(', '))))
        elif isinstance(first_val, (list, set)):
            # Sortir item sebelum frozenset untuk konsistensi hash
            return df_col.apply(lambda x: frozenset(sorted(list(x))))
        return df_col # Fallback, asumsi sudah dalam bentuk yang bisa di-hash

    # Buat salinan DataFrame untuk menghindari modifikasi input asli
    rules_A_processed = rules_segment_A.copy()
    rules_B_processed = rules_segment_B.copy()

    # Konversi kolom 'antecedents' dan 'consequents' ke frozenset di salinan DataFrame
    rules_A_processed['antecedents_fs_temp'] = _to_frozenset_col(rules_A_processed['antecedents'])
    rules_A_processed['consequents_fs_temp'] = _to_frozenset_col(rules_A_processed['consequents'])
    rules_B_processed['antecedents_fs_temp'] = _to_frozenset_col(rules_B_processed['antecedents'])
    rules_B_processed['consequents_fs_temp'] = _to_frozenset_col(rules_B_processed['consequents'])

    # Buat set dari tuple (frozenset(antecedents), frozenset(consequents)) untuk perbandingan
    rules_A_set = set(
        (row['antecedents_fs_temp'], row['consequents_fs_temp'])
        for _, row in rules_A_processed.iterrows()
    )
    rules_B_set = set(
        (row['antecedents_fs_temp'], row['consequents_fs_temp'])
        for _, row in rules_B_processed.iterrows()
    )

    # Temukan aturan di A yang tidak ada di B (GAP Rules)
    gap_rules_tuples = rules_A_set - rules_B_set

    # Buat DataFrame sementara dari set gap_rules
    # Gunakan kolom sementara untuk merge
    temp_gap_df = pd.DataFrame(list(gap_rules_tuples), columns=['antecedents_fs_temp', 'consequents_fs_temp'])

    # Gabungkan dengan rules_A_processed untuk mendapatkan metrik asli
    gap_rules_df = temp_gap_df.merge(
        rules_A_processed[['antecedents_fs_temp', 'consequents_fs_temp', 'support', 'confidence', 'lift']],
        on=['antecedents_fs_temp', 'consequents_fs_temp'],
        how='left'
    )

    # Konversi frozenset kembali ke list untuk kolom 'antecedents' dan 'consequents' asli
    # Ini akan menimpa kolom asli 'antecedents'/'consequents' jika ingin mempertahankan nama yang sama
    # Atau, buat kolom baru dengan nama yang mudah dibaca jika inputnya string
    gap_rules_df['antecedents'] = gap_rules_df['antecedents_fs_temp'].apply(lambda x: sorted(list(x)))
    gap_rules_df['consequents'] = gap_rules_df['consequents_fs_temp'].apply(lambda x: sorted(list(x)))

    # Buat kolom 'rule' string untuk tampilan
    gap_rules_df['rule'] = gap_rules_df['antecedents'].apply(lambda x: ', '.join(x)) + \
                           ' --> ' + gap_rules_df['consequents'].apply(lambda x: ', '.join(x))
    
    # Tambahkan kolom deskripsi
    gap_rules_df['Description'] = gap_rules_df.apply(
        lambda row: f"Produk ini ({', '.join(row['antecedents'])}) sering dibeli bersama ({', '.join(row['consequents'])}) di Segment A, tetapi tidak signifikan di Segment B.",
        axis=1
    )

    # Pilih dan urutkan kolom yang diinginkan (tetap menggunakan nama asli)
    final_gap_rules_df = gap_rules_df[[
        'antecedents', 'consequents', 'support', 'confidence', 'lift', 'rule', 'Description'
    ]]

    return final_gap_rules_df.sort_values(by='lift', ascending=False)

# --- Fungsi Rekomendasi Produk (Disesuaikan dengan Format Output Baru) ---
def recommend_products_based_on_gap(customer_items, gap_rules_df):
    """
    Merekomendasikan produk berdasarkan gap rules yang ditemukan,
    dengan asumsi nama kolom 'antecedents' dan 'consequents' di gap_rules_df.

    Args:
        customer_items (list or set): Daftar produk yang sudah dibeli oleh pelanggan.
        gap_rules_df (pd.DataFrame): DataFrame gap rules yang dihasilkan oleh get_gap_rules.
                                     Kolom 'antecedents' dan 'consequents' diharapkan dalam bentuk list.

    Returns:
        set: Kumpulan produk yang direkomendasikan.
    """
    rekomendasi = set()
    customer_items_frozenset = frozenset(customer_items)

    for _, row in gap_rules_df.iterrows():
        # Kolom 'antecedents' dan 'consequents' sudah dalam bentuk list
        antecedent = frozenset(row['antecedents'])
        consequent = frozenset(row['consequents'])
        
        if antecedent.issubset(customer_items_frozenset):
            rekomendasi.update(consequent - customer_items_frozenset)
            
    return rekomendasi