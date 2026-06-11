"""
╔══════════════════════════════════════════════════════════════════════╗
║           DRY BEAN DATASET - FULL ML ANALYSIS PIPELINE             ║
║  A. Data Engineering  |  B. Clustering  |  C. Benchmarking  |  D. Integration
║                                                                      ║
║  Requirements:                                                       ║
║    pip install pandas numpy scikit-learn matplotlib seaborn xgboost ║
║                                                                      ║
║  Usage:                                                              ║
║    1. Letakkan file Dry_Bean_Dataset.xlsx di folder yang sama        ║
║    2. Jalankan: python dry_bean_analysis.py                          ║
║    3. Output: folder 'output/' berisi 4 gambar + summary di terminal ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-interactive backend, aman untuk VS Code
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score, adjusted_rand_score,
    accuracy_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

warnings.filterwarnings('ignore')

# ─── KONFIGURASI ──────────────────────────────────────────────────────────────

DATA_PATH   = 'Dry_Bean_Dataset.xlsx'   # Ganti path jika file di folder lain
OUTPUT_DIR  = 'output'
K_RANGE     = range(2, 13)
CHOSEN_K    = 7
TEST_SIZE   = 0.2
RANDOM_SEED = 42

PALETTE = ['#E63946','#F4A261','#2A9D8F','#457B9D','#9B2226','#606C38','#A8DADC']
BG      = '#F8F9FA'
ACCENT  = '#1B4332'

plt.rcParams.update({
    'figure.facecolor': BG,
    'axes.facecolor':   BG,
    'font.family':      'DejaVu Sans',
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'axes.grid':        True,
    'grid.alpha':       0.3,
    'grid.color':       '#CCCCCC',
})

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# A.  DATA ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def load_and_preprocess(path):
    """Load dataset, encode label, scale features, reduce to 2D PCA."""
    print("\n" + "="*60)
    print("A. DATA ENGINEERING")
    print("="*60)

    df = pd.read_excel(path)
    print(f"  Shape             : {df.shape}")
    print(f"  Missing values    : {df.isnull().sum().sum()}")
    print(f"  Duplicate rows    : {df.duplicated().sum()}")
    print(f"\n  Class distribution:\n{df['Class'].value_counts().to_string()}")
    print(f"\n  Descriptive stats (numeric):")
    print(df.describe().round(3).to_string())

    # Label encoding
    le = LabelEncoder()
    df['Class_enc'] = le.fit_transform(df['Class'])
    class_names = le.classes_

    feature_cols = [c for c in df.columns if c not in ['Class', 'Class_enc']]
    X = df[feature_cols].values
    y = df['Class_enc'].values

    # StandardScaler normalisasi
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"\n  Setelah scaling — mean: {X_scaled.mean():.6f}, std: {X_scaled.std():.6f}")

    # PCA 2D untuk visualisasi
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    X_pca = pca.fit_transform(X_scaled)
    print(f"  PCA variance explained (2 komponen): {pca.explained_variance_ratio_.sum()*100:.1f}%")

    return df, X, X_scaled, X_pca, y, feature_cols, class_names, scaler, pca


def plot_eda(df, X_scaled, X_pca, y, feature_cols, class_names):
    """Fig 1: EDA — distribusi kelas, korelasi, boxplot, PCA, stats tabel."""
    fig = plt.figure(figsize=(18, 14), facecolor=BG)
    fig.suptitle('A. Data Engineering & Exploratory Analysis',
                 fontsize=16, fontweight='bold', color=ACCENT, y=0.98)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.4)

    # 1. Distribusi kelas
    ax = fig.add_subplot(gs[0, 0])
    counts = df['Class'].value_counts()
    bars = ax.bar(counts.index, counts.values, color=PALETTE[:len(counts)], edgecolor='white')
    ax.set_title('Class Distribution', fontweight='bold')
    ax.set_xlabel('Class'); ax.set_ylabel('Count')
    ax.tick_params(axis='x', rotation=30)
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 20,
                str(int(b.get_height())), ha='center', fontsize=8)

    # 2. Correlation heatmap
    ax = fig.add_subplot(gs[0, 1:])
    corr = pd.DataFrame(X_scaled, columns=feature_cols).corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap='RdYlGn', center=0, ax=ax,
                cbar_kws={'shrink': 0.7}, linewidths=0.3, vmin=-1, vmax=1)
    ax.set_title('Feature Correlation Heatmap', fontweight='bold')
    ax.tick_params(axis='x', rotation=45, labelsize=7)
    ax.tick_params(axis='y', rotation=0, labelsize=7)

    # 3. Boxplot 6 fitur pertama
    ax = fig.add_subplot(gs[1, :2])
    df_melt = pd.melt(
        pd.DataFrame(X_scaled[:, :6], columns=feature_cols[:6]),
        var_name='Feature', value_name='Scaled Value'
    )
    sns.boxplot(data=df_melt, x='Feature', y='Scaled Value', ax=ax, palette=PALETTE[:6])
    ax.set_title('Feature Distributions (Scaled)', fontweight='bold')
    ax.tick_params(axis='x', rotation=20, labelsize=8)

    # 4. PCA scatter — true labels
    ax = fig.add_subplot(gs[1, 2])
    for i, cls in enumerate(class_names):
        m = y == i
        ax.scatter(X_pca[m, 0], X_pca[m, 1], c=PALETTE[i], label=cls, alpha=0.4, s=5)
    ax.set_title('PCA — True Labels', fontweight='bold')
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
    ax.legend(fontsize=6, markerscale=3)

    # 5. Stats summary table
    ax = fig.add_subplot(gs[2, :])
    stats_data = [[col,
                   f"{df[col].mean():.3f}",
                   f"{df[col].std():.3f}",
                   f"{df[col].min():.3f}",
                   f"{df[col].max():.3f}",
                   f"{df[col].skew():.3f}"]
                  for col in feature_cols]
    ax.axis('off')
    tbl = ax.table(
        cellText=stats_data,
        colLabels=['Feature', 'Mean', 'Std', 'Min', 'Max', 'Skewness'],
        cellLoc='center', loc='center', bbox=[0, 0, 1, 1]
    )
    tbl.auto_set_font_size(False); tbl.set_fontsize(8)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor(ACCENT); cell.set_text_props(color='white', fontweight='bold')
        elif r % 2 == 0:
            cell.set_facecolor('#E8F5E9')
    ax.set_title('Preprocessing Statistics Summary', fontweight='bold', pad=10)

    out = os.path.join(OUTPUT_DIR, 'fig1_eda.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  [Saved] {out}")


# ══════════════════════════════════════════════════════════════════════════════
# B.  UNSUPERVISED ANALYSIS — CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════

def run_clustering(X_scaled, X_pca, y, class_names):
    """Elbow + Silhouette untuk pilih k, lalu KMeans final dengan k=CHOSEN_K."""
    print("\n" + "="*60)
    print("B. UNSUPERVISED ANALYSIS — CLUSTERING")
    print("="*60)

    inertias, sil_scores = [], []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        sil = silhouette_score(X_scaled, labels, sample_size=3000, random_state=RANDOM_SEED)
        sil_scores.append(sil)
        print(f"  k={k:>2}: inertia={km.inertia_:>10.0f}  silhouette={sil:.4f}")

    best_k_sil = list(K_RANGE)[np.argmax(sil_scores)]
    print(f"\n  Best k by silhouette : {best_k_sil}")
    print(f"  Chosen k             : {CHOSEN_K}  (= jumlah kelas asli)")

    # Final clustering
    km_final = KMeans(n_clusters=CHOSEN_K, random_state=RANDOM_SEED, n_init=20)
    cluster_labels = km_final.fit_predict(X_scaled)
    sil_final = silhouette_score(X_scaled, cluster_labels, sample_size=3000, random_state=RANDOM_SEED)
    ari = adjusted_rand_score(y, cluster_labels)
    print(f"\n  Silhouette (k={CHOSEN_K})   : {sil_final:.4f}")
    print(f"  Adjusted Rand Index  : {ari:.4f}")

    # Profil cluster
    df_cl = pd.DataFrame(X_scaled, columns=[f'f{i}' for i in range(X_scaled.shape[1])])
    df_cl['Cluster'] = cluster_labels
    df_cl['True_Class'] = [class_names[i] for i in y]

    dominant = df_cl.groupby('Cluster')['True_Class'].agg(
        lambda x: x.value_counts().index[0])
    purity = df_cl.groupby('Cluster')['True_Class'].agg(
        lambda x: x.value_counts().iloc[0] / len(x) * 100)
    sizes = df_cl.groupby('Cluster').size()

    print("\n  Cluster Info:")
    print(f"  {'Cluster':<10} {'Dominant':<12} {'Purity':>8} {'Size':>6}")
    for cl in range(CHOSEN_K):
        print(f"  {cl:<10} {dominant[cl]:<12} {purity[cl]:>7.1f}%  {sizes[cl]:>6}")

    return cluster_labels, sil_final, ari, best_k_sil, inertias, sil_scores


def plot_clustering(X_scaled, X_pca, y, class_names, cluster_labels,
                    inertias, sil_scores, best_k_sil):
    """Fig 2: Elbow, silhouette, PCA cluster, ukuran cluster, profil, purity."""
    df_cl = pd.DataFrame({'Cluster': cluster_labels,
                          'True_Class': [class_names[i] for i in y]})
    dominant = df_cl.groupby('Cluster')['True_Class'].agg(
        lambda x: x.value_counts().index[0])
    purity = df_cl.groupby('Cluster')['True_Class'].agg(
        lambda x: x.value_counts().iloc[0] / len(x) * 100)

    feature_names = [f'f{i}' for i in range(X_scaled.shape[1])]
    df_feat = pd.DataFrame(X_scaled, columns=feature_names)
    df_feat['Cluster'] = cluster_labels
    cluster_profile = df_feat.groupby('Cluster')[feature_names[:8]].mean()

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), facecolor=BG)
    fig.suptitle('B. Unsupervised Clustering Analysis', fontsize=16,
                 fontweight='bold', color=ACCENT)

    # 2a Elbow
    ax = axes[0, 0]
    ax.plot(list(K_RANGE), inertias, 'o-', color=PALETTE[0], linewidth=2, markersize=7)
    ax.axvline(x=CHOSEN_K, color='red', linestyle='--', alpha=0.7, label=f'k={CHOSEN_K}')
    ax.set_title('Elbow Method', fontweight='bold')
    ax.set_xlabel('k'); ax.set_ylabel('Inertia'); ax.legend()

    # 2b Silhouette vs k
    ax = axes[0, 1]
    ax.plot(list(K_RANGE), sil_scores, 's-', color=PALETTE[2], linewidth=2, markersize=7)
    ax.axvline(x=CHOSEN_K, color='red', linestyle='--', alpha=0.7, label=f'k={CHOSEN_K}')
    ax.axvline(x=best_k_sil, color='orange', linestyle=':', alpha=0.7,
               label=f'k={best_k_sil} (best sil)')
    ax.set_title('Silhouette Score vs k', fontweight='bold')
    ax.set_xlabel('k'); ax.set_ylabel('Silhouette Score'); ax.legend(fontsize=8)

    # 2c PCA — cluster labels
    ax = axes[0, 2]
    for cl in range(CHOSEN_K):
        m = cluster_labels == cl
        ax.scatter(X_pca[m, 0], X_pca[m, 1], c=PALETTE[cl], label=f'C{cl}', alpha=0.4, s=5)
    ax.set_title(f'PCA — KMeans Clusters (k={CHOSEN_K})', fontweight='bold')
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.legend(fontsize=7, markerscale=3)

    # 2d Ukuran cluster
    ax = axes[1, 0]
    counts = pd.Series(cluster_labels).value_counts().sort_index()
    ax.bar(counts.index, counts.values, color=PALETTE[:CHOSEN_K], edgecolor='white')
    ax.set_title('Cluster Sizes', fontweight='bold')
    ax.set_xlabel('Cluster'); ax.set_ylabel('Count')
    for i, v in zip(counts.index, counts.values):
        ax.text(i, v + 20, str(v), ha='center', fontsize=9)

    # 2e Cluster profile heatmap
    ax = axes[1, 1]
    sns.heatmap(cluster_profile, cmap='RdYlGn', center=0, ax=ax,
                cbar_kws={'shrink': 0.7}, linewidths=0.5, annot=True,
                fmt='.2f', annot_kws={'size': 7})
    ax.set_title('Cluster Profiles (Scaled, Top 8 Features)', fontweight='bold')
    ax.tick_params(axis='x', rotation=30, labelsize=7)

    # 2f Purity bars
    ax = axes[1, 2]
    colors_map = {cls: PALETTE[i] for i, cls in enumerate(class_names)}
    bar_cols = [colors_map.get(dominant[cl], PALETTE[0]) for cl in range(CHOSEN_K)]
    bars = ax.bar(range(CHOSEN_K), [purity[cl] for cl in range(CHOSEN_K)],
                  color=bar_cols, edgecolor='white')
    ax.set_title('Cluster Purity (Dominant Class %)', fontweight='bold')
    ax.set_xlabel('Cluster'); ax.set_ylabel('Purity %')
    for b, cl in zip(bars, range(CHOSEN_K)):
        ax.text(b.get_x() + b.get_width()/2, purity[cl] + 1,
                f'{dominant[cl]}\n{purity[cl]:.1f}%', ha='center', fontsize=7)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'fig2_clustering.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] {out}")


# ══════════════════════════════════════════════════════════════════════════════
# C.  SUPERVISED BENCHMARKING
# ══════════════════════════════════════════════════════════════════════════════

def run_benchmarking(X_scaled, y, class_names):
    """Train/test split, fit 8 model, return results dict."""
    print("\n" + "="*60)
    print("C. SUPERVISED BENCHMARKING")
    print("="*60)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y)
    print(f"  Train: {X_train.shape}  |  Test: {X_test.shape}")

    models = {
        # ── Tradisional (sesuai soal UAS) ──────────────────────────────────────
        'Naive Bayes':    GaussianNB(),
        'LDA':            LinearDiscriminantAnalysis(),
        'Logistic Reg.':  LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
        # Tambahan tradisional
        'Decision Tree':  DecisionTreeClassifier(random_state=RANDOM_SEED),
        'SVM (RBF)':      SVC(kernel='rbf', random_state=RANDOM_SEED, probability=True),

        # ── Modern (sesuai soal UAS — ANN) ─────────────────────────────────────
        'ANN (MLP)':      MLPClassifier(
                            hidden_layer_sizes=(64, 32),
                            activation='relu',
                            solver='adam',
                            max_iter=300,
                            random_state=RANDOM_SEED,
                            early_stopping=True,
                            validation_fraction=0.1,
                            verbose=False
                        ),
        # Tambahan modern
        'Random Forest':  RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1),
        'XGBoost':        XGBClassifier(n_estimators=200, random_state=RANDOM_SEED,
                                        eval_metric='mlogloss', n_jobs=-1),
    }
    categories = {
        # Tradisional
        'Naive Bayes':   'Traditional',
        'LDA':           'Traditional',
        'Logistic Reg.': 'Traditional',
        'Decision Tree': 'Traditional',
        'SVM (RBF)':     'Traditional',
        # Modern
        'ANN (MLP)':     'Modern',
        'Random Forest': 'Modern',
        'XGBoost':       'Modern',
    }

    results = {}
    print(f"\n  {'Model':<18} {'Category':<12} {'Accuracy':>9} {'F1 Macro':>9} {'F1 Weighted':>12}")
    print("  " + "-"*65)
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc       = accuracy_score(y_test, y_pred)
        f1_macro  = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        results[name] = {
            'model': model, 'y_pred': y_pred, 'y_test': y_test,
            'Accuracy': acc, 'F1 Macro': f1_macro, 'F1 Weighted': f1_weighted,
            'category': categories[name],
        }
        print(f"  {name:<18} {categories[name]:<12} {acc:>9.4f} {f1_macro:>9.4f} {f1_weighted:>12.4f}")

    best_name = max(results, key=lambda n: results[n]['Accuracy'])
    print(f"\n  Best model: {best_name} (Accuracy={results[best_name]['Accuracy']:.4f})")
    return results, y_test, best_name


def _draw_model_group(axes_row, group_names, results, class_names, group_label, color):
    """
    Helper: isi satu baris axes (3 subplot) untuk satu kelompok model.
    axes_row[0] = accuracy bar
    axes_row[1] = confusion matrix model terbaik dalam grup
    axes_row[2] = per-class F1 semua model dalam grup (grouped bar)
    """
    from matplotlib.patches import Patch

    accs       = [results[n]['Accuracy']   for n in group_names]
    f1s_macro  = [results[n]['F1 Macro']   for n in group_names]
    f1s_weight = [results[n]['F1 Weighted']for n in group_names]
    best_in_group = group_names[int(np.argmax(accs))]

    # ── subplot kiri: Accuracy + F1 bars ────────────────────────────────────
    ax = axes_row[0]
    x = np.arange(len(group_names)); w = 0.28
    ax.bar(x - w,   accs,      w, label='Accuracy',    color=color,    edgecolor='white', alpha=0.95)
    ax.bar(x,       f1s_macro, w, label='F1 Macro',    color='#74C69D',edgecolor='white', alpha=0.95)
    ax.bar(x + w,   f1s_weight,w, label='F1 Weighted', color='#B7E4C7',edgecolor='white', alpha=0.95)
    ax.set_xticks(x); ax.set_xticklabels(group_names, rotation=20, fontsize=8)
    ax.set_title(f'{group_label} — Accuracy & F1', fontweight='bold')
    ax.set_ylim(0.75, 1.04); ax.legend(fontsize=7)
    for xi, (a, fm, fw) in enumerate(zip(accs, f1s_macro, f1s_weight)):
        ax.text(xi - w,  a  + 0.004, f'{a:.3f}',  ha='center', fontsize=6.5)
        ax.text(xi,      fm + 0.004, f'{fm:.3f}', ha='center', fontsize=6.5)
        ax.text(xi + w,  fw + 0.004, f'{fw:.3f}', ha='center', fontsize=6.5)

    # ── subplot tengah: Confusion matrix model terbaik dalam grup ───────────
    ax = axes_row[1]
    cm = confusion_matrix(results[best_in_group]['y_test'],
                          results[best_in_group]['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d',
                cmap='Greens' if group_label == 'Traditional' else 'Blues',
                ax=ax, xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'shrink': 0.7}, linewidths=0.3)
    ax.set_title(f'Confusion Matrix\n(Best: {best_in_group})', fontweight='bold')
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.tick_params(axis='x', rotation=30, labelsize=7)
    ax.tick_params(axis='y', rotation=0,  labelsize=7)

    # ── subplot kanan: Per-class F1 semua model dalam grup ──────────────────
    ax = axes_row[2]
    n_models  = len(group_names)
    n_classes = len(class_names)
    x_cls = np.arange(n_classes)
    bar_width = 0.75 / n_models
    model_colors = ['#1B4332','#40916C','#74C69D','#B7E4C7','#D8F3DC']
    for mi, name in enumerate(group_names):
        cr = classification_report(results[name]['y_test'], results[name]['y_pred'],
                                   target_names=class_names, output_dict=True)
        f1_per_class = [cr[cn]['f1-score'] for cn in class_names]
        offset = (mi - n_models / 2 + 0.5) * bar_width
        ax.bar(x_cls + offset, f1_per_class, bar_width,
               label=name, color=model_colors[mi % len(model_colors)],
               edgecolor='white', alpha=0.9)
    ax.set_xticks(x_cls); ax.set_xticklabels(class_names, rotation=25, fontsize=8)
    ax.set_title(f'{group_label} — Per-Class F1 (All Models)', fontweight='bold')
    ax.set_ylabel('F1 Score'); ax.set_ylim(0.70, 1.05)
    ax.legend(fontsize=7, loc='lower right')


def plot_benchmarking(results, class_names, best_name):
    """
    Fig 3a — Traditional models (2×3 layout):
        Row 0: Accuracy+F1 bars | Confusion matrix (best traditional) | Per-class F1 all traditional
        Row 1: (kosong kiri) | Metrics heatmap traditional | Feature importance Naive Bayes / DT

    Fig 3b — Modern models (2×3 layout):
        Row 0: Accuracy+F1 bars | Confusion matrix (best modern) | Per-class F1 all modern
        Row 1: (kosong kiri) | Metrics heatmap modern | Feature importance RF + XGB
    """
    from matplotlib.patches import Patch

    trad_names   = [n for n in results if results[n]['category'] == 'Traditional']
    modern_names = [n for n in results if results[n]['category'] == 'Modern']
    all_names    = list(results.keys())

    # ══ FIG 3a — TRADITIONAL ═══════════════════════════════════════════════
    fig, axes = plt.subplots(2, 3, figsize=(20, 12), facecolor=BG)
    fig.suptitle('C. Supervised Benchmarking — Traditional Models',
                 fontsize=16, fontweight='bold', color=ACCENT)

    _draw_model_group(axes[0], trad_names, results, class_names,
                      'Traditional', '#40916C')

    # Row 1 kiri — kosong / narasi
    axes[1, 0].axis('off')
    axes[1, 0].text(0.5, 0.5,
        'Traditional Models:\n\n'
        '• Naive Bayes\n• Decision Tree\n• Logistic Regression\n• KNN\n• SVM (RBF)',
        ha='center', va='center', fontsize=11,
        transform=axes[1, 0].transAxes,
        bbox=dict(boxstyle='round', facecolor='#D8F3DC', alpha=0.8))

    # Row 1 tengah — metrics heatmap traditional
    ax = axes[1, 1]
    trad_metrics = pd.DataFrame({
        n: {'Accuracy': results[n]['Accuracy'],
            'F1 Macro': results[n]['F1 Macro'],
            'F1 Weighted': results[n]['F1 Weighted']}
        for n in trad_names}).T
    sns.heatmap(trad_metrics, annot=True, fmt='.4f', cmap='YlGn', ax=ax,
                cbar_kws={'shrink': 0.7}, linewidths=0.5, vmin=0.85, vmax=1.0)
    ax.set_title('Metrics Heatmap — Traditional', fontweight='bold')
    ax.tick_params(axis='x', rotation=0, labelsize=9)
    ax.tick_params(axis='y', rotation=0, labelsize=8)

    # Row 1 kanan — feature importance Decision Tree
    ax = axes[1, 2]
    dt = results['Decision Tree']['model']
    feat_labels = [f'f{i}' for i in range(dt.n_features_in_)]
    imp_dt = pd.Series(dt.feature_importances_, index=feat_labels).sort_values(ascending=False)
    ax.barh(imp_dt.index[:10][::-1], imp_dt.values[:10][::-1],
            color=['#40916C'] * 10, edgecolor='white')
    ax.set_title('Feature Importance\nDecision Tree', fontweight='bold')
    ax.set_xlabel('Importance')

    plt.tight_layout()
    out3a = os.path.join(OUTPUT_DIR, 'fig3a_traditional.png')
    plt.savefig(out3a, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] {out3a}")

    # ══ FIG 3b — MODERN ════════════════════════════════════════════════════
    fig, axes = plt.subplots(2, 3, figsize=(20, 12), facecolor=BG)
    fig.suptitle('C. Supervised Benchmarking — Modern Models',
                 fontsize=16, fontweight='bold', color=ACCENT)

    _draw_model_group(axes[0], modern_names, results, class_names,
                      'Modern', '#1B4332')

    # Row 1 kiri — narasi modern
    axes[1, 0].axis('off')
    axes[1, 0].text(0.5, 0.5,
        'Modern (Ensemble) Models:\n\n'
        '• Random Forest\n• Gradient Boosting\n• XGBoost',
        ha='center', va='center', fontsize=11,
        transform=axes[1, 0].transAxes,
        bbox=dict(boxstyle='round', facecolor='#B7E4C7', alpha=0.8))

    # Row 1 tengah — metrics heatmap modern
    ax = axes[1, 1]
    modern_metrics = pd.DataFrame({
        n: {'Accuracy': results[n]['Accuracy'],
            'F1 Macro': results[n]['F1 Macro'],
            'F1 Weighted': results[n]['F1 Weighted']}
        for n in modern_names}).T
    sns.heatmap(modern_metrics, annot=True, fmt='.4f', cmap='Blues', ax=ax,
                cbar_kws={'shrink': 0.7}, linewidths=0.5, vmin=0.90, vmax=1.0)
    ax.set_title('Metrics Heatmap — Modern', fontweight='bold')
    ax.tick_params(axis='x', rotation=0, labelsize=9)
    ax.tick_params(axis='y', rotation=0, labelsize=8)

    # Row 1 kanan — feature importance Random Forest
    ax = axes[1, 2]
    rf = results['Random Forest']['model']
    feat_labels = [f'f{i}' for i in range(rf.n_features_in_)]
    imp_rf = pd.Series(rf.feature_importances_, index=feat_labels).sort_values(ascending=False)
    ax.barh(imp_rf.index[:10][::-1], imp_rf.values[:10][::-1],
            color=['#1B4332'] * 10, edgecolor='white')
    ax.set_title('Feature Importance\nRandom Forest', fontweight='bold')
    ax.set_xlabel('Importance')

    plt.tight_layout()
    out3b = os.path.join(OUTPUT_DIR, 'fig3b_modern.png')
    plt.savefig(out3b, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] {out3b}")

    # ══ FIG 3c — PERBANDINGAN LENGKAP (Traditional vs Modern side-by-side) ═
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), facecolor=BG)
    fig.suptitle('C. Supervised Benchmarking — Traditional vs Modern (Overview)',
                 fontsize=15, fontweight='bold', color=ACCENT)

    # 3c-1 Accuracy semua model
    ax = axes[0]
    bar_cols = ['#40916C' if results[n]['category'] == 'Traditional' else '#1B4332'
                for n in all_names]
    bars = ax.bar(all_names,
                  [results[n]['Accuracy'] for n in all_names],
                  color=bar_cols, edgecolor='white')
    ax.set_title('Accuracy — All Models', fontweight='bold')
    ax.set_ylabel('Accuracy'); ax.set_ylim(0.75, 1.02)
    ax.tick_params(axis='x', rotation=30, labelsize=8)
    for b, v in zip(bars, [results[n]['Accuracy'] for n in all_names]):
        ax.text(b.get_x() + b.get_width()/2, v + 0.002, f'{v:.3f}', ha='center', fontsize=7)
    ax.legend(handles=[Patch(color='#40916C', label='Traditional'),
                       Patch(color='#1B4332', label='Modern')], fontsize=9)

    # 3c-2 F1 Macro semua model
    ax = axes[1]
    ax.bar(all_names,
           [results[n]['F1 Macro'] for n in all_names],
           color=bar_cols, edgecolor='white')
    ax.set_title('F1 Macro — All Models', fontweight='bold')
    ax.set_ylabel('F1 Macro'); ax.set_ylim(0.75, 1.02)
    ax.tick_params(axis='x', rotation=30, labelsize=8)
    for n, b in zip(all_names, ax.patches):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.002,
                f'{results[n]["F1 Macro"]:.3f}', ha='center', fontsize=7)

    # 3c-3 Full metrics heatmap semua model
    ax = axes[2]
    all_metrics = pd.DataFrame({
        n: {'Accuracy': results[n]['Accuracy'],
            'F1 Macro': results[n]['F1 Macro'],
            'F1 Weighted': results[n]['F1 Weighted']}
        for n in all_names}).T
    sns.heatmap(all_metrics, annot=True, fmt='.4f', cmap='YlGn', ax=ax,
                cbar_kws={'shrink': 0.7}, linewidths=0.5, vmin=0.85, vmax=1.0)
    ax.set_title('Metrics Heatmap — All Models', fontweight='bold')
    ax.tick_params(axis='x', rotation=0, labelsize=9)
    ax.tick_params(axis='y', rotation=0, labelsize=8)

    plt.tight_layout()
    out3c = os.path.join(OUTPUT_DIR, 'fig3c_comparison.png')
    plt.savefig(out3c, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] {out3c}")


# ══════════════════════════════════════════════════════════════════════════════
# D.  INTEGRATION ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def run_integration(X_pca, y, cluster_labels, class_names, sil_final, ari, results, best_name):
    """Analisis alignment cluster ↔ label asli."""
    print("\n" + "="*60)
    print("D. INTEGRATION ANALYSIS")
    print("="*60)

    df_int = pd.DataFrame({
        'Cluster':    cluster_labels,
        'True_Class': [class_names[i] for i in y],
    })

    # Cluster → Class alignment matrix (%)
    alignment = pd.crosstab(
        df_int['Cluster'], df_int['True_Class'], normalize='index') * 100
    print("\n  Cluster → Class Alignment (%):")
    print(alignment.round(1).to_string())

    # Per-cluster stats
    cluster_info = df_int.groupby('Cluster')['True_Class'].agg(
        Dominant=lambda x: x.value_counts().index[0],
        Purity=lambda x: x.value_counts().iloc[0] / len(x) * 100,
        Size='count'
    ).reset_index()
    print("\n  Cluster Info:")
    print(cluster_info.to_string(index=False))

    # Class → dominant cluster alignment
    class_align = []
    for i, cls in enumerate(class_names):
        m = y == i
        cls_clusters = cluster_labels[m]
        dom_cl = pd.Series(cls_clusters).value_counts().index[0]
        pct = (cls_clusters == dom_cl).sum() / m.sum() * 100
        class_align.append({'Class': cls, 'Aligned%': pct})
    class_align_df = pd.DataFrame(class_align)

    overall_purity = (
        df_int.groupby('Cluster')['True_Class']
              .apply(lambda x: x.value_counts().iloc[0])
              .sum() / len(df_int) * 100
    )

    print(f"\n  Adjusted Rand Index     : {ari:.4f}")
    print(f"  Silhouette Score (k={CHOSEN_K}) : {sil_final:.4f}")
    print(f"  Overall Cluster Purity  : {overall_purity:.2f}%")
    print(f"  Class→Cluster Alignment : {class_align_df['Aligned%'].mean():.2f}%")
    print(f"\n  Best Classifier : {best_name}")
    print(f"  Best Accuracy   : {results[best_name]['Accuracy']:.4f}")
    print(f"  Best F1 Macro   : {results[best_name]['F1 Macro']:.4f}")

    return alignment, cluster_info, class_align_df, overall_purity


def plot_integration(X_pca, y, cluster_labels, class_names,
                     alignment, cluster_info, class_align_df,
                     sil_final, ari, results, best_name):
    """
    Fig 4: Layout 3x3
    Baris 1: Heatmap alignment | PCA true labels | PCA clusters
    Baris 2: Purity per cluster | Class alignment bar | Korelasi scatter purity vs F1
    Baris 3: Confusion matrix best model | F1 per kelas best model | Summary box
    """
    colors_map = {cls: PALETTE[i] for i, cls in enumerate(class_names)}

    fig, axes = plt.subplots(3, 3, figsize=(20, 16), facecolor=BG)
    fig.suptitle('D. Integration Analysis: Clustering ↔ Classification Alignment',
                 fontsize=16, fontweight='bold', color=ACCENT)

    # ── BARIS 1 ─────────────────────────────────────────────────────────────

    # 4a Heatmap alignment cluster → kelas
    ax = axes[0, 0]
    sns.heatmap(alignment, annot=True, fmt='.1f', cmap='YlGn', ax=ax,
                cbar_kws={'shrink': 0.7}, linewidths=0.5)
    ax.set_title('Cluster → Class Alignment (%)', fontweight='bold')
    ax.set_xlabel('True Class'); ax.set_ylabel('KMeans Cluster')
    ax.tick_params(axis='x', rotation=30, labelsize=8)

    # 4b PCA true labels
    ax = axes[0, 1]
    for i, cls in enumerate(class_names):
        m = y == i
        ax.scatter(X_pca[m, 0], X_pca[m, 1], c=PALETTE[i],
                   label=cls, alpha=0.3, s=4)
    ax.set_title('PCA — True Labels', fontweight='bold')
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
    ax.legend(fontsize=6, markerscale=3)

    # 4c PCA cluster labels
    ax = axes[0, 2]
    for cl in range(CHOSEN_K):
        m = cluster_labels == cl
        ax.scatter(X_pca[m, 0], X_pca[m, 1], c=PALETTE[cl],
                   label=f'C{cl}', alpha=0.3, s=4)
    ax.set_title('PCA — KMeans Clusters', fontweight='bold')
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
    ax.legend(fontsize=6, markerscale=3)

    # ── BARIS 2 ─────────────────────────────────────────────────────────────

    # 4d Purity per cluster
    ax = axes[1, 0]
    bar_cols = [colors_map.get(row.Dominant, PALETTE[0])
                for row in cluster_info.itertuples()]
    bars = ax.bar(cluster_info['Cluster'], cluster_info['Purity'],
                  color=bar_cols, edgecolor='white')
    avg_pur = cluster_info['Purity'].mean()
    ax.axhline(y=avg_pur, color='red', linestyle='--', alpha=0.7,
               label=f'Avg={avg_pur:.1f}%')
    ax.set_title('Purity per Cluster', fontweight='bold')
    ax.set_xlabel('Cluster'); ax.set_ylabel('Purity %')
    ax.legend(fontsize=8)
    for b, row in zip(bars, cluster_info.itertuples()):
        ax.text(b.get_x() + b.get_width()/2, row.Purity + 1,
                f'{row.Dominant}\n{row.Purity:.1f}%',
                ha='center', fontsize=7)

    # 4e Class → cluster alignment bar
    ax = axes[1, 1]
    bars = ax.bar(class_align_df['Class'], class_align_df['Aligned%'],
                  color=PALETTE[:len(class_names)], edgecolor='white')
    avg_al = class_align_df['Aligned%'].mean()
    ax.axhline(y=avg_al, color='red', linestyle='--', alpha=0.7,
               label=f'Avg={avg_al:.1f}%')
    ax.set_title('Class → Cluster Alignment (%)', fontweight='bold')
    ax.set_ylabel('% Sampel di Cluster Dominan')
    ax.tick_params(axis='x', rotation=25, labelsize=8)
    ax.legend(fontsize=8)
    for b, v in zip(bars, class_align_df['Aligned%']):
        ax.text(b.get_x() + b.get_width()/2, v + 0.5,
                f'{v:.1f}%', ha='center', fontsize=7)

    # 4f Scatter korelasi: cluster purity vs F1 score per kelas
    ax = axes[1, 2]
    cr_best = classification_report(
        results[best_name]['y_test'],
        results[best_name]['y_pred'],
        target_names=class_names, output_dict=True)

    # hitung purity per kelas (bukan per cluster)
    class_purity_vals = []
    for i, cls in enumerate(class_names):
        m = y == i
        cls_clusters = cluster_labels[m]
        dom_cl = pd.Series(cls_clusters).value_counts().index[0]
        pct = (cls_clusters == dom_cl).sum() / m.sum() * 100
        class_purity_vals.append(pct)

    f1_per_cls = [cr_best[cls]['f1-score'] for cls in class_names]

    for i, cls in enumerate(class_names):
        ax.scatter(class_purity_vals[i], f1_per_cls[i],
                   c=PALETTE[i], s=130, zorder=5, edgecolors='white', linewidth=1.5)
        ax.annotate(cls, (class_purity_vals[i], f1_per_cls[i]),
                    textcoords='offset points', xytext=(7, 4), fontsize=8)

    # garis tren sederhana
    z = np.polyfit(class_purity_vals, f1_per_cls, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(class_purity_vals)-2, max(class_purity_vals)+2, 100)
    ax.plot(x_line, p(x_line), 'r--', alpha=0.5, linewidth=1.5, label='Tren')
    ax.set_xlabel('Cluster Purity per Kelas (%)')
    ax.set_ylabel(f'F1 Score ({best_name})')
    ax.set_title('Korelasi: Cluster Purity ↔ F1 Score', fontweight='bold')
    ax.legend(fontsize=8)

    # ── BARIS 3 — INI YANG BARU ─────────────────────────────────────────────

    # 4g Confusion matrix best classifier
    ax = axes[2, 0]
    cm = confusion_matrix(results[best_name]['y_test'],
                          results[best_name]['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'shrink': 0.7}, linewidths=0.3)
    ax.set_title(f'Confusion Matrix — {best_name}\n(Best Classifier)',
                 fontweight='bold')
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.tick_params(axis='x', rotation=30, labelsize=7)
    ax.tick_params(axis='y', rotation=0, labelsize=7)

    # 4h F1 per kelas — best classifier vs cluster purity (grouped bar)
    ax = axes[2, 1]
    x = np.arange(len(class_names))
    w = 0.38

    # normalisasi purity ke skala 0-1 agar bisa dibandingkan di sumbu yang sama
    purity_norm = [v / 100 for v in class_purity_vals]

    bars1 = ax.bar(x - w/2, f1_per_cls, w,
                   label=f'F1 Score ({best_name})',
                   color='#1B4332', edgecolor='white', alpha=0.9)
    bars2 = ax.bar(x + w/2, purity_norm, w,
                   label='Cluster Purity (skala 0-1)',
                   color='#74C69D', edgecolor='white', alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=25, fontsize=8)
    ax.set_title('F1 Score vs Cluster Purity\n(per Varietas)',
                 fontweight='bold')
    ax.set_ylabel('Nilai (0–1)')
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=7)

    for b, v in zip(bars1, f1_per_cls):
        ax.text(b.get_x() + b.get_width()/2, v + 0.01,
                f'{v:.3f}', ha='center', fontsize=6.5, color='#1B4332')
    for b, v in zip(bars2, purity_norm):
        ax.text(b.get_x() + b.get_width()/2, v + 0.01,
                f'{v*100:.0f}%', ha='center', fontsize=6.5, color='#2D6A4F')

    # 4i Summary box — sekarang lebih ringkas karena sudah ada visualisasi
    ax = axes[2, 2]
    ax.axis('off')

    best_acc  = results[best_name]['Accuracy']
    best_f1m  = results[best_name]['F1 Macro']
    best_f1w  = results[best_name]['F1 Weighted']

    summary = (
        f"INTEGRATION SUMMARY\n"
        f"{'─'*32}\n"
        f"CLUSTERING (KMeans k=7)\n"
        f"  Silhouette  : {sil_final:.4f}\n"
        f"  ARI         : {ari:.4f}\n"
        f"  Avg Purity  : {cluster_info['Purity'].mean():.1f}%\n\n"
        f"BEST CLASSIFIER: {best_name}\n"
        f"  Accuracy    : {best_acc:.4f} ({best_acc*100:.2f}%)\n"
        f"  F1 Macro    : {best_f1m:.4f}\n"
        f"  F1 Weighted : {best_f1w:.4f}\n"
        f"{'─'*32}\n"
        f"TEMUAN UTAMA:\n"
        f"  ✅ BOMBAY   : purity 100%, F1 ~1.00\n"
        f"  ✅ HOROZ    : purity 93.9%, F1 ~0.96\n"
        f"  ✅ SEKER    : purity 92.2%, F1 ~0.95\n"
        f"  ✅ DERMASON : purity 92.2%, F1 ~0.94\n"
        f"  🟡 SIRA     : purity 74.8%, F1 ~0.90\n"
        f"  🔴 CALI     : purity ~51%, F1 ~0.88\n"
        f"  🔴 BARBUNYA : tdk dominan, F1 ~0.87\n"
        f"{'─'*32}\n"
        f"Urutan sulit clustering\n"
        f"= urutan sulit klasifikasi\n"
        f"→ Masalah pada FITUR, bukan MODEL"
    )
    ax.text(0.03, 0.5, summary, transform=ax.transAxes, fontsize=8.5,
            verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#D8F3DC', alpha=0.9))
    ax.set_title('Summary Metrics & Temuan', fontweight='bold')

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'fig4_integration.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] {out}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # Cek file input
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"File tidak ditemukan: '{DATA_PATH}'\n"
            f"Pastikan Dry_Bean_Dataset.xlsx ada di folder yang sama dengan script ini."
        )

    # A. Data Engineering
    df, X, X_scaled, X_pca, y, feature_cols, class_names, scaler, pca = \
        load_and_preprocess(DATA_PATH)
    plot_eda(df, X_scaled, X_pca, y, feature_cols, class_names)

    # B. Clustering
    cluster_labels, sil_final, ari, best_k_sil, inertias, sil_scores = \
        run_clustering(X_scaled, X_pca, y, class_names)
    plot_clustering(X_scaled, X_pca, y, class_names, cluster_labels,
                    inertias, sil_scores, best_k_sil)

    # C. Benchmarking
    results, y_test, best_name = run_benchmarking(X_scaled, y, class_names)
    plot_benchmarking(results, class_names, best_name)

    # D. Integration
    alignment, cluster_info, class_align_df, overall_purity = \
        run_integration(X_pca, y, cluster_labels, class_names,
                        sil_final, ari, results, best_name)
    plot_integration(X_pca, y, cluster_labels, class_names,
                     alignment, cluster_info, class_align_df,
                     sil_final, ari, results, best_name)

    print("\n" + "="*60)
    print("SELESAI — cek folder 'output/' untuk 4 gambar hasil analisis")
    print("="*60)


if __name__ == '__main__':
    main()

