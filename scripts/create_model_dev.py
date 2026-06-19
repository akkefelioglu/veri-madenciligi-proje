import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.9.0"}
}

cells = []

def md(source):
    cells.append(nbf.v4.new_markdown_cell(source))

def code(source):
    cells.append(nbf.v4.new_code_cell(source))

# ============================================================
# BAŞLIK
# ============================================================
md("""# E-Ticaret Müşteri Segmentasyonu - Sınıflandırma Modelleri
## Final Projesi - Model Geliştirme

**Öğrenci:** Ali Kemal Kefelioğlu - 25221602009

**Amaç:** RFM analizi ile oluşturulan müşteri segmentlerini (Yüksek/Orta/Düşük Değerli) tahmin eden sınıflandırma modelleri geliştirmek.

**Kullanılan Algoritmalar:**
1. Random Forest (Ensemble - Bagging)
2. Support Vector Machine (Kernel Tabanlı)
3. Gradient Boosting (Ensemble - Boosting)

**Değerlendirme:** ROC Curve, Confusion Matrix, K-Fold CV, Hyperparameter Tuning
""")

# ============================================================
# BÖLÜM 1: KÜTÜPHANELER
# ============================================================
md("## 1. Kütüphanelerin Yüklenmesi")

code("""# Temel kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Makine öğrenmesi
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                      cross_val_score, GridSearchCV)
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, precision_score, recall_score,
                              f1_score, roc_curve, auc, ConfusionMatrixDisplay)

# Ayarlar
import warnings
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:.2f}'.format)

print("Tüm kütüphaneler başarıyla yüklendi!")
""")

# ============================================================
# BÖLÜM 2: VERİ YÜKLEME VE TEMİZLEME
# ============================================================
md("""## 2. Veri Setinin Yüklenmesi ve Ön İşleme

Vize projesinde detaylı olarak incelenen Online Retail veri seti yüklenerek,
aynı temizleme adımları uygulanacaktır.""")

code("""# Veri setini yükle
df = pd.read_csv('data/data.csv', encoding='utf-8-sig', parse_dates=['InvoiceDate'])
print(f"Orijinal veri boyutu: {df.shape[0]:,} satır, {df.shape[1]} sütun")

# Veri temizleme adımları
df_clean = df[~df['InvoiceNo'].astype(str).str.startswith('C')].copy()
df_clean = df_clean.dropna(subset=['CustomerID'])
df_clean = df_clean[(df_clean['Quantity'] > 0) & (df_clean['UnitPrice'] > 0)]
df_clean = df_clean.drop_duplicates()
df_clean = df_clean.dropna(subset=['Description'])

# Yeni özellikler
df_clean['TotalPrice'] = df_clean['Quantity'] * df_clean['UnitPrice']
df_clean['CustomerID'] = df_clean['CustomerID'].astype(int)
df_clean['DayOfWeek'] = df_clean['InvoiceDate'].dt.dayofweek
df_clean['Hour'] = df_clean['InvoiceDate'].dt.hour
df_clean['Month'] = df_clean['InvoiceDate'].dt.month

print(f"Temizlenmiş veri boyutu: {df_clean.shape[0]:,} satır")
print(f"Benzersiz müşteri: {df_clean['CustomerID'].nunique():,}")
print(f"Kalan veri oranı: {df_clean.shape[0]/df.shape[0]*100:.1f}%")
""")

# ============================================================
# BÖLÜM 3: MÜŞTERİ ÖZELLİK MATRİSİ
# ============================================================
md("""## 3. Müşteri Düzeyinde Özellik Mühendisliği

Her müşteri için işlem verilerinden anlamlı özellikler (feature) türetilecektir.
Bu özellikler sınıflandırma modellerinin girdisi olacaktır.""")

code("""# Referans tarih (son işlemden 1 gün sonra)
reference_date = df_clean['InvoiceDate'].max() + pd.Timedelta(days=1)

# Müşteri düzeyinde özellikler
customer_features = df_clean.groupby('CustomerID').agg(
    Recency=('InvoiceDate', lambda x: (reference_date - x.max()).days),
    Frequency=('InvoiceNo', 'nunique'),
    Monetary=('TotalPrice', 'sum'),
    AvgOrderValue=('TotalPrice', 'mean'),
    TotalQuantity=('Quantity', 'sum'),
    AvgQuantity=('Quantity', 'mean'),
    UniqueProducts=('StockCode', 'nunique'),
    AvgUnitPrice=('UnitPrice', 'mean'),
    TotalTransactions=('InvoiceNo', 'count'),
    AvgDayOfWeek=('DayOfWeek', 'mean'),
    AvgHour=('Hour', 'mean'),
).reset_index()

# Harcama değişkenliği
spending_std = df_clean.groupby('CustomerID')['TotalPrice'].std().reset_index()
spending_std.columns = ['CustomerID', 'SpendingStd']
customer_features = customer_features.merge(spending_std, on='CustomerID', how='left')
customer_features['SpendingStd'] = customer_features['SpendingStd'].fillna(0)

print(f"Müşteri sayısı: {len(customer_features):,}")
print(f"Özellik sayısı: {customer_features.shape[1] - 1}")
print("\\nÖzellik istatistikleri:")
customer_features.describe()
""")

# ============================================================
# BÖLÜM 4: HEDEF DEĞİŞKEN - MÜŞTERİ SEGMENTLERİ
# ============================================================
md("""## 4. Hedef Değişken: Müşteri Segmentasyonu (RFM Tabanlı)

RFM (Recency, Frequency, Monetary) skorları kullanılarak müşteriler
3 segmente ayrılacaktır:
- **Yüksek Değerli:** Sık alışveriş yapan, yüksek harcamalı, yakın zamanda aktif
- **Orta Değerli:** Orta düzey aktivite gösteren müşteriler
- **Düşük Değerli:** Nadir alışveriş yapan, düşük harcamalı müşteriler""")

code("""# RFM skorlama (1-3 arası)
customer_features['R_Score'] = pd.qcut(customer_features['Recency'], 3, labels=[3, 2, 1])
customer_features['F_Score'] = pd.qcut(
    customer_features['Frequency'].rank(method='first'), 3, labels=[1, 2, 3])
customer_features['M_Score'] = pd.qcut(
    customer_features['Monetary'].rank(method='first'), 3, labels=[1, 2, 3])

customer_features['RFM_Score'] = (customer_features['R_Score'].astype(int) +
                                   customer_features['F_Score'].astype(int) +
                                   customer_features['M_Score'].astype(int))

def segment_customer(score):
    if score >= 7: return 'Yüksek Değerli'
    elif score >= 5: return 'Orta Değerli'
    else: return 'Düşük Değerli'

customer_features['Segment'] = customer_features['RFM_Score'].apply(segment_customer)

print("SEGMENT DAĞILIMI:")
print("-" * 40)
print(customer_features['Segment'].value_counts())
print(f"\\nOranlar:")
print(customer_features['Segment'].value_counts(normalize=True).map('{:.1%}'.format))
""")

code("""# Segment dağılımı görselleştirmesi
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. Pie chart
colors = ['#2ecc71', '#f39c12', '#e74c3c']
segment_counts = customer_features['Segment'].value_counts()
axes[0].pie(segment_counts.values, labels=segment_counts.index, autopct='%1.1f%%',
            colors=colors, startangle=90, textprops={'fontsize': 11})
axes[0].set_title('Müşteri Segment Dağılımı', fontsize=14, fontweight='bold')

# 2. RFM Score histogram
axes[1].hist(customer_features['RFM_Score'], bins=range(3, 11), color='#3498db',
             alpha=0.7, edgecolor='white', align='left')
axes[1].set_xlabel('RFM Skoru')
axes[1].set_ylabel('Müşteri Sayısı')
axes[1].set_title('RFM Skor Dağılımı', fontsize=14, fontweight='bold')

# 3. Segment bazlı Monetary box plot
segment_order = ['Düşük Değerli', 'Orta Değerli', 'Yüksek Değerli']
data_box = [customer_features[customer_features['Segment']==s]['Monetary'].clip(upper=5000)
            for s in segment_order]
bp = axes[2].boxplot(data_box, labels=segment_order, patch_artist=True)
for patch, color in zip(bp['boxes'], colors[::-1]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
axes[2].set_ylabel('Monetary (£)')
axes[2].set_title('Segment Bazlı Harcama Dağılımı', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('data/segment_dagilimi.png', dpi=150, bbox_inches='tight')
plt.show()
""")

# ============================================================
# BÖLÜM 5: TRAIN / VALIDATION / TEST AYRIMI
# ============================================================
md("""## 5. Veri Bölme: Train / Validation / Test

Veriler %60 Eğitim, %20 Doğrulama, %20 Test olarak bölünecektir.
Stratified split ile sınıf dengesi korunacaktır.""")

code("""# Özellik ve hedef değişken
feature_cols = ['Recency', 'Frequency', 'Monetary', 'AvgOrderValue', 'TotalQuantity',
                'AvgQuantity', 'UniqueProducts', 'AvgUnitPrice', 'TotalTransactions',
                'AvgDayOfWeek', 'AvgHour', 'SpendingStd']

X = customer_features[feature_cols].copy()
le = LabelEncoder()
y = le.fit_transform(customer_features['Segment'])

print(f"Sınıf etiketleri: {dict(zip(le.classes_, le.transform(le.classes_)))}")
print(f"Toplam örnek: {len(X)}")

# 60/20/20 split (stratified)
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val)

print(f"\\nEğitim seti: {len(X_train)} ({len(X_train)/len(X)*100:.0f}%)")
print(f"Doğrulama seti: {len(X_val)} ({len(X_val)/len(X)*100:.0f}%)")
print(f"Test seti: {len(X_test)} ({len(X_test)/len(X)*100:.0f}%)")

# Normalizasyon
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("\\nStandardScaler ile normalizasyon uygulandı.")
""")

# ============================================================
# BÖLÜM 6: MODEL 1 - RANDOM FOREST
# ============================================================
md("""## 6. Model 1: Random Forest (Ensemble - Bagging)

Random Forest, birden fazla karar ağacının birleşiminden oluşan bir ensemble yöntemidir.
Bootstrap aggregating (bagging) tekniği ile aşırı öğrenmeye (overfitting) karşı dayanıklıdır.""")

code("""# Random Forest eğitimi
rf_model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf_model.fit(X_train_scaled, y_train)

# Tahmin
rf_pred = rf_model.predict(X_val_scaled)
rf_prob = rf_model.predict_proba(X_val_scaled)

print("RANDOM FOREST - DOĞRULAMA SETİ SONUÇLARI")
print("=" * 60)
print(f"Accuracy: {accuracy_score(y_val, rf_pred):.4f}")
print(f"Precision (weighted): {precision_score(y_val, rf_pred, average='weighted'):.4f}")
print(f"Recall (weighted): {recall_score(y_val, rf_pred, average='weighted'):.4f}")
print(f"F1-Score (weighted): {f1_score(y_val, rf_pred, average='weighted'):.4f}")
print("\\nSınıflandırma Raporu:")
print(classification_report(y_val, rf_pred, target_names=le.classes_))
""")

code("""# Random Forest - Confusion Matrix ve Feature Importance
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Confusion Matrix
cm_rf = confusion_matrix(y_val, rf_pred)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=le.classes_, yticklabels=le.classes_)
axes[0].set_xlabel('Tahmin')
axes[0].set_ylabel('Gerçek')
axes[0].set_title('Random Forest - Confusion Matrix', fontsize=14, fontweight='bold')

# Feature Importance
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]
axes[1].barh(range(len(feature_cols)), importances[indices[::-1]],
             color=plt.cm.viridis(np.linspace(0.3, 0.9, len(feature_cols))))
axes[1].set_yticks(range(len(feature_cols)))
axes[1].set_yticklabels([feature_cols[i] for i in indices[::-1]])
axes[1].set_xlabel('Önem Derecesi')
axes[1].set_title('Random Forest - Özellik Önem Sıralaması', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('data/rf_results.png', dpi=150, bbox_inches='tight')
plt.show()
""")

# ============================================================
# BÖLÜM 7: MODEL 2 - SVM
# ============================================================
md("""## 7. Model 2: Support Vector Machine (Kernel Tabanlı)

SVM, veriyi en iyi ayıran hiper-düzlemi (hyperplane) bulan bir sınıflandırıcıdır.
RBF (Radial Basis Function) kernel kullanılarak doğrusal olmayan karar sınırları oluşturulacaktır.""")

code("""# SVM eğitimi
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
svm_model.fit(X_train_scaled, y_train)

# Tahmin
svm_pred = svm_model.predict(X_val_scaled)
svm_prob = svm_model.predict_proba(X_val_scaled)

print("SVM (RBF Kernel) - DOĞRULAMA SETİ SONUÇLARI")
print("=" * 60)
print(f"Accuracy: {accuracy_score(y_val, svm_pred):.4f}")
print(f"Precision (weighted): {precision_score(y_val, svm_pred, average='weighted'):.4f}")
print(f"Recall (weighted): {recall_score(y_val, svm_pred, average='weighted'):.4f}")
print(f"F1-Score (weighted): {f1_score(y_val, svm_pred, average='weighted'):.4f}")
print("\\nSınıflandırma Raporu:")
print(classification_report(y_val, svm_pred, target_names=le.classes_))
""")

code("""# SVM - Confusion Matrix
fig, ax = plt.subplots(figsize=(8, 6))
cm_svm = confusion_matrix(y_val, svm_pred)
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Oranges', ax=ax,
            xticklabels=le.classes_, yticklabels=le.classes_)
ax.set_xlabel('Tahmin')
ax.set_ylabel('Gerçek')
ax.set_title('SVM - Confusion Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('data/svm_confusion.png', dpi=150, bbox_inches='tight')
plt.show()
""")

# ============================================================
# BÖLÜM 8: MODEL 3 - GRADIENT BOOSTING
# ============================================================
md("""## 8. Model 3: Gradient Boosting (Ensemble - Boosting)

Gradient Boosting, zayıf öğrenicileri (weak learners) ardışık olarak birleştiren
bir ensemble yöntemidir. Her yeni ağaç, önceki modelin hatalarını düzeltmeye odaklanır.""")

code("""# Gradient Boosting eğitimi
gb_model = GradientBoostingClassifier(
    n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
gb_model.fit(X_train_scaled, y_train)

# Tahmin
gb_pred = gb_model.predict(X_val_scaled)
gb_prob = gb_model.predict_proba(X_val_scaled)

print("GRADIENT BOOSTING - DOĞRULAMA SETİ SONUÇLARI")
print("=" * 60)
print(f"Accuracy: {accuracy_score(y_val, gb_pred):.4f}")
print(f"Precision (weighted): {precision_score(y_val, gb_pred, average='weighted'):.4f}")
print(f"Recall (weighted): {recall_score(y_val, gb_pred, average='weighted'):.4f}")
print(f"F1-Score (weighted): {f1_score(y_val, gb_pred, average='weighted'):.4f}")
print("\\nSınıflandırma Raporu:")
print(classification_report(y_val, gb_pred, target_names=le.classes_))
""")

code("""# Gradient Boosting - Confusion Matrix ve Feature Importance
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

cm_gb = confusion_matrix(y_val, gb_pred)
sns.heatmap(cm_gb, annot=True, fmt='d', cmap='Greens', ax=axes[0],
            xticklabels=le.classes_, yticklabels=le.classes_)
axes[0].set_xlabel('Tahmin')
axes[0].set_ylabel('Gerçek')
axes[0].set_title('Gradient Boosting - Confusion Matrix', fontsize=14, fontweight='bold')

importances_gb = gb_model.feature_importances_
indices_gb = np.argsort(importances_gb)[::-1]
axes[1].barh(range(len(feature_cols)), importances_gb[indices_gb[::-1]],
             color=plt.cm.plasma(np.linspace(0.3, 0.9, len(feature_cols))))
axes[1].set_yticks(range(len(feature_cols)))
axes[1].set_yticklabels([feature_cols[i] for i in indices_gb[::-1]])
axes[1].set_xlabel('Önem Derecesi')
axes[1].set_title('Gradient Boosting - Özellik Önem Sıralaması', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('data/gb_results.png', dpi=150, bbox_inches='tight')
plt.show()
""")

# Save Part 1
nb.cells = cells
with open('notebooks/ALIKEMAL_KEFELIOGLU_25221602009_model_dev.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Part 1 notebook oluşturuldu. Hücre sayısı: {len(cells)}")
