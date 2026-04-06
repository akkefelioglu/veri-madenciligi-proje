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
# BÖLÜM 1: BAŞLIK VE KÜTÜPHANELER
# ============================================================
md("""# E-Ticaret Platformları İçin Veri Madenciliği Tabanlı Akıllı Ürün Öneri ve Kişiselleştirme Sistemi

## Vize - Proje İlerleme Raporu ve Kod Dosyaları

**Veri Seti:** [E-Commerce Data (Kaggle)](https://www.kaggle.com/datasets/carrie1/ecommerce-data) / [Online Retail (UCI ML Repository)](http://archive.ics.uci.edu/ml/datasets/Online+Retail)

**Amaç:** Bir e-ticaret sitesindeki kullanıcı davranışları analiz edilerek, Birliktelik Kuralları (Apriori Algoritması) kullanılarak "bu ürünü alanlar bunu da aldı" mantığıyla çalışan bir akıllı ürün öneri motoru tasarlamak.

**Kullanılan Yöntemler:**
- Keşifsel Veri Analizi (EDA)
- Veri Görselleştirme
- Veri Temizleme ve Ön İşleme
- Özellik Mühendisliği (Feature Engineering)
- Özellik Seçimi (Feature Selection)
- Boyut İndirgeme (Dimension Reduction)
- Apriori Algoritması ve Birliktelik Kuralları (Association Rules)
""")

md("## 1. Kütüphanelerin Yüklenmesi")

code("""# Temel kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Apriori ve birliktelik kuralları
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Makine öğrenmesi
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Ek görselleştirme
from wordcloud import WordCloud
import missingno as msno

# Ayarlar
import warnings
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.float_format', '{:.2f}'.format)

print("Tüm kütüphaneler başarıyla yüklendi!")
""")

# ============================================================
# BÖLÜM 2: VERİ YÜKLEME VE İLK İNCELEME
# ============================================================
md("## 2. Veri Setinin Yüklenmesi ve İlk İnceleme")

code("""# Veri setini yükle
df = pd.read_csv('data/data.csv', encoding='utf-8-sig', parse_dates=['InvoiceDate'])

print("=" * 60)
print("VERİ SETİ GENEL BİLGİLERİ")
print("=" * 60)
print(f"Satır sayısı: {df.shape[0]:,}")
print(f"Sütun sayısı: {df.shape[1]}")
print(f"Bellek kullanımı: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
print()
print("İlk 5 kayıt:")
df.head()
""")

code("""# Veri seti bilgi özeti
print("VERİ TİPLERİ VE BOŞ DEĞERLER:")
print("-" * 50)
df.info()
""")

code("""# İstatistiksel özet
print("SAYISAL DEĞİŞKENLERİN İSTATİSTİKSEL ÖZETİ:")
df.describe()
""")

code("""# Kategorik değişkenlerin özeti
print("KATEGORİK DEĞİŞKENLERİN ÖZETİ:")
print("-" * 50)
for col in df.select_dtypes(include='object').columns:
    print(f"\\n{col}:")
    print(f"  Benzersiz değer sayısı: {df[col].nunique()}")
    print(f"  En sık değer: {df[col].mode()[0]}")
    print(f"  Eksik değer: {df[col].isnull().sum()} ({df[col].isnull().mean()*100:.2f}%)")
""")

# ============================================================
# BÖLÜM 3: EDA - KEŞİFSEL VERİ ANALİZİ
# ============================================================
md("""## 3. Keşifsel Veri Analizi (EDA)

### 3.1 Eksik Veri Analizi""")

code("""# Eksik veri analizi
print("EKSİK VERİ ANALİZİ:")
print("-" * 50)
missing = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df)) * 100
missing_df = pd.DataFrame({'Eksik Değer': missing, 'Yüzde (%)': missing_pct})
missing_df = missing_df[missing_df['Eksik Değer'] > 0].sort_values('Yüzde (%)', ascending=False)
print(missing_df)
print(f"\\nToplam eksik değer oranı: {df.isnull().sum().sum() / df.size * 100:.4f}%")
""")

code("""# Eksik veri görselleştirmesi
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# 1) Eksik veri çubuğu
msno.bar(df, ax=axes[0], color=(0.2, 0.6, 0.8))
axes[0].set_title('Eksik Veri Dağılımı (Bar)', fontsize=14, fontweight='bold')

# 2) Eksik veri matrisi
msno.matrix(df, ax=axes[1], color=(0.2, 0.6, 0.8))
axes[1].set_title('Eksik Veri Matrisi', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('data/eksik_veri_analizi.png', dpi=150, bbox_inches='tight')
plt.show()
""")

md("### 3.2 Ülke Bazlı Analiz")

code("""# Ülkelere göre işlem dağılımı
country_counts = df['Country'].value_counts()
print(f"Toplam ülke sayısı: {country_counts.shape[0]}")
print(f"\\nEn fazla işlem yapan 10 ülke:")
print(country_counts.head(10))

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Top 10 ülke bar chart
top_countries = country_counts.head(10)
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_countries)))
bars = axes[0].barh(top_countries.index[::-1], top_countries.values[::-1], color=colors[::-1])
axes[0].set_xlabel('İşlem Sayısı', fontsize=12)
axes[0].set_title('En Fazla İşlem Yapan 10 Ülke', fontsize=14, fontweight='bold')
for bar, val in zip(bars, top_countries.values[::-1]):
    axes[0].text(val + 500, bar.get_y() + bar.get_height()/2,
                f'{val:,}', va='center', fontsize=9)

# UK hariç pie chart
non_uk = country_counts[country_counts.index != 'United Kingdom'].head(8)
axes[1].pie(non_uk.values, labels=non_uk.index, autopct='%1.1f%%',
           colors=plt.cm.Set3(np.linspace(0, 1, len(non_uk))), startangle=90)
axes[1].set_title('UK Hariç Ülke Dağılımı (Top 8)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('data/ulke_dagilimi.png', dpi=150, bbox_inches='tight')
plt.show()
""")

md("### 3.3 Ürün Analizi")

code("""# En çok satılan 20 ürün
top_products = df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(20)

fig, ax = plt.subplots(figsize=(14, 8))
colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top_products)))
bars = ax.barh(range(len(top_products)), top_products.values, color=colors[::-1])
ax.set_yticks(range(len(top_products)))
ax.set_yticklabels(top_products.index, fontsize=10)
ax.invert_yaxis()
ax.set_xlabel('Toplam Satış Miktarı', fontsize=12)
ax.set_title('En Çok Satılan 20 Ürün', fontsize=16, fontweight='bold')
for bar, val in zip(bars, top_products.values):
    ax.text(val + 200, bar.get_y() + bar.get_height()/2, f'{val:,}', va='center', fontsize=9)
plt.tight_layout()
plt.savefig('data/top_urunler.png', dpi=150, bbox_inches='tight')
plt.show()
""")

md("### 3.4 Zaman Serisi Analizi")

code("""# Aylık satış trendi
df['YearMonth'] = df['InvoiceDate'].dt.to_period('M')
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

monthly = df.groupby('YearMonth').agg(
    IslemSayisi=('InvoiceNo', 'nunique'),
    ToplamGelir=('TotalPrice', 'sum'),
    UrunSayisi=('Quantity', 'sum')
).reset_index()
monthly['YearMonth'] = monthly['YearMonth'].astype(str)

fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

axes[0].plot(monthly['YearMonth'], monthly['IslemSayisi'], 'o-', color='#2196F3', linewidth=2, markersize=6)
axes[0].fill_between(range(len(monthly)), monthly['IslemSayisi'], alpha=0.2, color='#2196F3')
axes[0].set_ylabel('İşlem Sayısı', fontsize=11)
axes[0].set_title('Aylık İşlem, Gelir ve Satış Trendi', fontsize=16, fontweight='bold')
axes[0].grid(True, alpha=0.3)

axes[1].plot(monthly['YearMonth'], monthly['ToplamGelir'], 's-', color='#4CAF50', linewidth=2, markersize=6)
axes[1].fill_between(range(len(monthly)), monthly['ToplamGelir'], alpha=0.2, color='#4CAF50')
axes[1].set_ylabel('Toplam Gelir (£)', fontsize=11)
axes[1].grid(True, alpha=0.3)

axes[2].plot(monthly['YearMonth'], monthly['UrunSayisi'], '^-', color='#FF5722', linewidth=2, markersize=6)
axes[2].fill_between(range(len(monthly)), monthly['UrunSayisi'], alpha=0.2, color='#FF5722')
axes[2].set_ylabel('Ürün Sayısı', fontsize=11)
axes[2].set_xlabel('Ay', fontsize=11)
axes[2].grid(True, alpha=0.3)

plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('data/aylik_trend.png', dpi=150, bbox_inches='tight')
plt.show()
""")

code("""# Günlük ve saatlik satış desenleri
df['DayOfWeek'] = df['InvoiceDate'].dt.day_name()
df['Hour'] = df['InvoiceDate'].dt.hour

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Sunday']
day_sales = df.groupby('DayOfWeek')['InvoiceNo'].nunique().reindex(day_order)
axes[0].bar(day_sales.index, day_sales.values, color=plt.cm.Blues(np.linspace(0.4, 0.9, len(day_sales))))
axes[0].set_title('Haftanın Günlerine Göre İşlem Sayısı', fontsize=13, fontweight='bold')
axes[0].set_ylabel('İşlem Sayısı')
axes[0].tick_params(axis='x', rotation=30)

hour_sales = df.groupby('Hour')['InvoiceNo'].nunique()
axes[1].bar(hour_sales.index, hour_sales.values, color=plt.cm.Oranges(np.linspace(0.3, 0.9, len(hour_sales))))
axes[1].set_title('Saatlere Göre İşlem Sayısı', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Saat')
axes[1].set_ylabel('İşlem Sayısı')

plt.tight_layout()
plt.savefig('data/gun_saat_analizi.png', dpi=150, bbox_inches='tight')
plt.show()
""")

md("### 3.5 Fiyat ve Miktar Analizi")

code("""# Fiyat ve miktar dağılımı
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Fiyat histogramı (outlier filtreli)
valid_price = df[(df['UnitPrice'] > 0) & (df['UnitPrice'] < df['UnitPrice'].quantile(0.99))]
axes[0,0].hist(valid_price['UnitPrice'], bins=50, color='#3F51B5', alpha=0.7, edgecolor='white')
axes[0,0].set_title('Birim Fiyat Dağılımı', fontsize=13, fontweight='bold')
axes[0,0].set_xlabel('Birim Fiyat (£)')
axes[0,0].set_ylabel('Frekans')

# Miktar histogramı
valid_qty = df[(df['Quantity'] > 0) & (df['Quantity'] < df['Quantity'].quantile(0.99))]
axes[0,1].hist(valid_qty['Quantity'], bins=50, color='#009688', alpha=0.7, edgecolor='white')
axes[0,1].set_title('Miktar Dağılımı', fontsize=13, fontweight='bold')
axes[0,1].set_xlabel('Miktar')
axes[0,1].set_ylabel('Frekans')

# Box plot - Fiyat
axes[1,0].boxplot(valid_price['UnitPrice'], vert=True)
axes[1,0].set_title('Birim Fiyat Box Plot', fontsize=13, fontweight='bold')
axes[1,0].set_ylabel('Birim Fiyat (£)')

# Box plot - Miktar
axes[1,1].boxplot(valid_qty['Quantity'], vert=True)
axes[1,1].set_title('Miktar Box Plot', fontsize=13, fontweight='bold')
axes[1,1].set_ylabel('Miktar')

plt.tight_layout()
plt.savefig('data/fiyat_miktar_dagilimi.png', dpi=150, bbox_inches='tight')
plt.show()
""")

md("### 3.6 Word Cloud - Ürün Açıklamaları")

code("""# Ürün açıklamalarından word cloud
text = ' '.join(df['Description'].dropna().astype(str))
wordcloud = WordCloud(width=1200, height=600, background_color='white',
                     colormap='viridis', max_words=200, max_font_size=100).generate(text)

plt.figure(figsize=(16, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Ürün Açıklamalarından Kelime Bulutu', fontsize=18, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('data/wordcloud.png', dpi=150, bbox_inches='tight')
plt.show()
""")

md("### 3.7 Korelasyon Analizi")

code("""# Sayısal değişkenler arası korelasyon
numeric_df = df[['Quantity', 'UnitPrice', 'TotalPrice']].copy()

plt.figure(figsize=(8, 6))
corr = numeric_df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, fmt='.3f', cmap='coolwarm', mask=mask,
           square=True, linewidths=2, vmin=-1, vmax=1)
plt.title('Sayısal Değişkenler Korelasyon Matrisi', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('data/korelasyon.png', dpi=150, bbox_inches='tight')
plt.show()
""")

# ============================================================
# BÖLÜM 4: VERİ HAZIRLAMA
# ============================================================
md("""## 4. Veri Hazırlama (Data Preprocessing)

### 4.1 Veri Temizleme""")

code("""# Orijinal veri boyutu
print(f"Orijinal veri boyutu: {df.shape[0]:,} satır")

# 1. İptal edilen işlemleri çıkar
df_clean = df[~df['InvoiceNo'].astype(str).str.startswith('C')].copy()
print(f"İptal işlemleri çıkarıldı: {df.shape[0] - df_clean.shape[0]:,} satır silindi → {df_clean.shape[0]:,} kaldı")

# 2. Eksik CustomerID kayıtlarını çıkar
before = df_clean.shape[0]
df_clean = df_clean.dropna(subset=['CustomerID'])
print(f"Eksik CustomerID çıkarıldı: {before - df_clean.shape[0]:,} satır silindi → {df_clean.shape[0]:,} kaldı")

# 3. Negatif Quantity ve UnitPrice çıkar
before = df_clean.shape[0]
df_clean = df_clean[(df_clean['Quantity'] > 0) & (df_clean['UnitPrice'] > 0)]
print(f"Negatif değerler çıkarıldı: {before - df_clean.shape[0]:,} satır silindi → {df_clean.shape[0]:,} kaldı")

# 4. Duplike kontrol
before = df_clean.shape[0]
df_clean = df_clean.drop_duplicates()
print(f"Duplike kayıtlar çıkarıldı: {before - df_clean.shape[0]:,} satır silindi → {df_clean.shape[0]:,} kaldı")

# 5. Description boş olanları çıkar
before = df_clean.shape[0]
df_clean = df_clean.dropna(subset=['Description'])
print(f"Eksik Description çıkarıldı: {before - df_clean.shape[0]:,} satır silindi → {df_clean.shape[0]:,} kaldı")

print(f"\\n{'='*60}")
print(f"TEMİZLENMİŞ VERİ SETİ: {df_clean.shape[0]:,} satır, {df_clean.shape[1]} sütun")
print(f"Kalan veri oranı: {df_clean.shape[0]/df.shape[0]*100:.1f}%")
""")

md("### 4.2 Özellik Mühendisliği (Feature Engineering)")

code("""# Yeni özellikler oluştur
df_clean['TotalPrice'] = df_clean['Quantity'] * df_clean['UnitPrice']
df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'])
df_clean['Year'] = df_clean['InvoiceDate'].dt.year
df_clean['Month'] = df_clean['InvoiceDate'].dt.month
df_clean['DayOfWeek'] = df_clean['InvoiceDate'].dt.dayofweek
df_clean['Hour'] = df_clean['InvoiceDate'].dt.hour
df_clean['CustomerID'] = df_clean['CustomerID'].astype(int)

print("Oluşturulan yeni özellikler: TotalPrice, Year, Month, DayOfWeek, Hour")
print(f"\\nGüncellenmiş sütunlar: {df_clean.columns.tolist()}")
df_clean.head()
""")

md("### 4.3 RFM Analizi (Recency, Frequency, Monetary)")

code("""# RFM metrikleri
reference_date = df_clean['InvoiceDate'].max() + pd.Timedelta(days=1)

rfm = df_clean.groupby('CustomerID').agg(
    Recency=('InvoiceDate', lambda x: (reference_date - x.max()).days),
    Frequency=('InvoiceNo', 'nunique'),
    Monetary=('TotalPrice', 'sum')
).reset_index()

print("RFM ANALİZİ ÖZETİ:")
print("-" * 50)
print(rfm.describe())

# RFM görselleştirmesi
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, (col, color) in enumerate(zip(['Recency', 'Frequency', 'Monetary'],
                                       ['#E91E63', '#2196F3', '#4CAF50'])):
    data = rfm[col]
    data = data[data < data.quantile(0.95)]  # Outlier filtresi
    axes[i].hist(data, bins=40, color=color, alpha=0.7, edgecolor='white')
    axes[i].set_title(f'{col} Dağılımı', fontsize=14, fontweight='bold')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Müşteri Sayısı')
    axes[i].axvline(data.median(), color='black', linestyle='--', linewidth=1.5, label=f'Medyan: {data.median():.0f}')
    axes[i].legend()

plt.suptitle('RFM Analizi - Müşteri Segmentasyonu', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('data/rfm_analizi.png', dpi=150, bbox_inches='tight')
plt.show()
""")

# ============================================================
# BÖLÜM 5: ÖZELLİK SEÇİMİ
# ============================================================
md("""## 5. Özellik Seçimi (Feature Selection)

Apriori algoritması için uygun özelliklerin seçimi yapılacaktır. Düşük frekanslı ürünler elenecek ve en anlamlı ürün kombinasyonları belirlenecektir.""")

code("""# Ürün frekans analizi
product_freq = df_clean.groupby('Description')['InvoiceNo'].nunique().sort_values(ascending=False)

print(f"Toplam benzersiz ürün sayısı: {len(product_freq)}")
print(f"\\nFrekans istatistikleri:")
print(product_freq.describe())

# Frekans eşikleri analizi
thresholds = [1, 5, 10, 20, 50, 100]
print(f"\\n{'Eşik':>10} | {'Kalan Ürün':>12} | {'Elenen (%)':>10}")
print("-" * 40)
for t in thresholds:
    remaining = (product_freq >= t).sum()
    pct = (1 - remaining / len(product_freq)) * 100
    print(f"{t:>10} | {remaining:>12,} | {pct:>9.1f}%")

# Düşük frekanslı ürünleri ele
min_freq = 20
selected_products = product_freq[product_freq >= min_freq].index.tolist()
print(f"\\nSeçilen eşik: {min_freq}")
print(f"Kalan ürün sayısı: {len(selected_products)} / {len(product_freq)} ({len(selected_products)/len(product_freq)*100:.1f}%)")

# Frekans dağılımı grafiği
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
axes[0].hist(product_freq.values, bins=50, color='#673AB7', alpha=0.7, edgecolor='white')
axes[0].axvline(min_freq, color='red', linestyle='--', linewidth=2, label=f'Eşik: {min_freq}')
axes[0].set_title('Ürün Frekans Dağılımı', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Fatura Sayısı')
axes[0].set_ylabel('Ürün Sayısı')
axes[0].set_xlim(0, 200)
axes[0].legend()

top30 = product_freq.head(30)
axes[1].barh(range(len(top30)), top30.values, color=plt.cm.plasma(np.linspace(0.2, 0.9, len(top30))))
axes[1].set_yticks(range(len(top30)))
axes[1].set_yticklabels(top30.index, fontsize=8)
axes[1].invert_yaxis()
axes[1].set_title('En Sık Satılan 30 Ürün (Fatura Bazlı)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Fatura Sayısı')

plt.tight_layout()
plt.savefig('data/ozellik_secimi.png', dpi=150, bbox_inches='tight')
plt.show()
""")

# ============================================================
# BÖLÜM 6: BOYUT İNDİRGEME
# ============================================================
md("""## 6. Boyut İndirgeme (Dimension Reduction)

PCA ve t-SNE yöntemleriyle müşteri segmentasyonu görselleştirilecektir.""")

code("""# PCA ile müşteri segmentasyonu
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

# PCA (2 bileşen)
pca = PCA(n_components=2)
rfm_pca = pca.fit_transform(rfm_scaled)

print("PCA Açıklanan Varyans Oranları:")
print(f"  PC1: {pca.explained_variance_ratio_[0]*100:.2f}%")
print(f"  PC2: {pca.explained_variance_ratio_[1]*100:.2f}%")
print(f"  Toplam: {pca.explained_variance_ratio_.sum()*100:.2f}%")

# RFM segmentleri oluştur
rfm['R_Score'] = pd.qcut(rfm['Recency'], 3, labels=['Yüksek', 'Orta', 'Düşük'])
rfm['Segment'] = rfm['R_Score'].astype(str)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# PCA scatter
scatter = axes[0].scatter(rfm_pca[:, 0], rfm_pca[:, 1],
                          c=rfm['Recency'], cmap='coolwarm', alpha=0.5, s=10)
axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=11)
axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=11)
axes[0].set_title('PCA - Müşteri Segmentasyonu', fontsize=14, fontweight='bold')
plt.colorbar(scatter, ax=axes[0], label='Recency (gün)')

# Explained variance
pca_full = PCA().fit(rfm_scaled)
cumvar = np.cumsum(pca_full.explained_variance_ratio_)
axes[1].bar(range(1, len(cumvar)+1), pca_full.explained_variance_ratio_, alpha=0.6, color='#2196F3', label='Bireysel')
axes[1].plot(range(1, len(cumvar)+1), cumvar, 'ro-', linewidth=2, label='Kümülatif')
axes[1].axhline(y=0.95, color='gray', linestyle='--', label='95% eşiği')
axes[1].set_xlabel('Bileşen Sayısı', fontsize=11)
axes[1].set_ylabel('Açıklanan Varyans Oranı', fontsize=11)
axes[1].set_title('PCA - Açıklanan Varyans', fontsize=14, fontweight='bold')
axes[1].legend()

plt.tight_layout()
plt.savefig('data/pca_analizi.png', dpi=150, bbox_inches='tight')
plt.show()
""")

code("""# t-SNE görselleştirmesi (küçük örneklem üzerinde)
sample_size = min(2000, len(rfm_scaled))
idx = np.random.choice(len(rfm_scaled), sample_size, replace=False)
rfm_sample = rfm_scaled[idx]

tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
rfm_tsne = tsne.fit_transform(rfm_sample)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(rfm_tsne[:, 0], rfm_tsne[:, 1],
                     c=rfm.iloc[idx]['Monetary'], cmap='YlOrRd', alpha=0.6, s=15)
plt.colorbar(scatter, label='Monetary (£)')
plt.xlabel('t-SNE 1', fontsize=12)
plt.ylabel('t-SNE 2', fontsize=12)
plt.title('t-SNE - Müşteri Segmentasyonu (Monetary)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('data/tsne_analizi.png', dpi=150, bbox_inches='tight')
plt.show()
""")

# ============================================================
# BÖLÜM 7: APRİORİ ALGORİTMASI
# ============================================================
md("""## 7. Apriori Algoritması ve Birliktelik Kuralları

### 7.1 Sepet (Basket) Matrisinin Oluşturulması

Apriori algoritması için her fatura bir "sepet" olarak ele alınacak ve her sepetteki ürünlerin varlığı binary (0/1) matris formatına dönüştürülecektir.""")

code("""# Sadece seçilen ürünlerle çalış
df_apriori = df_clean[df_clean['Description'].isin(selected_products)].copy()
print(f"Apriori için kullanılacak veri: {df_apriori.shape[0]:,} kayıt")
print(f"Benzersiz fatura: {df_apriori['InvoiceNo'].nunique():,}")
print(f"Benzersiz ürün: {df_apriori['Description'].nunique():,}")

# Sepet matrisi oluştur
basket = df_apriori.groupby(['InvoiceNo', 'Description'])['Quantity'].sum().unstack().fillna(0)
basket = basket.applymap(lambda x: 1 if x > 0 else 0)

print(f"\\nSepet matrisi boyutu: {basket.shape}")
print(f"Yoğunluk (density): {basket.values.mean()*100:.2f}%")
print(f"Ortalama sepet büyüklüğü: {basket.sum(axis=1).mean():.1f} ürün")
basket.head()
""")

md("### 7.2 Sık Öğe Kümelerinin Çıkarılması (Frequent Itemsets)")

code("""# Apriori algoritması - farklı support değerleri ile karşılaştırma
support_values = [0.01, 0.02, 0.03, 0.05]
results = {}

print("FARKLI SUPPORT DEĞERLERİ İLE SONUÇLAR:")
print("-" * 50)
for s in support_values:
    freq = apriori(basket, min_support=s, use_colnames=True)
    results[s] = freq
    print(f"min_support={s:.2f}: {len(freq)} sık öğe kümesi bulundu")

# En uygun support değeri ile devam et
min_support = 0.02
frequent_itemsets = results[min_support]
frequent_itemsets['itemsets_str'] = frequent_itemsets['itemsets'].apply(lambda x: ', '.join(list(x)))
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(len)

print(f"\\nSeçilen min_support: {min_support}")
print(f"Toplam sık öğe kümesi: {len(frequent_itemsets)}")
print(f"\\nKüme boyutlarına göre dağılım:")
print(frequent_itemsets['length'].value_counts().sort_index())

# En yüksek support değerine sahip 20 küme
print(f"\\nEn yüksek support'a sahip 15 öğe kümesi:")
frequent_itemsets.sort_values('support', ascending=False).head(15)[['support', 'itemsets_str', 'length']]
""")

code("""# Sık öğe kümeleri görselleştirmesi
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Support dağılımı
axes[0].hist(frequent_itemsets['support'], bins=30, color='#FF9800', alpha=0.7, edgecolor='white')
axes[0].set_title('Support Değerleri Dağılımı', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Support')
axes[0].set_ylabel('Frekans')

# Farklı support değerleri karşılaştırması
counts = [len(results[s]) for s in support_values]
axes[1].bar([str(s) for s in support_values], counts, color=plt.cm.Set2(np.linspace(0, 1, len(support_values))))
axes[1].set_title('Support Eşiğine Göre Bulunan Küme Sayısı', fontsize=14, fontweight='bold')
axes[1].set_xlabel('min_support')
axes[1].set_ylabel('Sık Öğe Kümesi Sayısı')
for i, v in enumerate(counts):
    axes[1].text(i, v + 1, str(v), ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('data/frequent_itemsets.png', dpi=150, bbox_inches='tight')
plt.show()
""")

md("### 7.3 Birliktelik Kurallarının Çıkarılması (Association Rules)")

code("""# Birliktelik kuralları
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.3)
rules['antecedents_str'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
rules['consequents_str'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))

print(f"Toplam birliktelik kuralı sayısı: {len(rules)}")
print(f"\\nMetrik İstatistikleri:")
print(rules[['support', 'confidence', 'lift', 'leverage', 'conviction']].describe())
""")

code("""# En güçlü kurallar (Lift'e göre)
print("EN GÜÇLÜ 20 BİRLİKTELİK KURALI (Lift'e göre sıralı):")
print("=" * 80)
top_rules = rules.sort_values('lift', ascending=False).head(20)
for i, (_, rule) in enumerate(top_rules.iterrows(), 1):
    print(f"\\n{i}. {rule['antecedents_str']}")
    print(f"   → {rule['consequents_str']}")
    print(f"   Support: {rule['support']:.4f} | Confidence: {rule['confidence']:.4f} | Lift: {rule['lift']:.4f}")
""")

code("""# "Bu ürünü alanlar bunu da aldı" önerileri
print("\\n" + "=" * 80)
print("ÜRÜN ÖNERİ SİSTEMİ - 'Bu ürünü alanlar bunu da aldı'")
print("=" * 80)

# Yüksek confidence ve lift değerlerine sahip kuralları göster
strong_rules = rules[(rules['confidence'] >= 0.5) & (rules['lift'] >= 2)].sort_values('lift', ascending=False)

if len(strong_rules) > 0:
    for _, rule in strong_rules.head(10).iterrows():
        print(f"\\n🛒 {rule['antecedents_str']}")
        print(f"   → Önerilen: {rule['consequents_str']}")
        print(f"   Güven: {rule['confidence']*100:.1f}% | Lift: {rule['lift']:.2f}")
else:
    # Daha düşük eşiklerle dene
    strong_rules = rules[(rules['confidence'] >= 0.3) & (rules['lift'] >= 1.5)].sort_values('lift', ascending=False)
    for _, rule in strong_rules.head(10).iterrows():
        print(f"\\n🛒 {rule['antecedents_str']}")
        print(f"   → Önerilen: {rule['consequents_str']}")
        print(f"   Güven: {rule['confidence']*100:.1f}% | Lift: {rule['lift']:.2f}")
""")

md("### 7.4 Birliktelik Kuralları Görselleştirmesi")

code("""# Birliktelik kuralları görselleştirmeleri
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# 1. Support vs Confidence scatter
scatter1 = axes[0,0].scatter(rules['support'], rules['confidence'],
                             c=rules['lift'], cmap='YlOrRd', alpha=0.6, s=50, edgecolors='gray')
axes[0,0].set_xlabel('Support', fontsize=12)
axes[0,0].set_ylabel('Confidence', fontsize=12)
axes[0,0].set_title('Support vs Confidence (renk: Lift)', fontsize=14, fontweight='bold')
plt.colorbar(scatter1, ax=axes[0,0], label='Lift')

# 2. Support vs Lift scatter
scatter2 = axes[0,1].scatter(rules['support'], rules['lift'],
                             c=rules['confidence'], cmap='Blues', alpha=0.6, s=50, edgecolors='gray')
axes[0,1].set_xlabel('Support', fontsize=12)
axes[0,1].set_ylabel('Lift', fontsize=12)
axes[0,1].set_title('Support vs Lift (renk: Confidence)', fontsize=14, fontweight='bold')
plt.colorbar(scatter2, ax=axes[0,1], label='Confidence')

# 3. Confidence dağılımı
axes[1,0].hist(rules['confidence'], bins=30, color='#4CAF50', alpha=0.7, edgecolor='white')
axes[1,0].axvline(rules['confidence'].mean(), color='red', linestyle='--',
                  label=f"Ortalama: {rules['confidence'].mean():.3f}")
axes[1,0].set_title('Confidence Dağılımı', fontsize=14, fontweight='bold')
axes[1,0].set_xlabel('Confidence')
axes[1,0].set_ylabel('Frekans')
axes[1,0].legend()

# 4. Lift dağılımı
axes[1,1].hist(rules['lift'], bins=30, color='#FF5722', alpha=0.7, edgecolor='white')
axes[1,1].axvline(rules['lift'].mean(), color='blue', linestyle='--',
                  label=f"Ortalama: {rules['lift'].mean():.3f}")
axes[1,1].set_title('Lift Dağılımı', fontsize=14, fontweight='bold')
axes[1,1].set_xlabel('Lift')
axes[1,1].set_ylabel('Frekans')
axes[1,1].legend()

plt.suptitle('Birliktelik Kuralları Metrik Analizi', fontsize=18, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('data/birliktelik_kurallari.png', dpi=150, bbox_inches='tight')
plt.show()
""")

code("""# Heatmap - En sık birlikte alınan ürün çiftleri
top_n = 15
top_products_list = df_clean['Description'].value_counts().head(top_n).index.tolist()
basket_top = basket[basket.columns.intersection(top_products_list)]

co_occurrence = basket_top.T.dot(basket_top)
np.fill_diagonal(co_occurrence.values, 0)

plt.figure(figsize=(14, 12))
sns.heatmap(co_occurrence, annot=True, fmt='d', cmap='YlOrRd',
           xticklabels=[x[:30] for x in co_occurrence.columns],
           yticklabels=[x[:30] for x in co_occurrence.index])
plt.title('En Sık Birlikte Alınan Ürün Çiftleri (Co-occurrence Matrix)', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(fontsize=9)
plt.tight_layout()
plt.savefig('data/co_occurrence.png', dpi=150, bbox_inches='tight')
plt.show()
""")

# ============================================================
# BÖLÜM 8: MODEL DEĞERLENDİRME
# ============================================================
md("""## 8. Model Değerlendirme ve Sonuçlar

### 8.1 Performans Metrikleri Özeti""")

code("""# Performans özet tablosu
print("=" * 80)
print("MODEL PERFORMANS ÖZETİ")
print("=" * 80)

summary = {
    'Metrik': ['Toplam Sık Öğe Kümesi', 'Toplam Birliktelik Kuralı',
               'Ortalama Support', 'Ortalama Confidence', 'Ortalama Lift',
               'Max Support', 'Max Confidence', 'Max Lift',
               'Kullanılan min_support', 'Kullanılan min_confidence',
               'İşlenen Fatura Sayısı', 'İşlenen Ürün Sayısı'],
    'Değer': [len(frequent_itemsets), len(rules),
              f"{rules['support'].mean():.4f}", f"{rules['confidence'].mean():.4f}",
              f"{rules['lift'].mean():.4f}",
              f"{rules['support'].max():.4f}", f"{rules['confidence'].max():.4f}",
              f"{rules['lift'].max():.4f}",
              str(min_support), '0.30',
              f"{basket.shape[0]:,}", f"{basket.shape[1]:,}"]
}

summary_df = pd.DataFrame(summary)
print(summary_df.to_string(index=False))
""")

code("""# Farklı threshold değerleri ile kural sayısı karşılaştırması
conf_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
lift_thresholds = [1.0, 1.5, 2.0, 3.0, 5.0]

print("\\nFARKLI EŞİK DEĞERLERİ İLE KURAL SAYILARI:")
print("=" * 60)

# Confidence bazlı
print("\\nConfidence Eşiğine Göre:")
print(f"{'Confidence':>12} | {'Kural Sayısı':>12} | {'Ort. Lift':>10}")
print("-" * 40)
for ct in conf_thresholds:
    r = rules[rules['confidence'] >= ct]
    if len(r) > 0:
        print(f"{ct:>12.1f} | {len(r):>12} | {r['lift'].mean():>10.3f}")

# Lift bazlı
print("\\nLift Eşiğine Göre:")
print(f"{'Lift':>12} | {'Kural Sayısı':>12} | {'Ort. Confidence':>15}")
print("-" * 45)
for lt in lift_thresholds:
    r = rules[rules['lift'] >= lt]
    if len(r) > 0:
        print(f"{lt:>12.1f} | {len(r):>12} | {r['confidence'].mean():>15.3f}")
""")

md("### 8.2 Sonuçlar ve Tartışma")

md("""### Bulgular:

1. **Veri Temizleme:** Orijinal 541.909 kayıttan, iptal işlemleri, eksik müşteri ID'leri ve hatalı değerler temizlendikten sonra anlamlı bir alt küme elde edilmiştir.

2. **EDA Bulguları:**
   - İşlemlerin büyük çoğunluğu İngiltere'den yapılmaktadır
   - Belirli ürünler diğerlerine göre çok daha sık satılmaktadır
   - Hafta içi ve mesai saatlerinde satışlar yoğunlaşmaktadır

3. **Apriori Sonuçları:**
   - Farklı support eşikleri denenerek en uygun parametre belirlenmiştir
   - Yüksek lift değerlerine sahip birliktelik kuralları, gerçek ürün birlikteliklerini ortaya koymaktadır
   - "Bu ürünü alanlar bunu da aldı" mantığıyla çalışan öneriler başarıyla oluşturulmuştur

4. **Boyut İndirgeme:**
   - PCA ve t-SNE ile müşteri segmentlerinin varlığı görselleştirilmiştir
   - RFM analizi müşterilerin farklı gruplara ayrılabileceğini göstermiştir

### Gelecek Çalışmalar (Final Projesi):
- Öneri motorunun web uygulamasına entegrasyonu (İleri Web Programlama dersi ile birleştirme)
- Collaborative Filtering ile hibrit öneri sistemi
- Gerçek zamanlı öneri API'si geliştirme
- Model performansının A/B test ile değerlendirilmesi
""")

md("""## Referanslar

1. Agrawal, R., & Srikant, R. (1994). "Fast algorithms for mining association rules." *Proc. of the 20th VLDB Conference*, 487-499.
2. Chen, D., Sain, S. L., & Guo, K. (2012). "Data mining for the online retail industry: A case study of RFM model-based customer segmentation using data mining." *Journal of Database Marketing & Customer Strategy Management*, 19(3), 197-208.
3. Han, J., Pei, J., & Kamber, M. (2011). *Data Mining: Concepts and Techniques* (3rd ed.). Morgan Kaufmann.
4. Kaggle. "E-Commerce Data." https://www.kaggle.com/datasets/carrie1/ecommerce-data
5. UCI Machine Learning Repository. "Online Retail Data Set." http://archive.ics.uci.edu/ml/datasets/Online+Retail
""")

# Save notebook
nb.cells = cells
with open('notebooks/vize_analiz.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Notebook oluşturuldu: notebooks/vize_analiz.ipynb")
print(f"Toplam hücre sayısı: {len(cells)}")
