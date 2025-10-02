import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import warnings
warnings.filterwarnings('ignore')

# Veriyi yükle
df = pd.read_excel('data.xlsx')

print("=" * 80)
print("📊 VERİ SETİ GENEL BİLGİLERİ")
print("=" * 80)

# Temel bilgiler
print(f"\n✓ Toplam kayıt sayısı: {len(df)}")
print(f"✓ Toplam sütun sayısı: {len(df.columns)}")
print(f"✓ Veri boyutu: {df.shape}")

print("\n" + "=" * 80)
print("📋 SÜTUN BİLGİLERİ VE VERİ TÜRLERİ")
print("=" * 80)
print(df.dtypes)

print("\n" + "=" * 80)
print("🔍 EKSİK VERİ ANALİZİ")
print("=" * 80)

# Eksik veri kontrolü
missing_data = pd.DataFrame({
    'Sütun': df.columns,
    'Eksik Sayısı': df.isnull().sum(),
    'Eksik Yüzdesi': (df.isnull().sum() / len(df) * 100).round(2),
    'Boş String Sayısı': [sum(df[col].astype(str).str.strip() == '') for col in df.columns]
})
missing_data['Toplam Problem'] = missing_data['Eksik Sayısı'] + missing_data['Boş String Sayısı']
print(missing_data[missing_data['Toplam Problem'] > 0])

print("\n" + "=" * 80)
print("📊 SAYISAL DEĞİŞKENLER İSTATİSTİKLERİ")
print("=" * 80)
print(df[['amount']].describe())

print("\n💰 Amount Dağılımı Detayları:")
print(f"  • Minimum: {df['amount'].min():,.2f}")
print(f"  • Maksimum: {df['amount'].max():,.2f}")
print(f"  • Ortalama: {df['amount'].mean():,.2f}")
print(f"  • Medyan: {df['amount'].median():,.2f}")
print(f"  • Std Sapma: {df['amount'].std():,.2f}")

# Outlier analizi (IQR yöntemi)
Q1 = df['amount'].quantile(0.25)
Q3 = df['amount'].quantile(0.75)
IQR = Q3 - Q1
outlier_threshold_low = Q1 - 1.5 * IQR
outlier_threshold_high = Q3 + 1.5 * IQR
outliers = df[(df['amount'] < outlier_threshold_low) | (df['amount'] > outlier_threshold_high)]
print(f"\n⚠️  Outlier tespit edildi: {len(outliers)} kayıt ({len(outliers)/len(df)*100:.1f}%)")

print("\n" + "=" * 80)
print("🏷️  KATEGORİK DEĞİŞKENLER ANALİZİ")
print("=" * 80)

categorical_cols = ['company_code', 'payment_type', 'currency_code', 'transaction_type']
for col in categorical_cols:
    print(f"\n📌 {col.upper()}:")
    print(f"   Benzersiz değer sayısı: {df[col].nunique()}")
    print(f"   Değerler: {df[col].value_counts().to_dict()}")

print("\n" + "=" * 80)
print("🎯 HEDEF DEĞİŞKENLER ANALİZİ (TAHMIN EDİLECEKLER)")
print("=" * 80)

target_cols = ['seller_number', 'customer_number', 'main_account']
for col in target_cols:
    print(f"\n🔸 {col.upper()}:")
    non_null = df[col].notna().sum()
    null_count = df[col].isna().sum()
    print(f"   Dolu kayıt: {non_null} ({non_null/len(df)*100:.1f}%)")
    print(f"   Boş kayıt: {null_count} ({null_count/len(df)*100:.1f}%)")
    print(f"   Benzersiz değer sayısı: {df[col].nunique()}")
    print(f"   Top 10 en sık görülen değer:")
    print(df[col].value_counts().head(10))

print("\n" + "=" * 80)
print("📝 DESCRIPTION ALANI DETAYLI ANALİZ")
print("=" * 80)

df['desc_length'] = df['description'].astype(str).str.len()
df['desc_word_count'] = df['description'].astype(str).str.split().str.len()

print(f"\n📏 Uzunluk İstatistikleri:")
print(f"   • Min karakter: {df['desc_length'].min()}")
print(f"   • Max karakter: {df['desc_length'].max()}")
print(f"   • Ortalama karakter: {df['desc_length'].mean():.1f}")
print(f"   • Ortalama kelime: {df['desc_word_count'].mean():.1f}")

# Yaygın pattern'ler
print(f"\n🔍 Description Pattern Analizi:")
tckn_pattern = len(df[df['description'].str.contains('TCKN', na=False)])
vkn_pattern = len(df[df['description'].str.contains('VKN', na=False)])
hvl_pattern = len(df[df['description'].str.contains('HVL', na=False)])
int_pattern = len(df[df['description'].str.contains('INT-', na=False)])

print(f"   • TCKN içeren: {tckn_pattern} kayıt ({tckn_pattern/len(df)*100:.1f}%)")
print(f"   • VKN içeren: {vkn_pattern} kayıt ({vkn_pattern/len(df)*100:.1f}%)")
print(f"   • HVL içeren: {hvl_pattern} kayıt ({hvl_pattern/len(df)*100:.1f}%)")
print(f"   • INT- içeren: {int_pattern} kayıt ({int_pattern/len(df)*100:.1f}%)")

# En sık geçen kelimeler
all_words = ' '.join(df['description'].astype(str)).upper()
words = re.findall(r'\b[A-ZÇĞİÖŞÜ]{3,}\b', all_words)
word_freq = Counter(words).most_common(20)
print(f"\n📊 En sık geçen kelimeler (Top 20):")
for word, count in word_freq:
    print(f"   • {word}: {count} kez")

print("\n" + "=" * 80)
print("🔗 DEĞIŞKENLER ARASI İLİŞKİ ANALİZİ")
print("=" * 80)

# Transaction type ve payment type ilişkisi
print("\n📊 Transaction Type vs Payment Type:")
crosstab = pd.crosstab(df['transaction_type'], df['payment_type'])
print(crosstab)

# Seller vs Customer dolu olma durumu
print("\n🔗 Seller ve Customer birlikte dolu mu?")
both_filled = len(df[(df['seller_number'].notna()) & (df['customer_number'].notna())])
seller_only = len(df[(df['seller_number'].notna()) & (df['customer_number'].isna())])
customer_only = len(df[(df['seller_number'].isna()) & (df['customer_number'].notna())])
both_empty = len(df[(df['seller_number'].isna()) & (df['customer_number'].isna())])

print(f"   • Her ikisi de dolu: {both_filled} ({both_filled/len(df)*100:.1f}%)")
print(f"   • Sadece seller: {seller_only} ({seller_only/len(df)*100:.1f}%)")
print(f"   • Sadece customer: {customer_only} ({customer_only/len(df)*100:.1f}%)")
print(f"   • Her ikisi de boş: {both_empty} ({both_empty/len(df)*100:.1f}%)")

print("\n" + "=" * 80)
print("✅ ANALİZ TAMAMLANDI")
print("=" * 80)

# Temel özet
print("\n📋 ÖZET VE ÖNERİLER:")
print("\n1. VERİ KALİTESİ:")
print("   ✓ Eksik veriler tespit edildi (özellikle hedef değişkenlerde)")
print("   ✓ Description alanı zengin ve yapılandırılmış bilgi içeriyor")
print("   ✓ Amount değişkeninde outlier'lar mevcut")

print("\n2. FEATURE ENGINEERING FIRSATLARı:")
print("   ✓ Description'dan TCKN/VKN çıkarılabilir")
print("   ✓ Şirket isimleri, işlem türleri extract edilebilir")
print("   ✓ HVL, INT- gibi özel pattern'ler özellik olabilir")
print("   ✓ Kelime frekansları ve TF-IDF kullanılabilir")

print("\n3. MODEL YAKLAŞIMI:")
print("   ✓ Multi-output classification problemi")
print("   ✓ Seller, customer ve main_account tahmin edilecek")
print("   ✓ Dengesiz sınıf dağılımı göz önünde bulundurulmalı")
print("   ✓ Transaction/payment type ilişkisi önemli feature olabilir")

# Veriyi kaydet (temizlenmiş hali için)
df.drop(['desc_length', 'desc_word_count'], axis=1, inplace=True)
print("\n✅ Analiz tamamlandı. Sonraki adıma hazırız!")