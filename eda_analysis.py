import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Türkçe karakter desteği
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class EDAAnalyzer:
    """
    Veri keşif analizi sınıfı
    """
    
    def __init__(self, data_path: str = "data.xlsx"):
        self.data_path = data_path
        self.df = None
        self.output_dir = Path("eda_outputs")
        self.output_dir.mkdir(exist_ok=True)
    
    def load_data(self):
        """Veriyi yükle"""
        print("📊 Veri yükleniyor...")
        self.df = pd.read_excel(self.data_path)
        print(f"✓ {self.df.shape[0]} satır, {self.df.shape[1]} sütun yüklendi\n")
        return self
    
    def basic_info(self):
        """Temel veri bilgileri"""
        print("=" * 80)
        print("TEMEL VERİ BİLGİLERİ")
        print("=" * 80)
        
        print("\n1. Veri Boyutu:")
        print(f"   Satır sayısı: {self.df.shape[0]}")
        print(f"   Sütun sayısı: {self.df.shape[1]}")
        
        print("\n2. Sütun Tipleri:")
        print(self.df.dtypes)
        
        print("\n3. Eksik Değerler:")
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        missing_df = pd.DataFrame({
            'Eksik Sayısı': missing,
            'Yüzde (%)': missing_pct
        })
        print(missing_df[missing_df['Eksik Sayısı'] > 0])
        
        print("\n4. İlk 5 Kayıt:")
        print(self.df.head())
        
        return self
    
    def analyze_targets(self):
        """Hedef değişkenleri analiz et"""
        print("\n" + "=" * 80)
        print("HEDEF DEĞİŞKEN ANALİZİ")
        print("=" * 80)
        
        targets = ['seller_number', 'customer_number', 'main_account']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('Hedef Değişken Dağılımları', fontsize=16, y=1.02)
        
        for idx, target in enumerate(targets):
            # Unique değer sayısı
            unique_count = self.df[target].nunique()
            missing_count = self.df[target].isnull().sum()
            
            print(f"\n{target}:")
            print(f"   Unique değer: {unique_count}")
            print(f"   Eksik değer: {missing_count} ({missing_count/len(self.df)*100:.1f}%)")
            print(f"   En sık 5 değer:")
            print(self.df[target].value_counts().head())
            
            # Bar plot - top 10
            top_values = self.df[target].value_counts().head(10)
            axes[idx].barh(range(len(top_values)), top_values.values, color='skyblue')
            axes[idx].set_yticks(range(len(top_values)))
            axes[idx].set_yticklabels(top_values.index, fontsize=8)
            axes[idx].set_xlabel('Frekans')
            axes[idx].set_title(f'{target}\n(Top 10)')
            axes[idx].grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'target_distributions.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Grafik kaydedildi: {self.output_dir / 'target_distributions.png'}")
        plt.close()
        
        return self
    
    def analyze_payment_transaction(self):
        """Payment type ve transaction type ilişkisi"""
        print("\n" + "=" * 80)
        print("PAYMENT TYPE vs TRANSACTION TYPE ANALİZİ")
        print("=" * 80)
        
        # Crosstab
        ct = pd.crosstab(
            self.df['payment_type'],
            self.df['transaction_type'],
            margins=True
        )
        print("\nÇapraz Tablo:")
        print(ct)
        
        # Heatmap
        plt.figure(figsize=(10, 6))
        ct_plot = pd.crosstab(
            self.df['payment_type'],
            self.df['transaction_type']
        )
        sns.heatmap(ct_plot, annot=True, fmt='d', cmap='YlOrRd', cbar_kws={'label': 'Frekans'})
        plt.title('Payment Type vs Transaction Type Heatmap', fontsize=14, pad=20)
        plt.xlabel('Transaction Type')
        plt.ylabel('Payment Type')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'payment_transaction_heatmap.png', dpi=300, bbox_inches='tight')
        print(f"✓ Grafik kaydedildi: {self.output_dir / 'payment_transaction_heatmap.png'}")
        plt.close()
        
        return self
    
    def analyze_amount(self):
        """Amount değişkenini analiz et"""
        print("\n" + "=" * 80)
        print("AMOUNT (TUTAR) ANALİZİ")
        print("=" * 80)
        
        print(f"\nİstatistikler:")
        print(self.df['amount'].describe())
        
        # Outlier analizi
        Q1 = self.df['amount'].quantile(0.25)
        Q3 = self.df['amount'].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((self.df['amount'] < (Q1 - 1.5 * IQR)) | 
                   (self.df['amount'] > (Q3 + 1.5 * IQR))).sum()
        print(f"\nOutlier sayısı (IQR yöntemi): {outliers} ({outliers/len(self.df)*100:.1f}%)")
        
        # Görselleştirme
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Amount (Tutar) Analizi', fontsize=16, y=1.00)
        
        # 1. Histogram
        axes[0, 0].hist(self.df['amount'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        axes[0, 0].set_xlabel('Tutar')
        axes[0, 0].set_ylabel('Frekans')
        axes[0, 0].set_title('Histogram (Normal Skala)')
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # 2. Log scale histogram
        axes[0, 1].hist(np.log1p(self.df['amount']), bins=50, color='salmon', edgecolor='black', alpha=0.7)
        axes[0, 1].set_xlabel('Log(Tutar + 1)')
        axes[0, 1].set_ylabel('Frekans')
        axes[0, 1].set_title('Histogram (Log Skala)')
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # 3. Box plot
        axes[1, 0].boxplot(self.df['amount'], vert=True)
        axes[1, 0].set_ylabel('Tutar')
        axes[1, 0].set_title('Box Plot')
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # 4. Payment type'a göre amount
        payment_amounts = [
            self.df[self.df['payment_type'] == 'S']['amount'].values,
            self.df[self.df['payment_type'] == 'H']['amount'].values
        ]
        axes[1, 1].boxplot(payment_amounts, labels=['Satış (S)', 'Alış (H)'])
        axes[1, 1].set_ylabel('Tutar')
        axes[1, 1].set_title('Payment Type\'a Göre Tutar Dağılımı')
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'amount_analysis.png', dpi=300, bbox_inches='tight')
        print(f"✓ Grafik kaydedildi: {self.output_dir / 'amount_analysis.png'}")
        plt.close()
        
        return self
    
    def analyze_description_length(self):
        """Description uzunluğu analizi"""
        print("\n" + "=" * 80)
        print("DESCRIPTION UZUNLUK ANALİZİ")
        print("=" * 80)
        
        desc_lengths = self.df['description'].astype(str).str.len()
        
        print(f"\nİstatistikler:")
        print(desc_lengths.describe())
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.hist(desc_lengths, bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
        plt.xlabel('Description Uzunluğu (karakter)')
        plt.ylabel('Frekans')
        plt.title('Description Uzunluk Dağılımı')
        plt.grid(axis='y', alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.boxplot(desc_lengths)
        plt.ylabel('Uzunluk (karakter)')
        plt.title('Description Uzunluk Box Plot')
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'description_length.png', dpi=300, bbox_inches='tight')
        print(f"✓ Grafik kaydedildi: {self.output_dir / 'description_length.png'}")
        plt.close()
        
        return self
    
    def analyze_business_rules(self):
        """İş kuralı analizi: Seller vs Customer mutual exclusivity"""
        print("\n" + "=" * 80)
        print("İŞ KURALI ANALİZİ: SELLER vs CUSTOMER")
        print("=" * 80)
        
        # Seller ve Customer beraber dolu mu?
        both_filled = ((~self.df['seller_number'].isnull()) & 
                      (~self.df['customer_number'].isnull())).sum()
        
        seller_filled = (~self.df['seller_number'].isnull()).sum()
        customer_filled = (~self.df['customer_number'].isnull()).sum()
        both_null = ((self.df['seller_number'].isnull()) & 
                    (self.df['customer_number'].isnull())).sum()
        
        print(f"\nSeller dolu: {seller_filled} ({seller_filled/len(self.df)*100:.1f}%)")
        print(f"Customer dolu: {customer_filled} ({customer_filled/len(self.df)*100:.1f}%)")
        print(f"Her ikisi de boş: {both_null} ({both_null/len(self.df)*100:.1f}%)")
        print(f"Her ikisi de dolu: {both_filled} ({both_filled/len(self.df)*100:.1f}%)")
        
        if both_filled == 0:
            print("\n✓ İş kuralı doğrulandı: Seller ve Customer asla birlikte dolu değil (Mutual Exclusivity)")
        else:
            print(f"\n⚠ Uyarı: {both_filled} kayıtta hem seller hem customer dolu!")
        
        # Payment type ile ilişki
        print("\nPayment Type ile İlişki:")
        seller_payment = self.df[~self.df['seller_number'].isnull()]['payment_type'].value_counts()
        customer_payment = self.df[~self.df['customer_number'].isnull()]['payment_type'].value_counts()
        
        print(f"\nSeller dolu olduğunda payment_type:")
        print(seller_payment)
        print(f"\nCustomer dolu olduğunda payment_type:")
        print(customer_payment)
        
        # Görselleştirme
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Seller vs Customer dağılımı
        categories = ['Sadece Seller', 'Sadece Customer', 'Her İkisi Boş', 'Her İkisi Dolu']
        values = [
            seller_filled - both_filled,
            customer_filled - both_filled,
            both_null,
            both_filled
        ]
        colors = ['#FF6B6B', '#4ECDC4', '#95E1D3', '#FFE66D']
        axes[0].bar(categories, values, color=colors, alpha=0.8, edgecolor='black')
        axes[0].set_ylabel('Kayıt Sayısı')
        axes[0].set_title('Seller vs Customer Dağılımı')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(axis='y', alpha=0.3)
        
        # Payment type analizi
        payment_data = pd.DataFrame({
            'Seller Dolu': seller_payment,
            'Customer Dolu': customer_payment
        }).fillna(0)
        payment_data.plot(kind='bar', ax=axes[1], color=['#FF6B6B', '#4ECDC4'], alpha=0.8, edgecolor='black')
        axes[1].set_xlabel('Payment Type')
        axes[1].set_ylabel('Kayıt Sayısı')
        axes[1].set_title('Payment Type ile Seller/Customer İlişkisi')
        axes[1].legend(['Seller Dolu', 'Customer Dolu'])
        axes[1].grid(axis='y', alpha=0.3)
        plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=0)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'business_rules.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Grafik kaydedildi: {self.output_dir / 'business_rules.png'}")
        plt.close()
        
        return self
    
    def generate_summary_report(self):
        """Özet rapor oluştur"""
        print("\n" + "=" * 80)
        print("ÖZET RAPOR OLUŞTURULUYOR")
        print("=" * 80)
        
        summary = []
        summary.append("# VERİ ANALİZ RAPORU (EDA)\n")
        summary.append(f"Oluşturulma Tarihi: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        summary.append("## 1. Genel Bilgiler\n")
        summary.append(f"- Toplam Kayıt: {len(self.df)}\n")
        summary.append(f"- Toplam Sütun: {len(self.df.columns)}\n")
        summary.append(f"- Bellek Kullanımı: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n\n")
        
        summary.append("## 2. Eksik Değerler\n")
        missing = self.df.isnull().sum()
        for col in missing[missing > 0].index:
            pct = (missing[col] / len(self.df)) * 100
            summary.append(f"- {col}: {missing[col]} ({pct:.1f}%)\n")
        summary.append("\n")
        
        summary.append("## 3. Hedef Değişkenler\n")
        for target in ['seller_number', 'customer_number', 'main_account']:
            unique = self.df[target].nunique()
            summary.append(f"- {target}: {unique} unique değer\n")
        summary.append("\n")
        
        summary.append("## 4. Amount İstatistikleri\n")
        summary.append(f"- Ortalama: {self.df['amount'].mean():,.2f}\n")
        summary.append(f"- Medyan: {self.df['amount'].median():,.2f}\n")
        summary.append(f"- Minimum: {self.df['amount'].min():,.2f}\n")
        summary.append(f"- Maksimum: {self.df['amount'].max():,.2f}\n\n")
        
        summary.append("## 5. Oluşturulan Grafikler\n")
        summary.append("- target_distributions.png\n")
        summary.append("- payment_transaction_heatmap.png\n")
        summary.append("- amount_analysis.png\n")
        summary.append("- description_length.png\n")
        summary.append("- business_rules.png\n")
        
        # Dosyaya yaz
        with open(self.output_dir / 'eda_summary.md', 'w', encoding='utf-8') as f:
            f.writelines(summary)
        
        print(f"✓ Özet rapor kaydedildi: {self.output_dir / 'eda_summary.md'}")
        
        return self
    
    def run_full_analysis(self):
        """Tüm analizleri çalıştır"""
        self.load_data()
        self.basic_info()
        self.analyze_targets()
        self.analyze_payment_transaction()
        self.analyze_amount()
        self.analyze_description_length()
        self.analyze_business_rules()
        self.generate_summary_report()
        
        print("\n" + "=" * 80)
        print("✅ TÜM ANALİZLER TAMAMLANDI")
        print("=" * 80)
        print(f"Çıktı dizini: {self.output_dir.absolute()}")
        print("\nOluşturulan dosyalar:")
        for file in sorted(self.output_dir.glob('*')):
            print(f"  - {file.name}")


if __name__ == "__main__":
    # EDA çalıştır
    analyzer = EDAAnalyzer("data.xlsx")
    analyzer.run_full_analysis()