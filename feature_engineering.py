import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    """
    Finansal işlem verileri için özel feature engineering sınıfı.
    
    Özellikler:
    - Description alanından akıllı özellik çıkarımı
    - Kategorik encoding
    - İş kuralı tabanlı özellikler
    - Missing value handling
    """
    
    def __init__(self):
        self.label_encoders = {}
        self.feature_stats = {}
        
    def extract_description_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Description alanından zengin özellikler çıkarır.
        """
        print("🔧 Description özelliklerini çıkarıyorum...")
        
        df = df.copy()
        
        # 1. Temel metin özellikleri
        df['desc_length'] = df['description'].astype(str).str.len()
        df['desc_has_numbers'] = df['description'].astype(str).str.contains(r'\d').astype(int)
        df['desc_uppercase_ratio'] = df['description'].astype(str).apply(
            lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0
        )
        
        # 2. Pattern-based features (İŞ KURALLARI)
        # TCKN/VKN varlığı
        df['has_tckn'] = df['description'].astype(str).str.contains('TCKN', na=False).astype(int)
        df['has_vkn'] = df['description'].astype(str).str.contains('VKN', na=False).astype(int)
        df['has_tckn_or_vkn'] = (df['has_tckn'] | df['has_vkn']).astype(int)
        
        # Özel işlem türleri
        df['has_hvl'] = df['description'].astype(str).str.contains('HVL', na=False).astype(int)
        df['has_int'] = df['description'].astype(str).str.contains('INT-', na=False).astype(int)
        df['has_cepsube'] = df['description'].astype(str).str.contains('CEPSUBE', na=False).astype(int)
        
        # Komisyon ve ücret
        df['has_komisyon'] = df['description'].astype(str).str.contains('KOM', na=False).astype(int)
        df['has_ucret'] = df['description'].astype(str).str.contains('UCRET', na=False).astype(int)
        df['has_masraf'] = df['description'].astype(str).str.contains('MASRAF', na=False).astype(int)
        
        # Ödeme türleri
        df['has_odeme'] = df['description'].astype(str).str.contains('ODEME', na=False).astype(int)
        df['has_tahsilat'] = df['description'].astype(str).str.contains('TAHSILAT', na=False).astype(int)
        df['has_havale'] = df['description'].astype(str).str.contains('HAVALE', na=False).astype(int)
        df['has_virman'] = df['description'].astype(str).str.contains('VIRMAN', na=False).astype(int)
        
        # Özel anahtar kelimeler
        df['has_fatura'] = df['description'].astype(str).str.contains('FATURA', na=False).astype(int)
        df['has_cek'] = df['description'].astype(str).str.contains('CEK', na=False).astype(int)
        df['has_eft'] = df['description'].astype(str).str.contains('EFT', na=False).astype(int)
        
        # 3. Şirket/Kişi ismi varlığı (büyük harf ardışıklığı)
        df['has_company_name'] = df['description'].astype(str).str.contains(
            r'[A-ZÇĞİÖŞÜ]{3,}', na=False
        ).astype(int)
        
        # 4. Sayı çıkarımı (belge no, tutar vs.)
        df['number_count'] = df['description'].astype(str).apply(
            lambda x: len(re.findall(r'\d+', x))
        )
        
        # 5. Özel karakterler
        df['has_dash'] = df['description'].astype(str).str.contains('-', na=False).astype(int)
        df['has_dot'] = df['description'].astype(str).str.contains('\.', na=False).astype(int)
        
        print(f"   ✓ {len([c for c in df.columns if c.startswith('has_') or c.startswith('desc_')])} özellik oluşturuldu")
        
        return df
    
    def extract_tfidf_features(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """
        Description alanından TF-IDF features çıkarır (NLP tekniği)
        
        Case Requirement: NLP, TF-IDF kullanımı
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        print("🔧 TF-IDF features oluşturuluyor...")
        
        df = df.copy()
        
        if is_training:
            # Training: Yeni vectorizer oluştur
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=20,  # Top 20 kelime
                min_df=2,  # En az 2 dokümanda geçmeli
                max_df=0.8,  # Maks %80 dokümanda
                ngram_range=(1, 2),  # Unigram ve bigram
                stop_words=None  # Türkçe için custom stop words eklenebilir
            )
            
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(
                df['description'].fillna('').astype(str)
            )
            
            # TF-IDF features'ı dataframe'e ekle
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            tfidf_df = pd.DataFrame(
                tfidf_matrix.toarray(),
                columns=[f'tfidf_{name}' for name in feature_names],
                index=df.index
            )
            
            df = pd.concat([df, tfidf_df], axis=1)
            
            print(f"   ✓ {len(feature_names)} TF-IDF feature oluşturuldu")
            print(f"   Top 5 terms: {list(feature_names[:5])}")
            
        else:
            # Inference: Mevcut vectorizer'ı kullan
            if hasattr(self, 'tfidf_vectorizer'):
                tfidf_matrix = self.tfidf_vectorizer.transform(
                    df['description'].fillna('').astype(str)
                )
                
                feature_names = self.tfidf_vectorizer.get_feature_names_out()
                tfidf_df = pd.DataFrame(
                    tfidf_matrix.toarray(),
                    columns=[f'tfidf_{name}' for name in feature_names],
                    index=df.index
                )
                
                df = pd.concat([df, tfidf_df], axis=1)
            else:
                # Vectorizer yoksa boş columns ekle
                print("   ⚠ TF-IDF vectorizer bulunamadı, inference için sıfırlar kullanılıyor")
        
        return df
    
    def create_business_rules_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        İş kurallarına dayalı özellikler oluşturur.
        """
        print("🔧 İş kuralı özelliklerini oluşturuyorum...")
        
        df = df.copy()
        
        # 1. Amount kategorileri (log scale)
        df['amount_log'] = np.log1p(df['amount'])
        
        # Amount aralıkları
        df['amount_range'] = pd.cut(
            df['amount'], 
            bins=[0, 100, 1000, 10000, 100000, 1000000, float('inf')],
            labels=['micro', 'small', 'medium', 'large', 'xlarge', 'mega']
        )
        
        # 2. Transaction-Payment kombinasyonu (çok önemli!)
        df['trans_pay_combo'] = df['transaction_type'] + '_' + df['payment_type']
        
        # 3. Company code grupları
        df['company_group'] = df['company_code'].apply(
            lambda x: 'group_A' if x in [1000, 1504] else 'group_B'
        )
        
        # 4. Belge numarası özellikleri
        df['doc_number_first_digit'] = df['document_number'].astype(str).str[0]
        df['doc_number_length'] = df['document_number'].astype(str).str.len()
        
        # 5. Özel durum flagleri
        # Çek işlemleri
        df['is_check_transaction'] = (df['transaction_type'] == 'NCHK').astype(int)
        
        # EFT işlemleri
        df['is_eft_transaction'] = (df['transaction_type'] == 'NEFT').astype(int)
        
        # Transfer işlemleri
        df['is_transfer'] = (df['transaction_type'] == 'NTRF').astype(int)
        
        # Yüksek tutarlı işlem
        df['is_high_amount'] = (df['amount'] > df['amount'].quantile(0.75)).astype(int)
        
        # 6. Müşteri/Satıcı indikasyonu (description'a göre)
        df['desc_suggests_seller'] = df['description'].astype(str).str.contains(
            'SATIS|TAHSILAT|GELIR', na=False
        ).astype(int)
        
        df['desc_suggests_customer'] = df['description'].astype(str).str.contains(
            'ODEME|GIDER|TEDARIK', na=False
        ).astype(int)
        
        print(f"   ✓ {len([c for c in df.columns if c.startswith('is_') or c in ['amount_log', 'amount_range', 'trans_pay_combo']])} özellik oluşturuldu")
        
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """
        Kategorik değişkenleri encode eder.
        """
        print("🔧 Kategorik değişkenleri encode ediyorum...")
        
        df = df.copy()
        
        # Encode edilecek kolonlar
        categorical_cols = [
            'payment_type', 'transaction_type', 'trans_pay_combo',
            'amount_range', 'company_group', 'doc_number_first_digit'
        ]
        
        for col in categorical_cols:
            if col not in df.columns:
                continue
                
            if is_training:
                # Training: yeni encoder oluştur
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                # Inference: mevcut encoder'ı kullan
                le = self.label_encoders.get(col)
                if le:
                    # Yeni değerler için unknown handling
                    df[f'{col}_encoded'] = df[col].astype(str).apply(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )
        
        # Company code'u direkt kullan (zaten numeric)
        df['company_code_encoded'] = df['company_code']
        
        print(f"   ✓ {len([c for c in df.columns if c.endswith('_encoded')])} encoded özellik")
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Eksik değerleri akıllıca doldurur.
        """
        print("🔧 Eksik değerleri işliyorum...")
        
        df = df.copy()
        
        # Amount'ta eksik olamaz (zaten yok)
        if df['amount'].isna().any():
            df['amount'].fillna(df['amount'].median(), inplace=True)
        
        # Description eksikse boş string yap
        df['description'].fillna('', inplace=True)
        
        # Kategorik eksikler
        df['payment_type'].fillna('UNKNOWN', inplace=True)
        df['transaction_type'].fillna('UNKNOWN', inplace=True)
        
        print("   ✓ Eksik değerler işlendi")
        
        return df
    
    def create_target_hints(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Hedef değişkenler için ipucu özellikleri oluşturur.
        
        ÖNEMLİ: Seller ve Customer ASLA birlikte dolu olmadığı için
        bu bir iş kuralıdır ve modele öğretilmelidir.
        """
        print("🔧 Hedef değişken ipuçları oluşturuyorum...")
        
        df = df.copy()
        
        # Payment type'a göre seller/customer beklentisi
        # S (Satış): Muhtemelen seller dolu
        # H (Alış): Muhtemelen customer dolu
        df['expects_seller'] = (df['payment_type'] == 'S').astype(int)
        df['expects_customer'] = (df['payment_type'] == 'H').astype(int)
        
        # Transaction type'a göre main_account beklentisi
        # Komisyon/ücret işlemlerinde main_account dolu olabilir
        df['expects_main_account'] = df['transaction_type'].isin(['NCHG', 'NTAX']).astype(int)
        
        print("   ✓ Hedef ipucu özellikleri oluşturuldu")
        
        return df
    
    def transform(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """
        Tüm feature engineering pipeline'ını çalıştırır.
        """
        print("\n" + "="*80)
        print("🚀 FEATURE ENGINEERING PIPELINE BAŞLIYOR")
        print("="*80)
        print(f"📊 Giriş: {df.shape[0]} satır, {df.shape[1]} sütun")
        
        # Pipeline adımları
        df = self.handle_missing_values(df)
        df = self.extract_description_features(df)
        df = self.extract_tfidf_features(df, is_training)  # YENİ: TF-IDF
        df = self.create_business_rules_features(df)
        df = self.create_target_hints(df)
        df = self.encode_categorical_features(df, is_training)
        
        print("\n" + "="*80)
        print("✅ FEATURE ENGINEERING TAMAMLANDI")
        print("="*80)
        print(f"📊 Çıkış: {df.shape[0]} satır, {df.shape[1]} sütun")
        print(f"🎯 Oluşturulan toplam özellik sayısı: {df.shape[1] - 10}")
        
        return df
    
    def get_feature_columns(self) -> list:
        """
        Model için kullanılacak özellik kolonlarını döndürür.
        """
        # Orijinal hedef ve ID kolonlarını hariç tut
        exclude_cols = [
            'seller_number', 'customer_number', 'main_account',
            'description', 'document_number',  # Ham metin ve ID
            'amount_range', 'trans_pay_combo', 'company_group',  # Encoded halleri var
            'doc_number_first_digit', 'payment_type', 'transaction_type'  # Encoded halleri var
        ]
        
        return [col for col in self.feature_stats.keys() if col not in exclude_cols]


# ============================================================================
# KULLANIM ÖRNEĞİ
# ============================================================================

if __name__ == "__main__":
    # Veriyi yükle
    df = pd.read_excel('data.xlsx')
    
    print("\n🎯 ORİJİNAL VERİ:")
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    
    # Feature Engineering
    fe = FeatureEngineer()
    df_transformed = fe.transform(df, is_training=True)
    
    # Oluşturulan özellikleri göster
    new_features = [col for col in df_transformed.columns if col not in df.columns]
    print(f"\n✨ YENİ OLUŞTURULAN ÖZELLİKLER ({len(new_features)} adet):")
    for i, feat in enumerate(new_features, 1):
        print(f"   {i:2d}. {feat}")
    
    # Feature importance için hazır
    print(f"\n📊 MODEL İÇİN HAZIR VERİ:")
    print(f"   Shape: {df_transformed.shape}")
    print(f"   Hedef değişkenler: seller_number, customer_number, main_account")
    print(f"   Özellik sayısı: {len(new_features)}")
    
    # Veriyi kaydet
    df_transformed.to_csv('data_with_features.csv', index=False)
    print("\n✅ İşlenmiş veri 'data_with_features.csv' olarak kaydedildi!")
    
    print("\n" + "="*80)
    print("🎯 SONRAKİ ADIM: Model Training")
    print("="*80)