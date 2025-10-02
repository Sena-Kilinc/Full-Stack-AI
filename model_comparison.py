import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, List

class ModelComparator:
    """
    Model performanslarını karşılaştıran sınıf
    """
    
    def __init__(self):
        self.results = {}
        self.output_dir = Path("comparison_outputs")
        self.output_dir.mkdir(exist_ok=True)
    
    def add_result(self, algorithm: str, metrics: Dict):
        """
        Bir algoritmanın sonuçlarını ekle
        
        Args:
            algorithm: Algoritma adı
            metrics: Metrik dictionary'si
        """
        self.results[algorithm] = metrics
        print(f"✓ {algorithm} sonuçları eklendi")
    def load_from_file_with_mapping(self, algorithms: List[str], mapping: Dict[str, str], models_dir: str = "models"):
        """
        Kaydedilmiş model sonuçlarını özel dosya adı mapping ile yükle
        
        Args:
            algorithms: Algoritma listesi (string formatında)
            mapping: {algo_name: filename} mapping dictionary
            models_dir: Model dizini
        """
        import joblib
        from pathlib import Path
        
        for algo in algorithms:
            try:
                # Mapping'den dosya adını al
                if algo in mapping:
                    filename = mapping[algo]
                    results_path = Path(models_dir) / f'{filename}.pkl'
                else:
                    # Fallback: standart format dene
                    results_path = Path(models_dir) / f'results_{algo}.pkl'
                
                if results_path.exists():
                    results = joblib.load(results_path)
                    self.results[algo] = results
                    print(f"✓ {algo} sonuçları yüklendi")
                else:
                    print(f"⚠ {algo} için sonuç dosyası bulunamadı: {results_path}")
                    
            except Exception as e:
                print(f"⚠ {algo} sonuçları yüklenemedi: {e}")    
    def load_from_file(self, algorithms: List[str], models_dir: str = "models"):
        """
        Kaydedilmiş model sonuçlarını yükle
        
        Args:
            algorithms: Algoritma listesi
            models_dir: Model dizini
        """
        import joblib
        from pathlib import Path
        
        for algo in algorithms:
            try:
                # Algoritma-spesifik results dosyası
                results_path = Path(models_dir) / f'results_{algo}.pkl'
                
                if results_path.exists():
                    results = joblib.load(results_path)
                    self.results[algo] = results
                    print(f"✓ {algo} sonuçları yüklendi")
                else:
                    print(f"⚠ {algo} için sonuç dosyası bulunamadı: {results_path}")
                    
            except Exception as e:
                print(f"⚠ {algo} sonuçları yüklenemedi: {e}")
    
    def create_comparison_table(self):
        """
        Karşılaştırma tablosu oluştur
        """
        print("\n" + "=" * 80)
        print("MODEL PERFORMANS KARŞILAŞTIRMASI")
        print("=" * 80)
        
        if not self.results:
            print("⚠ Henüz sonuç eklenmemiş")
            return None, None
        
        # Test metrics'leri topla
        comparison_data = []
        
        for algo, results in self.results.items():
            test_metrics = results.get('test_metrics', {})
            
            # Her hedef değişken için
            for target in ['seller_number', 'customer_number', 'main_account']:
                if target in test_metrics:
                    metrics = test_metrics[target]
                    comparison_data.append({
                        'Algorithm': algo,
                        'Target': target,
                        'Accuracy': metrics.get('accuracy', 0),
                        'F1 Weighted': metrics.get('f1_weighted', 0),
                        'F1 Macro': metrics.get('f1_macro', 0),
                        'Precision': metrics.get('precision', 0),
                        'Recall': metrics.get('recall', 0)
                    })
        
        df = pd.DataFrame(comparison_data)
        
        print("\nDetaylı Karşılaştırma:")
        print(df.to_string(index=False))
        
        # CSV olarak kaydet
        df.to_csv(self.output_dir / 'model_comparison_detailed.csv', index=False)
        print(f"\n✓ Detaylı tablo kaydedildi: {self.output_dir / 'model_comparison_detailed.csv'}")
        
        # Overall karşılaştırma
        overall_data = []
        for algo, results in self.results.items():
            test_metrics = results.get('test_metrics', {})
            overall = test_metrics.get('overall', {})
            
            overall_data.append({
                'Algorithm': algo,
                'Overall Accuracy': overall.get('accuracy', 0),
                'Overall F1': overall.get('f1_weighted', 0),
                'Train Samples': results.get('train_samples', 0),
                'Test Samples': results.get('test_samples', 0),
                'Feature Count': results.get('feature_count', 0)
            })
        
        overall_df = pd.DataFrame(overall_data)
        print("\n" + "=" * 80)
        print("OVERALL PERFORMANS")
        print("=" * 80)
        print(overall_df.to_string(index=False))
        
        overall_df.to_csv(self.output_dir / 'model_comparison_overall.csv', index=False)
        print(f"\n✓ Overall tablo kaydedildi: {self.output_dir / 'model_comparison_overall.csv'}")
        
        return df, overall_df
    
    def plot_comparison(self):
        """
        Karşılaştırma grafikleri oluştur
        """
        if not self.results:
            print("⚠ Henüz sonuç eklenmemiş")
            return
        
        print("\n" + "=" * 80)
        print("KARŞILAŞTIRMA GRAFİKLERİ OLUŞTURULUYOR")
        print("=" * 80)
        
        # 1. Overall Accuracy Karşılaştırması
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Model Performans Karşılaştırması', fontsize=16, y=0.995)
        
        # Overall accuracy bar chart
        algorithms = list(self.results.keys())
        overall_acc = [self.results[algo]['test_metrics']['overall']['accuracy'] 
                      for algo in algorithms]
        overall_f1 = [self.results[algo]['test_metrics']['overall']['f1_weighted'] 
                     for algo in algorithms]
        
        x = range(len(algorithms))
        width = 0.35
        
        axes[0, 0].bar([i - width/2 for i in x], overall_acc, width, 
                      label='Accuracy', color='skyblue', alpha=0.8, edgecolor='black')
        axes[0, 0].bar([i + width/2 for i in x], overall_f1, width, 
                      label='F1 Weighted', color='salmon', alpha=0.8, edgecolor='black')
        axes[0, 0].set_xlabel('Algorithm')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Overall Performance (Test Set)')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(algorithms, rotation=45, ha='right')
        axes[0, 0].legend()
        axes[0, 0].grid(axis='y', alpha=0.3)
        axes[0, 0].set_ylim([0, 1])
        
        # 2. Target-wise Accuracy
        targets = ['seller_number', 'customer_number', 'main_account']
        target_data = {target: [] for target in targets}
        
        for algo in algorithms:
            for target in targets:
                acc = self.results[algo]['test_metrics'][target]['accuracy']
                target_data[target].append(acc)
        
        x = range(len(algorithms))
        width = 0.25
        
        for idx, target in enumerate(targets):
            offset = (idx - 1) * width
            axes[0, 1].bar([i + offset for i in x], target_data[target], width,
                          label=target.replace('_', ' ').title(), alpha=0.8, edgecolor='black')
        
        axes[0, 1].set_xlabel('Algorithm')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Target-wise Accuracy Comparison')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(algorithms, rotation=45, ha='right')
        axes[0, 1].legend()
        axes[0, 1].grid(axis='y', alpha=0.3)
        axes[0, 1].set_ylim([0, 1])
        
        # 3. F1 Score Heatmap
        f1_data = []
        for algo in algorithms:
            algo_f1 = []
            for target in targets:
                f1 = self.results[algo]['test_metrics'][target]['f1_weighted']
                algo_f1.append(f1)
            f1_data.append(algo_f1)
        
        f1_df = pd.DataFrame(f1_data, 
                            columns=[t.replace('_', ' ').title() for t in targets],
                            index=algorithms)
        
        sns.heatmap(f1_df, annot=True, fmt='.3f', cmap='RdYlGn', 
                   ax=axes[1, 0], cbar_kws={'label': 'F1 Score'},
                   vmin=0, vmax=1, linewidths=1, linecolor='black')
        axes[1, 0].set_title('F1 Score Heatmap (Test Set)')
        axes[1, 0].set_xlabel('Target Variable')
        axes[1, 0].set_ylabel('Algorithm')
        
        # 4. Precision vs Recall
        for target_idx, target in enumerate(targets):
            precisions = []
            recalls = []
            
            for algo in algorithms:
                prec = self.results[algo]['test_metrics'][target]['precision']
                rec = self.results[algo]['test_metrics'][target]['recall']
                precisions.append(prec)
                recalls.append(rec)
            
            axes[1, 1].scatter(recalls, precisions, s=200, alpha=0.6, 
                             label=target.replace('_', ' ').title(),
                             edgecolors='black', linewidths=1.5)
            
            # Algoritma isimlerini ekle
            for i, algo in enumerate(algorithms):
                axes[1, 1].annotate(algo[:4], (recalls[i], precisions[i]),
                                   fontsize=7, ha='center', va='center')
        
        axes[1, 1].set_xlabel('Recall')
        axes[1, 1].set_ylabel('Precision')
        axes[1, 1].set_title('Precision vs Recall (All Targets)')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)
        axes[1, 1].plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Perfect Balance')
        axes[1, 1].set_xlim([0, 1])
        axes[1, 1].set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'model_comparison_charts.png', 
                   dpi=300, bbox_inches='tight')
        print(f"✓ Grafik kaydedildi: {self.output_dir / 'model_comparison_charts.png'}")
        plt.close()
    
    def generate_recommendation(self):
        """
        En iyi algoritmayı öner
        """
        print("\n" + "=" * 80)
        print("ALGORİTMA ÖNERİSİ")
        print("=" * 80)
        
        if not self.results:
            print("⚠ Henüz sonuç eklenmemiş")
            return []
        
        # Overall F1 score'a göre sırala
        rankings = []
        for algo, results in self.results.items():
            overall_f1 = results['test_metrics']['overall']['f1_weighted']
            overall_acc = results['test_metrics']['overall']['accuracy']
            rankings.append((algo, overall_f1, overall_acc))
        
        rankings.sort(key=lambda x: x[1], reverse=True)
        
        print("\nGenel Sıralama (F1 Score'a göre):")
        for idx, (algo, f1, acc) in enumerate(rankings, 1):
            print(f"{idx}. {algo}: F1={f1:.4f}, Accuracy={acc:.4f}")
        
        best_algo = rankings[0][0]
        print(f"\n🏆 EN İYİ ALGORİTMA: {best_algo}")
        print(f"   F1 Score: {rankings[0][1]:.4f}")
        print(f"   Accuracy: {rankings[0][2]:.4f}")
        
        # Hedef bazlı en iyi
        print("\nHedef Değişken Bazlı En İyiler:")
        for target in ['seller_number', 'customer_number', 'main_account']:
            best_f1 = 0
            best_algo_target = None
            
            for algo in self.results.keys():
                f1 = self.results[algo]['test_metrics'][target]['f1_weighted']
                if f1 > best_f1:
                    best_f1 = f1
                    best_algo_target = algo
            
            print(f"  {target}: {best_algo_target} (F1={best_f1:.4f})")
        
        return rankings
    
    def export_report(self):
        """
        Karşılaştırma raporunu dışa aktar
        """
        report_lines = []
        report_lines.append("# MODEL KARŞILAŞTIRMA RAPORU\n\n")
        report_lines.append(f"Tarih: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        report_lines.append("## Karşılaştırılan Algoritmalar\n\n")
        for algo in self.results.keys():
            report_lines.append(f"- {algo}\n")
        report_lines.append("\n")
        
        report_lines.append("## Performans Özeti\n\n")
        report_lines.append("| Algorithm | Overall Accuracy | Overall F1 |\n")
        report_lines.append("|-----------|-----------------|------------|\n")
        
        for algo, results in self.results.items():
            overall = results['test_metrics']['overall']
            acc = overall['accuracy']
            f1 = overall['f1_weighted']
            report_lines.append(f"| {algo} | {acc:.4f} | {f1:.4f} |\n")
        
        report_lines.append("\n## Oluşturulan Dosyalar\n\n")
        report_lines.append("- model_comparison_detailed.csv\n")
        report_lines.append("- model_comparison_overall.csv\n")
        report_lines.append("- model_comparison_charts.png\n")
        
        with open(self.output_dir / 'comparison_report.md', 'w', encoding='utf-8') as f:
            f.writelines(report_lines)
        
        print(f"\n✓ Rapor kaydedildi: {self.output_dir / 'comparison_report.md'}")


# Kullanım örneği
if __name__ == "__main__":
    comparator = ModelComparator()
    
    # Manuel sonuç ekleme örneği
    # comparator.add_result('random_forest', {...})
    
    # Veya dosyadan yükleme
    algorithms = ['random_forest', 'gradient_boosting', 'logistic_regression', 'neural_network']
    comparator.load_from_file(algorithms)
    
    # Analiz
    if comparator.results:
        comparator.create_comparison_table()
        comparator.plot_comparison()
        comparator.generate_recommendation()
        comparator.export_report()