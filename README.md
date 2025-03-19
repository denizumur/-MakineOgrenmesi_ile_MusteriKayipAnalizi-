#  Müşteri Churn (Abone Kaybı) Tahmini

Bu proje, bir **bankacılık veri seti** kullanarak **hangi müşterilerin bankayı terk etme olasılığının yüksek olduğunu tahmin eden** bir **Makine Öğrenimi Modeli** geliştirir.  
**Karar Ağacı, Random Forest, XGBoost ve LightGBM** gibi popüler **sınıflandırma algoritmalarını karşılaştırarak en iyi modeli seçer ve tahminlerde bu modeli kullanır**.  

Ayrıca, **müşteri churn ile yaş, kredi skoru, bakiye ve tahmini maaş arasındaki ilişkileri görselleştiren interaktif grafikler** içerir.  

---

##  Proje İçeriği  

✔ **Veri Analizi ve Ön İşleme**  
- **Eksik ve gereksiz sütunlar temizlendi** (RowNumber, CustomerID, Surname kaldırıldı).  
- **Kategorik değişkenler sayısala çevrildi** (Cinsiyet & Ülke kodlandı).  
- **Veri ölçeklendirildi** (StandardScaler ile).  

✔ **Model Eğitimi ve Karşılaştırma**  
- **Karar Ağacı, Random Forest, XGBoost, LightGBM** kullanılarak **karşılaştırmalı bir eğitim yapıldı**.  
- **Modellerin doğruluk oranları hesaplandı** ve **en iyi model otomatik seçildi**.  

✔ **Gerçek Zamanlı Tahmin**  
- Kullanıcıdan **müşteri bilgileri alınıyor** ve **en iyi modelle tahmin yapılıyor**.  

✔ **Veri Görselleştirme**  
- **Yaş - Churn ilişkisi**  
- **Bakiye - Churn ilişkisi**  
- **Kredi Skoru - Churn ilişkisi**  
- **Maaş - Churn ilişkisi**  

---

##  Kullanılan Teknolojiler ve Kütüphaneler  
| Teknoloji | Açıklama |
|-----------|---------|
| Python | Ana programlama dili |
| Pandas | Veri işleme ve analiz |
| NumPy | Sayısal işlemler |
| Matplotlib | Grafik çizimi |
| Seaborn | Gelişmiş veri görselleştirme |
| Scikit-learn | Makine öğrenimi modelleri |
| XGBoost | Gradient boosting modeli |
| LightGBM | Hafif ve hızlı boosting modeli |

---

##  Gereksinimler ve Kurulum  

Projeyi çalıştırmak için aşağıdaki adımları takip edin:  

### 1 Gerekli Kütüphaneleri Yükleyin  
Aşağıdaki komutu terminalde çalıştırın:  
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm

## Referanslar

Bu projede kullanılan veri seti Kaggle üzerinden alınmıştır. Veri setine aşağıdaki bağlantıdan ulaşabilirsiniz:

- [Kaggle - Churn Modelling Data](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling)