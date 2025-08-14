# 📞 Telekomünikasyon Müşteri Kaybı (Churn) Tahmin Projesi

Bu proje, telekomünikasyon şirketlerinin müşteri kaybını (churn) tahmin etmek için makine öğrenmesi ve derin öğrenme tekniklerini kullanarak kapsamlı bir analiz sunmaktadır.

## 🎯 Proje Amacı

Telekomünikasyon sektöründe müşteri kaybı, şirketlerin kar marjlarını doğrudan etkileyen kritik bir faktördür. Bu proje, müşteri davranışlarını analiz ederek hangi müşterilerin hizmeti bırakma olasılığının yüksek olduğunu tahmin etmeyi amaçlamaktadır.

## 📊 Veri Seti

**Veri Seti:** `WA_Fn-UseC_-Telco-Customer-Churn.csv` (https://www.kaggle.com/datasets/jazidesigns/telecom-dataset)

**Veri Boyutu:** 7,045 müşteri kaydı, 21 özellik

### Özellikler:
- **Demografik Bilgiler:** Gender, SeniorCitizen, Partner, Dependents
- **Hizmet Bilgileri:** PhoneService, MultipleLines, InternetService
- **Ek Hizmetler:** OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies
- **Sözleşme Bilgileri:** Contract, PaperlessBilling, PaymentMethod
- **Finansal Bilgiler:** MonthlyCharges, TotalCharges
- **Müşteri Geçmişi:** Tenure (müşteri olma süresi)
- **Hedef Değişken:** Churn (Müşteri Kaybı)

## 🛠️ Kullanılan Teknolojiler

### Makine Öğrenmesi Kütüphaneleri:
- **Scikit-learn:** Geleneksel ML algoritmaları
- **XGBoost:** Gradient boosting
- **LightGBM:** Light gradient boosting
- **CatBoost:** Categorical boosting

### Derin Öğrenme:
- **TensorFlow/Keras:** Neural network modelleri
- **Keras Tuner:** Hiperparametre optimizasyonu

### Veri Analizi ve Görselleştirme:
- **Pandas:** Veri manipülasyonu
- **NumPy:** Sayısal işlemler
- **Matplotlib & Seaborn:** Görselleştirme

## 📈 Proje Yapısı

```
project-telco/
├── TELCO_eda_ml_dl.ipynb          # Ana analiz notebook'u
├── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Veri seti
├── base_model.keras               # Temel derin öğrenme modeli
├── best_random_model.keras        # En iyi random search modeli
├── dl_final.keras                 # Final derin öğrenme modeli
├── best_random_hp.joblib          # En iyi hiperparametreler
├── rfc_final.png                  # Random Forest özellik önem grafiği
```

## 🔍 Analiz Süreci

### 1. Veri Keşfi ve Temizleme (EDA)
- Veri setinin genel yapısının incelenmesi
- Eksik değerlerin tespiti ve işlenmesi
- Kategorik değişkenlerin analizi
- Sayısal değişkenlerin dağılımının incelenmesi
- Hedef değişken (Churn) dağılımının analizi

### 2. Veri Ön İşleme
- Kategorik değişkenlerin encoding işlemi
- Sayısal değişkenlerin normalizasyonu
- Özellik ölçeklendirme (MinMaxScaler, StandardScaler, RobustScaler)
- Train-test split (80-20 oranında)

### 3. Makine Öğrenmesi Modelleri

#### Geleneksel ML Algoritmaları:
- **Logistic Regression:** Temel sınıflandırma modeli
- **K-Nearest Neighbors (KNN):** Mesafe tabanlı sınıflandırma
- **Decision Tree:** Ağaç tabanlı sınıflandırma
- **Random Forest:** Ensemble ağaç modeli
- **Gradient Boosting:** Gradient boosting ensemble

#### Gelişmiş ML Algoritmaları:
- **XGBoost:** Extreme gradient boosting
- **LightGBM:** Light gradient boosting machine
- **CatBoost:** Categorical boosting

### 4. Derin Öğrenme Modeli
- **Neural Network:** Çok katmanlı perceptron
- **Hiperparametre Optimizasyonu:** Keras Tuner ile otomatik arama
- **Regularization:** Dropout, BatchNormalization, L2 regularization
- **Early Stopping:** Overfitting'i önleme

### 5. Model Değerlendirme
- **Accuracy:** Doğruluk oranı
- **Precision:** Kesinlik
- **Recall:** Duyarlılık
- **F1-Score:** Harmonik ortalama
- **ROC-AUC:** ROC eğrisi altındaki alan
- **Confusion Matrix:** Karışıklık matrisi

## 🏆 Model Performansları

### En İyi Performans Gösteren Modeller:

1. **CatBoost:** En yüksek genel performans
2. **XGBoost:** Yüksek doğruluk ve dengeli metrikler
3. **LightGBM:** Hızlı eğitim ve iyi performans
4. **Random Forest:** Kararlı ve yorumlanabilir sonuçlar
5. **Deep Learning:** Karmaşık pattern'leri yakalama

### Model Karşılaştırması:
- Tüm modeller cross-validation ile değerlendirilmiştir
- Hiperparametre optimizasyonu GridSearch ve RandomSearch ile yapılmıştır
- Ensemble yöntemler genellikle daha iyi performans göstermiştir

## 📊 Önemli Bulgular

### Müşteri Kaybını Etkileyen Faktörler:
1. **Sözleşme Türü:** Aylık sözleşmeler daha yüksek churn oranı
2. **İnternet Hizmeti:** Fiber optik müşterilerde daha yüksek kayıp
3. **Ödeme Yöntemi:** Elektronik çek ile ödeme yapanlarda daha yüksek churn
4. **Müşteri Süresi:** Yeni müşterilerde daha yüksek kayıp riski
5. **Ek Hizmetler:** Online güvenlik ve teknik destek almayan müşterilerde daha yüksek risk

### İş Önerileri:
- Aylık sözleşmeli müşterilere özel kampanyalar
- Fiber optik hizmet kalitesinin iyileştirilmesi
- Online güvenlik hizmetlerinin teşvik edilmesi
- Yeni müşteri deneyiminin iyileştirilmesi

## 🚀 Kullanım

### Sistem Gereksinimleri:
- **Python:** 3.9.2
- **İşletim Sistemi:** Windows 10/11, macOS, Linux

### Gereksinimler:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
pip install xgboost lightgbm catboost
pip install tensorflow keras-tuner
```

### Çalıştırma:
1. Jupyter Notebook'u başlatın
2. `TELCO_eda_ml_dl.ipynb` dosyasını açın
3. Tüm hücreleri sırayla çalıştırın

### Model Yükleme:
```python
# Makine öğrenmesi modeli
from joblib import load
model = load('best_random_hp.joblib')

# Derin öğrenme modeli
from keras.models import load_model
dl_model = load_model('dl_final.keras')
```

