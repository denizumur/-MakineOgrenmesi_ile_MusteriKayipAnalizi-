import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Veriyi yükleme
df = pd.read_csv("churn_modelling.csv")
df.drop(["RowNumber", "CustomerId", "Surname"], axis=1, inplace=True)
le = LabelEncoder()
df["Geography"] = le.fit_transform(df["Geography"])
df["Gender"] = le.fit_transform(df["Gender"])
X = df.drop("Exited", axis=1)
y = df["Exited"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

models = {
    "Karar Ağacı": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
    "LightGBM": LGBMClassifier(random_state=42)
}

# Modelleri eğitme ve karşılaştırma
best_model = None
best_score = 0
scores = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    scores[name] = score
    
    if score > best_score:
        best_score = score
        best_model = model

print("\n **Model Performans Karşılaştırması:**")
for model_name, score in scores.items():
    print(f"{model_name}: {score:.4f}")

print(f"\n✅ En iyi model: {best_model.__class__.__name__} ({best_score:.4f} doğruluk)")

# Kullanıcıdan müşteri bilgisi al
print("\nLütfen tahmin için aşağıdaki bilgileri giriniz:")
geography = int(input("Ülke (0: Fransa, 1: Almanya, 2: İspanya): "))
gender = int(input("Cinsiyet (0: Kadın, 1: Erkek): "))
credit_score = int(input("Kredi Skoru: "))
age = int(input("Yaş: "))
tenure = int(input("Çalışma Süresi: "))
balance = float(input("Bakiye: "))
num_of_products = int(input("Ürün Sayısı: "))
has_cr_card = int(input("Kredi Kartı Var mı? (0: Hayır, 1: Evet): "))
is_active_member = int(input("Aktif Müşteri mi? (0: Hayır, 1: Evet): "))
estimated_salary = float(input("Tahmini Maaş: "))

yeni_musteri = np.array([[geography, gender, credit_score, age, tenure, balance, num_of_products, has_cr_card, is_active_member, estimated_salary]])
yeni_musteri = scaler.transform(yeni_musteri)

# En iyi model ile tahmin yapma
tahmin = best_model.predict(yeni_musteri)
sonuc = "Müşteri kaybedilecek" if tahmin[0] == 1 else "Müşteri kalacak"

print(f"\n🔍 Tahmin Sonucu: {sonuc}")


def grafik_menu():
    print("\nHangi grafiği görmek istersiniz?")
    print("1. Yaş Dağılımı ve Churn")
    print("2. Bakiye ve Churn")
    print("3. Tahmini Maaş ve Churn")
    print("4. Kredi Skoru ve Churn")
    print("0. Çıkış")

# Grafik çizme
while True:
    grafik_menu()
    try:
        secim = int(input("Seçiminizi yapın (0-4): "))
        
        if secim == 1:
            plt.figure(figsize=(10, 6))
            sns.histplot(data=df, x='Age', hue='Exited', multiple='stack', kde=True)
            plt.title('Yaş Dağılımı ve Churn')
            plt.xlabel('Yaş')
            plt.ylabel('Müşteri Sayısı')
            plt.show()
        elif secim == 2:
            plt.figure(figsize=(10, 6))
            sns.histplot(data=df, x='Balance', hue='Exited', multiple='stack', kde=True)
            plt.title('Bakiye ve Churn')
            plt.xlabel('Bakiye')
            plt.ylabel('Müşteri Sayısı')
            plt.show()
        elif secim == 3:
            plt.figure(figsize=(10, 6))
            sns.histplot(data=df, x='EstimatedSalary', hue='Exited', multiple='stack', kde=True)
            plt.title('Tahmini Maaş ve Churn')
            plt.xlabel('Tahmini Maaş')
            plt.ylabel('Müşteri Sayısı')
            plt.show()
        elif secim == 4:
            plt.figure(figsize=(10, 6))
            sns.histplot(data=df, x='CreditScore', hue='Exited', multiple='stack', kde=True)
            plt.title('Kredi Skoru ve Churn')
            plt.xlabel('Kredi Skoru')
            plt.ylabel('Müşteri Sayısı')
            plt.show()
        elif secim == 0:
            print("Programdan çıkılıyor...")
            break
        else:
            print("Geçersiz seçim! Lütfen 0-4 arasında bir sayı girin.")
    except ValueError:
        print("Lütfen geçerli bir sayı girin!")
