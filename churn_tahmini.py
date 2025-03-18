import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("churn_modelling.csv")  # Dosya adını doğru gir!

df = df.drop(columns=['RowNumber', 'CustomerId', 'Surname'])
# Kategorik Değişkenleri Sayısala Çevirme
df = pd.get_dummies(df, columns=['Geography', 'Gender'], drop_first=True)
X = df.drop(columns=['Exited'])
y = df['Exited']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

classifier = DecisionTreeClassifier(max_depth=5, min_samples_split=5, min_samples_leaf=2, random_state=42)
classifier.fit(X_train, y_train)

# Doğruluk Değerleri
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Doğruluk Oranı: {accuracy:.2f}")
print(classification_report(y_test, y_pred))

# Kullanıcıdan Veri Alma
print("\nYeni Müşteri Bilgilerini Giriniz:")
credit_score = int(input("Kredi Skoru: "))
age = int(input("Yaş: "))
tenure = int(input("Müşteri Kıdemi (Yıl): "))
balance = float(input("Bakiye: "))
num_of_products = int(input("Ürün Sayısı: "))
has_credit_card = int(input("Kredi Kartı Var mı? (1: Evet, 0: Hayır): "))
is_active_member = int(input("Aktif Üye mi? (1: Evet, 0: Hayır): "))
estimated_salary = float(input("Tahmini Maaş: "))
geography = input("Ülke (France/Germany/Spain): ").strip().capitalize()
gender = input("Cinsiyet (Male/Female): ").strip().capitalize()


yeni_musteri = pd.DataFrame([[credit_score, age, tenure, balance, num_of_products, 
                               has_credit_card, is_active_member, estimated_salary,
                               1 if geography == "Germany" else 0,  
                               1 if geography == "Spain" else 0,   
                               1 if gender == "Male" else 0]],       
                            columns=X.columns)

# tahmini ve ölçeklendirme
yeni_musteri = scaler.transform(yeni_musteri)

tahmin = classifier.predict(yeni_musteri)[0]
sonuc = "Bu müşteri büyük ihtimalle aboneliği bırakmayacak." if tahmin == 0 else "Bu müşteri büyük ihtimalle aboneliği bırakacak!"
print(f"\nTahmin: {sonuc}")

# Görselliştirme(kullanıcı seçimi)
def grafik_menu():
    print("\nHangi grafiği görmek istersiniz?")
    print("1. Yaş Dağılımı ve Churn")
    print("2. Bakiye ve Churn")
    print("3. Tahmini Maaş ve Churn")
    print("4. Kredi Skoru ve Churn")
    print("0. Çıkış")

while True:
    grafik_menu()
    try:
        secim = int(input("Seçiminizi yapın (0-4): "))
        if secim == 1:
            plt.figure(figsize=(10, 6))
            sns.histplot(data=df, x='Age', hue='Exited', multiple='stack', kde=True)
            plt.title('Yaş Dağılımı ve Churn')
            plt.show()
        elif secim == 2:
            plt.figure(figsize=(10, 6))
            sns.histplot(data=df, x='Balance', hue='Exited', multiple='stack', kde=True)
            plt.title('Bakiye ve Churn')
            plt.show()
        elif secim == 3:
            plt.figure(figsize=(10, 6))
            sns.histplot(data=df, x='EstimatedSalary', hue='Exited', multiple='stack', kde=True)
            plt.title('Tahmini Maaş ve Churn')
            plt.show()
        elif secim == 4:
            plt.figure(figsize=(10, 6))
            sns.histplot(data=df, x='CreditScore', hue='Exited', multiple='stack', kde=True)
            plt.title('Kredi Skoru ve Churn')
            plt.show()
        elif secim == 0:
            print("Programdan çıkılıyor...")
            break
        else:
            print("Geçersiz seçim! Lütfen 0-4 arasında bir değer girin.")
    except ValueError:
        print("Lütfen geçerli bir sayı girin!")
