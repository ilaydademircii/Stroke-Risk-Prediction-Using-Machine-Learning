'''
Created on 9 Nis 2025

@author: zehra
'''
import pandas as pd

df = pd.read_csv("C:/Users/zehra/Downloads/archive (6)/stroke_risk_dataset_v2.csv")

# print(df.info())  # Veri türleri, eksik değerler

import matplotlib.pyplot as plt
import seaborn as sns

# Yaş verisinin minimum ve maksimum değerlerini alalım
age_min = df['age'].min()
age_max = df['age'].max()

# Yaş gruplarını doğal şekilde belirlemek için uygun bin sayısı seçelim
bin_count = 8  # Örneğin 10 grup kullanacağız

# Yaş aralıklarını tam sayılarla oluşturalım
age_bins = pd.cut(df['age'], bins=bin_count, right=False, include_lowest=True)

# Her yaş grubu için ortalama inme riskini hesaplayalım
age_group_risk = df.groupby(age_bins)['stroke_risk_percentage'].mean()

# Yaş grubu etiketlerini tam sayılara dönüştürelim
age_group_risk.index = age_group_risk.index.astype(str)

# Grafik ayarları
plt.figure(figsize=(10, 6))

# Çubuk grafik çizdirelim
sns.barplot(x=age_group_risk.index, y=age_group_risk.values, palette='viridis')

# Başlık ve etiketler
plt.title('Yaş Gruplarına Göre Ortalama İnme Riski', fontsize=16, fontweight='bold', color='darkblue', pad=20)
plt.xlabel('Yaş Grubu', fontsize=12, color='black')
plt.ylabel('Ortalama İnme Riski (%)', fontsize=12, color='black')

# Ekstra stil ayarları
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)

# Grafik gösterimi
plt.tight_layout()
plt.show()

