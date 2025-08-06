'''
Created on 9 Nis 2025

@author: zehra
'''
import pandas as pd

df = pd.read_csv("C:/Users/zehra/Downloads/archive (6)/stroke_risk_dataset_v2.csv")

# print(df.info())  # Veri türleri, eksik değerler

import matplotlib.pyplot as plt
import seaborn as sns

# Yaş aralıklarını belirliyoruz
bins = [17, 30, 40, 50, 60, 70, 80, 90]
labels = ['18–30', '31–40', '41–50', '51–60', '61–70', '71–80', '81–90']
df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels)

# Her yaş grubundaki stroke oranı
stroke_risk_by_age = df.groupby('age_group')['at_risk'].mean() * 100

# Her yaş grubundaki ortalama stroke yüzdesi
avg_stroke_percentage_by_age = df.groupby('age_group')['stroke_risk_percentage'].mean()

# Grafik çizimi
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.barplot(x=stroke_risk_by_age.index, y=stroke_risk_by_age.values, palette='Reds')
plt.title('Yaş Aralığına Göre İnme Olasılığı (%)')
plt.ylabel('İnme Olasılığı (%)')
plt.xlabel('Yaş Aralığı')
plt.ylim(0, 100)

plt.subplot(1, 2, 2)
sns.lineplot(x=avg_stroke_percentage_by_age.index, y=avg_stroke_percentage_by_age.values, marker='o', color='blue')
plt.title('Yaş Aralığına Göre Ortalama İnme Riski (%)')
plt.ylabel('Ortalama Risk (%)')
plt.xlabel('Yaş Aralığı')
plt.ylim(0, 100)

plt.tight_layout()
plt.show()

