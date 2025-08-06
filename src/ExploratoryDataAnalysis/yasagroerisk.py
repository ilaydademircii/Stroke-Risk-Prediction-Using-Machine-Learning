'''
Created on 9 Nis 2025

@author: zehra
'''
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
df = pd.read_csv("C:/Users/zehra/Downloads/archive (6)/stroke_risk_dataset_v2.csv")

# Yaş verisinin minimum ve maksimum değerlerini alalım
age_min = df['age'].min()
age_max = df['age'].max()

# Yaş gruplarını dinamik bir şekilde belirleyelim
# Bu örnekte 10 yaş aralığı oluşturuyoruz, ancak veriye göre bunu değiştirebilirsiniz
bin_count = 10

# Yaş gruplarını oluşturuyoruz
age_bins = pd.cut(df['age'], bins=bin_count, right=False, include_lowest=True)

# Her yaş grubu için ortalama inme riski ve ortalama olasılık hesaplayalım
age_group_risk = df.groupby(age_bins)['stroke_risk_percentage'].mean()
age_group_risk_percentage = df.groupby(age_bins)['at_risk'].mean()

# Yaş grubu etiketlerini tam sayılara dönüştürelim
age_group_risk.index = age_group_risk.index.astype(str)

# Grafik ayarları
plt.figure(figsize=(12, 6))

# Çift eksenli grafik: sol eksende ortalama inme riski, sağ eksende olasılık
fig, ax1 = plt.subplots(figsize=(12, 6))

# Ortalama inme riski
sns.barplot(x=age_group_risk.index, y=age_group_risk.values, palette='viridis', ax=ax1)
ax1.set_xlabel('Yaş Grubu', fontsize=12, color='black')
ax1.set_ylabel('Ortalama İnme Riski (%)', fontsize=12, color='black')
ax1.tick_params(axis='y', labelcolor='black')

# İkinci eksende olasılık
ax2 = ax1.twinx()
sns.lineplot(x=age_group_risk.index, y=age_group_risk_percentage.values, color='red', marker='o', ax=ax2)
ax2.set_ylabel('At Risk Olasılığı (%)', fontsize=12, color='red')
ax2.tick_params(axis='y', labelcolor='red')

# Başlık
plt.title('Yaş Gruplarına Göre İnme Riski ve Olasılığı', fontsize=16, fontweight='bold', color='darkblue', pad=20)

# Grafik gösterimi
plt.tight_layout()
plt.show()
