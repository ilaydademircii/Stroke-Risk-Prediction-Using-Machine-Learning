import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
df = pd.read_csv("C:/Users/zehra/Downloads/archive (6)/stroke_risk_dataset_v2.csv")

# Yaş verisinin minimum ve maksimum değerlerini alalım
age_min = df['age'].min()
age_max = df['age'].max()

# Yaş gruplarını dinamik bir şekilde belirleyelim
# Bu örnekte 10 yaş aralığı oluşturuyoruz, ancak veriye göre bunu değiştirebilirsiniz
bin_count = 4

# Yaş gruplarını oluşturuyoruz
age_bins = pd.cut(df['age'], bins=bin_count, right=False, include_lowest=True)

# Her yaş grubu için yüksek tansiyon oranını hesaplayalım
age_group_high_blood_pressure = df.groupby(age_bins)['chest_discomfort'].mean() * 100

# Yaş grubu etiketlerini tam sayılara dönüştürelim
age_group_high_blood_pressure.index = age_group_high_blood_pressure.index.astype(str)

# Grafik ayarları
plt.figure(figsize=(12, 6))

# Yüksek tansiyon oranı
sns.barplot(x=age_group_high_blood_pressure.index, y=age_group_high_blood_pressure.values, palette='Reds', color='skyblue')

# Grafik etiketleri
plt.xlabel('Yaş Grubu', fontsize=12)
plt.ylabel('Göğüste Rahatsızlık Hissi Oranı (%)', fontsize=12)

# Başlık
plt.title('Yaş Gruplarına Göre Göğüste Rahatsızlık Hissi Oranı', fontsize=16, fontweight='bold', color='darkred')

# Grafik gösterimi
plt.tight_layout()
plt.show()
'''
Created on 9 Nis 2025

@author: zehra
'''
