import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Veri setini yükleyin
df = pd.read_csv("C:/Users/zehra/Downloads/archive (6)/stroke_risk_dataset_v2.csv")

# Yaş aralıklarını belirleyin
bins = [18, 30, 40, 50, 60, 70, 80, 90]
labels = ['18-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90']
df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)

# Cinsiyet ve yaş grubu kombinasyonuna göre inme oranını hesaplayalım
age_gender_stats = df.groupby(['age_group', 'gender'])['at_risk'].agg(
    stroke_count='sum',
    total_count='size'
).reset_index()

# Inme oranını yüzdelik olarak hesaplayalım
age_gender_stats['stroke_rate'] = age_gender_stats['stroke_count'] / age_gender_stats['total_count'] * 100

# Grafik boyutlarını ayarlayın
plt.figure(figsize=(14, 7))

# Yaş grubu ve cinsiyete göre inme oranını gösterecek stacked bar plot çizelim
sns.barplot(x='age_group', y='stroke_rate', hue='gender', data=age_gender_stats, palette='coolwarm', ci=None)

# Başlık ve etiketleri ayarlayın
plt.title('Yaş Grubu ve Cinsiyete Göre İnme Oranı', fontsize=16, fontweight='bold', color='darkblue', pad=20)
plt.xlabel('Yaş Grubu', fontsize=12, color='black')
plt.ylabel('İnme Oranı (%)', fontsize=12, color='black')

# Ekstra stil ayarları
plt.xticks(rotation=45, fontsize=10)
plt.yticks(fontsize=10)

# Grafik gösterimi
plt.tight_layout()
plt.show()
'''
Created on 9 Nis 2025

@author: zehra
'''
