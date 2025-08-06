'''
Created on 9 Nis 2025

@author: zehra
'''
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
df = pd.read_csv("C:/Users/zehra/Downloads/archive (6)/stroke_risk_dataset_v2.csv")
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Veriyi yükle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Veriyi yükle

# Kadınları filtreleyelim
female_data = df[df['gender'] == 'Female']

# Yaş aralıklarını dinamik olarak belirleyelim
# 10 yaş aralıklarıyla yaş gruplarını oluşturuyoruz
bin_count = 10
age_min = female_data['age'].min()
age_max = female_data['age'].max()

# Yaş gruplarını oluşturuyoruz
age_bins = pd.cut(female_data['age'], bins=bin_count, right=False, include_lowest=True)

# Her yaş grubu için ortalama inme riski hesaplıyoruz
age_group_risk = female_data.groupby(age_bins)['stroke_risk_percentage'].mean()

# Yaş grubu etiketlerini tam sayılara dönüştürme
age_group_risk.index = age_group_risk.index.astype(str)

# Grafik
plt.figure(figsize=(12, 8))
sns.lineplot(x=age_group_risk.index, y=age_group_risk.values, color='purple')
plt.title("Kadınlarda Yaşa Göre İnme Riski Oranı")
plt.xlabel("Yaş Grupları")
plt.ylabel("Ortalama İnme Riski (%)")
plt.xticks(rotation=45)
plt.grid(True)
plt.show()
