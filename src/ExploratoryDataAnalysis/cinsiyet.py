'''
Created on 9 Nis 2025

@author: zehra
'''
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Veri setini yükleyin
df = pd.read_csv("C:/Users/zehra/Downloads/archive (6)/stroke_risk_dataset_v2.csv")
import matplotlib.pyplot as plt
import pandas as pd

# Cinsiyet dağılımını hesaplayalım
gender_counts = df['gender'].value_counts()

# Pasta grafiği için ayarlar
plt.figure(figsize=(7, 7))

# Pasta grafiğini çizdirelim
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=90, colors=['skyblue', 'orange'])

# Başlık ekleyelim
plt.title('Cinsiyet Dağılımı', fontsize=16, fontweight='bold', color='darkblue')

# Grafik gösterimi
plt.axis('equal')  # Pasta grafiklerinin yuvarlak görünmesi için
plt.show()
