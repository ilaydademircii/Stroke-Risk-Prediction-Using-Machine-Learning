'''
Created on 9 Nis 2025

@author: zehra
'''
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv("C:/Users/zehra/Downloads/archive (6)/stroke_risk_dataset_v2.csv")

# at_risk değişkeninin dağılımını hesaplayalım
at_risk_counts = df['at_risk'].value_counts()

# Pasta grafiği için ayarlar
plt.figure(figsize=(7, 7))

# Pasta grafiğini çizdirelim
plt.pie(at_risk_counts, labels=['Not At Risk (0)', 'At Risk (1)'], autopct='%1.1f%%', startangle=90, colors=['lightgreen', 'salmon'])

# Başlık ekleyelim
plt.title('Risk Durumu Dağılımı', fontsize=16, fontweight='bold', color='darkblue')

# Grafik gösterimi
plt.axis('equal')  # Pasta grafiklerinin yuvarlak görünmesi için
plt.show()
