'''
Created on 9 Nis 2025

@author: zehra
'''

import pandas as pd

df = pd.read_csv("C:/Users/zehra/Downloads/archive (6)/stroke_risk_dataset_v2.csv")
import pandas as pd
import matplotlib.pyplot as plt

# Veriyi yükle

# 'at_risk' değişkeninin dağılımını hesaplayalım
at_risk_distribution = df['at_risk'].value_counts()

# Pasta grafiği ile gösterelim
plt.figure(figsize=(6, 6))
at_risk_distribution.plot(kind='pie', autopct='%1.1f%%', labels=['Not at Risk', 'At Risk'], colors=['#ff9999', '#66b3ff'])
plt.title('İ')
plt.ylabel('')  # Y-eksenini gizleyelim
plt.show()
