import pandas as pd
import matplotlib.pyplot as plt

# Veriyi yükle
df = pd.read_csv("C:/Users/zehra/Downloads/archive (6)/stroke_risk_dataset_v2.csv")

# 1️⃣ Düzensiz kalp atışı OLANLAR
afib_var = df[df['irregular_heartbeat'] == 1]
afib_var_risk = afib_var['at_risk'].value_counts().sort_index()

# 2️⃣ Düzensiz kalp atışı OLMAYANLAR
afib_yok = df[df['irregular_heartbeat'] == 0]
afib_yok_risk = afib_yok['at_risk'].value_counts().sort_index()

# Etiket ve renkler
etiketler = ['Risksiz ', 'Riskli ']
renkler = ['lightgray', 'lightblue']

# Grafik alanı
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# 1. Pasta: Kalp Atışı Düzensiz (1)
ax[0].pie(afib_var_risk, labels=etiketler, autopct='%1.1f%%', colors=renkler, startangle=140, wedgeprops={'edgecolor': 'black'})
ax[0].set_title("Irregular Heartbeat = 1\n(Düzensiz Kalp Atışı Olanlar)")

# 2. Pasta: Kalp Atışı Normal (0)
ax[1].pie(afib_yok_risk, labels=etiketler, autopct='%1.1f%%', colors=renkler, startangle=140, wedgeprops={'edgecolor': 'black'})
ax[1].set_title("Irregular Heartbeat = 0\n(Düzensiz Kalp Atışı Olmayanlar)")

plt.tight_layout()
plt.show()
