import matplotlib.pyplot as plt
import pandas as pd

data = {
    "ESP8266 NodeMCU": [
        "D2 (GPIO4)",
        "D1 (GPIO5)",
        "3.3V",
        "GND",
        "5V",
        "D7 (GPIO13)",
        "D8 (GPIO15)"
    ],
    "MAX30102": ["SDA", "SCL", "VCC", "GND", "", "", ""],
    "MPU6050": ["SDA", "SCL", "VCC", "GND", "", "", ""],
    "MLX90614": ["SDA", "SCL", "VCC", "GND", "", "", ""],
    "GY-NEO6MV2": ["", "", "", "GND", "VCC", "TX", "RX"]
}

df = pd.DataFrame(data)

fig, ax = plt.subplots(figsize=(8, 3.5))  # Uygun boyutta çizim

ax.axis('off')

table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')

table.auto_set_font_size(False)
table.set_fontsize(9)

# Genişlik ayarı
n_rows, n_cols = df.shape
for i in range(n_rows + 1):
    for j in range(n_cols):
        table[(i, j)].set_width(1.0 / n_cols)

# Satır yüksekliği
for i in range(n_rows + 1):
    for j in range(n_cols):
        table[(i, j)].set_height(0.11)

# Renk ve çizgi
header_color = "#d9ead3"
row_colors = ["#f7f9f8", "#ffffff"]

for j in range(n_cols):
    table[(0, j)].set_facecolor(header_color)
    table[(0, j)].get_text().set_weight('bold')

for i in range(1, n_rows + 1):
    color = row_colors[i % 2]
    for j in range(n_cols):
        table[(i, j)].set_facecolor(color)

for key, cell in table.get_celld().items():
    cell.set_linewidth(0.5)

plt.title("ESP8266 - Sensör Pin Bağlantı Tablosu", fontsize=14, weight='bold', pad=15)
plt.tight_layout()
plt.show()
