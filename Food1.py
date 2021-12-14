import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import seaborn as sns
import pandas as pd 
from datetime import datetime

path1 = "/Users/justin/Downloads/foodlog.xlsx"
df = pd.read_excel(path1)
path2 = "/Users/justin/Downloads/Exerciselog.xlsx"
df2 = pd.read_excel(path2)
path3 = '/Users/justin/Downloads/weightlog.xlsx'
df3 = pd.read_excel(path3)

Y = df["Fecha"].dt.strftime('%m %d %Y')
X0 = df['Calorias (kcal)']
X1 = df['Carbohidratos (g)']
X2 = df['Lípidos (g)']
X3 = df['Proteína (g)']

Y2 = df2['Date'].dt.strftime('%m %d %Y')
Z1 = df2["Lifted in pounds"]

Y3 = df3['Date'].dt.strftime('%m %d %Y')
W1 = df3['Peso en kg']

sumatory = df.groupby(Y).sum()
print(sumatory.describe())
total= sumatory['Calorias (kcal)']
totalc = sumatory['Carbohidratos (g)']
totall =sumatory['Lípidos (g)']
sodio = sumatory['Sodio (mg)']

Y = list(dict.fromkeys(Y))

px = df.describe()
px = np.array(px)
meandia = px[1]
std = px[2]
mincomida = px[3]
q25 = px[4]
q50 =px[5]
q75 = px[6]
maxdia = px[7]

px1 = sumatory.describe()
px1 = np.array(px1)
meantodo= px1[1]
stdtodo = px1[2]
mintodo = px1[3]
q25todo = px1[4] 
q50todo =px1[5]
q75todo = px1[6]
maxtodo = px1[7]

labels = ['Carbohidratos (g)', 'Lípidos (g)', 'Proteína (g)']
item_means = [meandia[1], meandia[2], meandia[3]]
day_means = [meantodo[1], meantodo[2], meantodo[3]]

x = np.arange(len(labels)) 
width = 0.35  

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, item_means, width, label='Por alimento')
rects2 = ax.bar(x + width/2, day_means, width, label='Por dia')

ax.set_ylabel('Gramos')
ax.set_title('Promedio ')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

fig.tight_layout()


f1 = plt.figure()
f2 = plt.figure()
f3 = plt.figure()
ax1 = f1.add_subplot(111)
ax1.plot(Y, total, c="turquoise")
ax1.set_xlabel("Fecha")
ax1.set_ylabel('Calorias')
ax1.set_title("Calorias por dia")
ax1.set_xticklabels(Y, fontsize=2)

ax2 = f2.add_subplot(111)
ax2.plot(Y2, Z1, c="blueviolet")
ax2.set_xlabel("Fecha")
ax2.set_ylabel('Peso total en libras')
ax2.set_title("Entrenamiento")
ax2.set_xticklabels(Y2, fontsize=4)

ax3 = f3.add_subplot(111)
ax3.plot(Y3, W1, c='red')
ax3.set_xlabel("Fecha")
ax3.set_ylabel('Peso en Kg')
ax3.set_title("Peso corporal")
ax3.set_xticklabels(Y3, fontsize=4)

fig = plt.figure()
ax = fig.add_subplot(111,projection = '3d')
ax.scatter(total, totalc,totall, c='pink', marker = 'o', s = 30)

ax.set_xlabel('Calorias')
ax.set_ylabel('Carbohidratos')
ax.set_zlabel('Lipidos')
plt.xticks(fontsize=12)

c = "gold"
c2 = "darkorange" 
c3 = "goldenrod"
c4 = "brown"
fig, ax4 = plt.subplots()
ax4.boxplot(total, boxprops=dict(color=c),
            capprops=dict(color=c2),
            whiskerprops=dict(color=c2),
            flierprops=dict(color=c4, markeredgecolor=c4),
            medianprops=dict(color=c3),)
ax4.set_title('Dispersión de Calorias total por dia')

f4 = plt.figure()
ax4 = f4.add_subplot(111)
ax4.stem(Y, sodio, linefmt="lime" )
ax4.set_xlabel('Dia')
ax4.set_ylabel('Sodio en mg')
ax4.set_xticklabels(Y, fontsize=2)
ax4.set_title("Consumo de sodio por dia")

plt.show()

