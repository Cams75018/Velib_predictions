#!/usr/bin/env python
# coding: utf-8

# In[21]:


#pip install tensorflow


# In[22]:
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow import keras


# In[23]:


from keras.models import load_model
model1=load_model("modelLSTM.h5")


# In[ ]:


import tkinter as tk
import tkinter.ttk as ttk
class Inputbox:
	def __init__(self, text=""):
		"""An inputbox made by you
		example:
		inp = Inputbox(text="entrer l'heure de départ (format HHhMM)")
		print(inp.get)
		"""
		self.root = tk.Tk()
		self.get = ""
		self.root.title("Etape 1")
		self.root["bg"] = "black"
		style = ttk.Style()
		style2 = ttk.Style()
		style.configure("BW.TLabel", foreground="brown", background="white")
#		style2.configure("EntryStyle.TEntry",
#							foreground="blue",
#							background=[("active","red")],
#							fieldbackground="red")
 
		estyle = ttk.Style()
		estyle.element_create("plain.field", "from", "clam")
		estyle.layout("EntryStyle.TEntry",
	                   [('Entry.plain.field',
	                   	{'children':
	                   			[('Entry.background', {'children':
	                   				[('Entry.padding', {'children':
	                   					[('Entry.textarea', {'sticky': 'nswe'})],
	                      		'sticky': 'nswe'})],
	                      		'sticky': 'nswe'})],
	                      'border':'2',
	                      'sticky': 'nswe'})])
		estyle.configure("EntryStyle.TEntry",
	                 background="green", 
	                 foreground="blue",
	                 fieldbackground="white")
		self.label = ttk.Label(self.root, text = text, font="Arial 20", style="BW.TLabel")
 
		self.label.pack(fill=tk.BOTH, expand=1)
 
		self.entry = ttk.Entry(self.root, font="Arial 20", style="EntryStyle.TEntry")
 
		self.entry.pack(fill=tk.BOTH, padx=10, pady=10)
		self.entry.focus()
		self.entry.bind("<Return>", lambda x: self.getinput())
		self.root.mainloop()
 
	def getinput(self):
		self.get = self.entry.get()
		self.root.destroy()
 
 
inp = Inputbox(text="entrer l'heure de départ (format HHhMM)")

heure = inp.get

# In[ ]:





# In[24]:


#heure = input("entrer l'heure de départ (format HHhMM)")
heure = heure.upper()


# In[25]:


heur=int(heure.split("H")[0])
minu=int(heure.split("H")[1])
heur, minu


# In[26]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
pd.set_option('display.float_format', lambda x: '%.4f' % x)
import seaborn as sns
sns.set_context("paper", font_scale=1.3)
sns.set_style('white')
import warnings
warnings.filterwarnings('ignore')
from time import time
import matplotlib.ticker as tkr
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from sklearn import preprocessing
from statsmodels.tsa.stattools import pacf
import datetime


# In[ ]:





# In[27]:


import timeit
start_time = timeit.default_timer()


# In[28]:


heurefin=9   # heure de test de la journée et donc cut du dataset au dernier jour à cette heure
minute=15
#heuredebut=datetime.datetime(2020,1,28,15,30,0)
heuredebut=datetime.datetime(2020,1,28,heur,minu,0)
nsteps_in, nsteps_out = 30, 30
nsteps_in2=60
nsteps_in3=90
epok =5


# In[29]:



df=pd.read_csv("https://raw.githubusercontent.com/alexisgcomte/velib-prediction/master/3.%20Modeling%20Research/datasets/madeleine_weekday.csv")
#print('Number of rows and columns:', df.shape)
df.head(5)


# In[30]:


df['datedate'] = pd.to_datetime(df['date'])


# In[31]:


# on repère l'index de la dernière minutes à prédire
lastindex=df[df.datedate==heuredebut].index.max()
dfx=df.iloc[0:lastindex+1,:]
dfx.tail()


# In[ ]:





# In[32]:


#dfx=df[(df.hour==heurefin-1) | (df.hour== heurefin)]
#lastindex=dfx.index.max()
for i in range(lastindex,lastindex +1+ nsteps_out,1):
  dfx.loc[i]=df.iloc[i,:]
dfx.reset_index(inplace=True)


# In[33]:


df2=dfx.loc[:dfx.index.max(),['date','availiable_docks']]
#df3.sort_values('datenew', inplace=True, ascending=True)
#df3 = df3.reset_index(drop=True)
#print('Number of rows and columns:',dfx.shape)
#print(df2.date.min())
#print(df2.date.max())
df2.columns=["datenew","TOTALFD"]


# In[34]:


df3=df2.loc[:len(df2)-1-nsteps_out,["datenew","TOTALFD"]]
df3.tail()


# In[35]:


#df4=df3.loc[:,['datenew','TOTALFD']]
#df5=df3.loc[:,['datenew','TOTALFD']]
#df4.set_index('datenew',inplace=True)
#df4.plot(figsize=(12,5))
#plt.ylabel('TOTALFD')
#plt.legend().set_visible(False)
#plt.tight_layout()
#plt.title('TOTALFD Time Series')
#sns.despine(top=True)
#plt.show();


# In[36]:


from sklearn.preprocessing import MinMaxScaler


# In[37]:


from numpy import array
scaler = MinMaxScaler(feature_range=(-1, 1))
raw_seq = df3.TOTALFD.to_numpy()
#normalisation
raw_seq_scaled = scaler.fit_transform(raw_seq.reshape(-1,1))


# In[38]:



heurefin=9   # heure de test de la journée et donc cut du dataset au dernier jour à cette heure
minute=15
heuredebutt=heuredebut #datetime.datetime(2020,1,29,8,30,0)
nsteps_in, nsteps_out = 30, 30
nsteps_in2= 60
nsteps_in3= 90
epok =2

n_steps_in=nsteps_in2
n_steps_out= nsteps_out

lastindexx=df[df.datedate==heuredebutt].index.max()
dfxx=df.iloc[0:lastindexx+1,:]
dfxx.tail()

for i in range(lastindexx,lastindexx +1+ nsteps_out,1):
  dfxx.loc[i]=df.iloc[i,:]
dfxx.reset_index(inplace=True)

df22=dfxx.loc[:dfxx.index.max(),['date','availiable_docks']]
#df3.sort_values('datenew', inplace=True, ascending=True)
#df3 = df3.reset_index(drop=True)
#print('Number of rows and columns:',dfxx.shape)
#print(df22.date.min())
#print(df22.date.max())
df22.columns=["datenew","TOTALFD"]

df33=df22.loc[:len(df22)-1-nsteps_out,["datenew","TOTALFD"]]
df33.tail()

#scaler = MinMaxScaler(feature_range=(-1, 1))
n_features=1

raw_seq = df33.TOTALFD.to_numpy()
X_input_scal=scaler.transform(array(raw_seq[-n_steps_in:]).reshape(-1,1))
X_input_scal.reshape(X_input_scal.shape[0],)
X_input_scal = X_input_scal.reshape((1, n_steps_in, n_features))
#x_input = x_input.reshape((1, n_steps_in, n_features))
yhatt = model1.predict(X_input_scal, verbose=0)
yhatt=scaler.inverse_transform(yhatt.reshape(yhatt.shape[0],yhatt.shape[1]))


dfress=df22.tail(n_steps_out*3)
dfress["pred"]=0.0
dfress = dfress.reset_index()
#lastindex=dfres.tail(1).index.start
yhat22=pd.DataFrame(yhatt.reshape(n_steps_out,1))
for ind in range(len(dfress)-1, -len(yhat22) + len(dfress)-1 , -1):
   dfress["pred"][ind]=yhat22.iloc[ind-len(dfress)]

from math import sqrt
from sklearn.metrics import mean_squared_error

def measure_rmse(actual, predicted):
    return sqrt(mean_squared_error(actual, predicted))
msedff=dfress.tail(n_steps_out)
heurefin=heuredebutt+datetime.timedelta(minutes=nsteps_out)
#print("RMSE 3 ",measure_rmse(msedff.TOTALFD,msedff.pred))
#print("RMSE 3 prédictions arrondies ", measure_rmse(msedff.TOTALFD,round(msedff.pred)))
#print("nb epoch", epok)
#print("nb stepIN", n_steps_in)
#print("nb stepOUT", nsteps_out)
#print("prédiction de ",heuredebutt.hour,"h",heuredebutt.minute," à " ,heurefin.hour,"h",heurefin.minute,"min")
#print("temps d'execution " , int(elapsed3/60), "min", int((round(elapsed3/60,2,)-(int(elapsed3/60)))*60), "s")

dfress.tail(n_steps_out)


# In[39]:


prediction= int(round(dfress.pred[len(dfress)-1]))
print(prediction)
if heurefin.minute== 0:
  minutefin = "00"
else:
  minutefin= str(heurefin.minute)
heureprediction =str(heurefin.hour) + "h" + minutefin
#print(heureprediction)


# In[40]:


prediction


# In[41]:


#print("Nous prévoyons ",prediction," dock(s) Velib disponible(s) à la station Madeleine à ",heureprediction )
val= "Nous prevoyons " + str(prediction) +" dock(s) Velib disponible(s) à la Station Madeleine à "+ heureprediction
from tkinter import * 
fenetre = Tk()
l = LabelFrame(fenetre, text="Prédictions", padx=40, pady=40,font="Arial 20")
l.pack(fill="both", expand="yes")
 
Label(l, text=val,font="Arial 24").pack()
fenetre.mainloop()

# In[ ]:




