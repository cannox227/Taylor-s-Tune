import numpy as np
import os
import pandas as pd

data = pd.read_csv('data/cleaned_data/rag_dataset.csv')
def get_songs(answers):
    numsongs=180

    title=[0]*numsongs
    album=[0]*numsongs
    allhappy=np.zeros(numsongs)
    alldate=np.zeros(numsongs)
    selffeel=np.zeros(numsongs)
    glassfull=np.zeros(numsongs)
    stages=np.zeros(numsongs)
    tempo=np.zeros(numsongs)
    seriousness=np.zeros(numsongs)
    future=np.zeros(numsongs)
    malefeel=np.zeros(numsongs)
    together=np.zeros(numsongs)

    i=0
    for number in range(0,numsongs):
        set=data.iloc[number]
        title[i]=set.iloc[0]
        album[i]=set.iloc[1]
        allhappy[i]=float(set.iloc[2])
        alldate[i]=float(set.iloc[3])
        selffeel[i]=float(set.iloc[4])
        glassfull[i]=float(set.iloc[5])
        stages[i]=float(set.iloc[6])
        tempo[i]=float(set.iloc[7])
        seriousness[i]=float(set.iloc[8])
        future[i]=float(set.iloc[9])
        malefeel[i]=float(set.iloc[10])
        together[i]=float(set.iloc[11])
        i+=1

    
    selferr=np.array([selffeel[i]-answers[0] for i in range(0,numsongs)])
    stageserr=np.array([stages[i]-answers[1] for i in range(0,numsongs)])
    seriouserr=np.array([seriousness[i]-answers[2] for i in range(0,numsongs)])
    futureerr=np.array([future[i]-answers[3] for i in range(0,numsongs)])
    maleerr=np.array([malefeel[i]-answers[4] for i in range(0,numsongs)])
    togethererr=np.array([together[i]-answers[5] for i in range(0,numsongs)])

    neterr=np.zeros(numsongs)
    for i in range(0,numsongs):
        neterr=selferr**2.+stageserr**2.+seriouserr**2.+futureerr**2.+maleerr**2.+togethererr**2.
		
    oklist=np.zeros(20)
    index=0
    for n in range(0,40):
	    if any(np.abs(neterr)==n):
	        ok=np.where(np.abs(neterr)==n)[0]
	        for x in ok:
	            oklist[index]=x
	            index+=1
	        if index>4:
	            break

    okintlist=[int(i) for i in oklist]
    finalok=okintlist[0:5]
    
    res = []
    for x,item in enumerate(finalok):
        res.append(title[item])

    return res

def find_match(song_title):
    # find song url based on the song title
    try:
        url = data[data['song_name'] == song_title]['url'].values[0]
    except:
        url = 'No url available'
    return url