print(__doc__)

from time import time
import numpy as np
import pylab as pl

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale


def getIndex(item, arr, strict):
  try:
    return arr.index(item)
  except:
    if strict:
      return len(arr) - 1
    else:
      arr.append(item)
      return len(arr) - 1
  

def insertInDict(data, key):
  if key in data:
    data[key] = data[key] + 1
  else:
    data[key] = 1

f = open('profiles_dataset.csv')
lines = []
features = []
states = []
countries = []
scale = 33
booleans = ['False', 'True', 'None']
gender = ['M', 'F', 'None']
f.readline()
countriesDict = {}
for line in f.readlines():
  words = line.split(';')
  locations = words[1].split(',')
  if len(locations) >= 2:
    state = locations[0]
    country = locations[1].strip()
  else:
    state = "None"
    country = locations[0].strip()
  insertInDict(countriesDict, country) 
  feature = []
  #feature.append(getIndex(state, states))
  #feature.append( getIndex(country, countries))
  #feature.append(getIndex(words[2], booleans, strict = True) * scale)
  try:
    feature.append(int(words[3]))
  except:
    continue
  feature.append(getIndex(words[4], gender, strict = True) * scale)
  # try:
  #   feature.append(int(words[5][0:4]))
  # except:
  #   feature.append(1900)
  features.append(feature)
  lines.append(line)


print len(features)

def showCountryPlot(data):
  values = sorted(data.values(), reverse=True)[:10]
  countries = sorted(data, key=data.get, reverse=True)[:10]

  fig = pl.figure()
  xlocations = np.array(range(len(values)))+0.5
  width = 0.5
  pl.bar(xlocations, values, width=width)
  pl.xticks(xlocations+ width/2, countries)
  pl.show()

showCountryPlot(countriesDict)
    
def printStr(line):
  #return states[line[0]] + ', ' + countries[line[1]] + ', ' + booleans[line[2]] + ', Age:' + str(line[3]) + ', ' + gender[line[4]] + ', Year' + str(line[5]) 
  return 'Age:' + str(line[0]) + ', ' + gender[line[1] / scale] #+ ', Year' + str(line[3]) 

labels = ['State', 'Country', 'Location Shared', 'Age', 'Gender', 'Year Joined']
estimator = KMeans(init='k-means++', n_clusters=5, n_init=10)
data=features
estimator.fit(data)
clusters = estimator.predict(data)
print len(clusters)
ans = {}
for cluster, feature in zip(clusters, features):
  line = printStr(feature)
  if cluster not in ans:
    ans[cluster] = [feature]
  else:
    ans[cluster].append(feature)



for cluster in ans:
  print cluster, len(ans[cluster])
  print np.median(ans[cluster][:][0]), gender[ans[cluster][0][1] / scale]
f.close()
