# Spectral-Read
Code that reads Hyperspectral data and clusters every band into 5 different centers

```python
from spectral import *

filename = "PRS_L2D_STD_20210307145414_20210307145418_0001.hdr"#new6

img = open_image(filename)
img
```




    	Data Source:   '.\.\PRS_L2D_STD_20210307145414_20210307145418_0001.bsq'
    	# Rows:           1181
    	# Samples:        1230
    	# Bands:           234
    	Interleave:        BSQ
    	Quantization:  32 bits
    	Data format:   float32




```python
# Reference system
# NDVI
# VNIR Why?
# sklearn, statistics
# nan to 0

pc = principal_components(img.load())
```


```python
xdata = pc.transform(img)
xdata
```




    	TransformedImage object with output dimensions:
    	# Rows:           1181
    	# Samples:        1230
    	# Bands:           234
    
    	The linear transform is applied to the following image:
    
    	Data Source:   '.\.\PRS_L2D_STD_20210307145414_20210307145418_0001.bsq'
    	# Rows:           1181
    	# Samples:        1230
    	# Bands:           234
    	Interleave:        BSQ
    	Quantization:  32 bits
    	Data format:   float32




```python
img[0,0,:]
```




    array([[[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]], dtype=float32)




```python
import matplotlib.pyplot as plt


plt.imshow(img[:,:,230], cmap='nipy_spectral')
```




   




    
![](https://github.com/highjoule/Spectral-Read/blob/main/images/output_4_1.png)

    



```python
import numpy as np

x = np.array(img[:,:,:])
```


```python
x.shape
```




    (1181, 1230, 234)




```python
import matplotlib.pyplot as plt

plt.plot(x[432,307,:])
```



![](https://github.com/highjoule/Spectral-Read/blob/main/images/output_7_1.png)   

    



```python
j = 2
ds = x[:,:,j]
plt.imshow(ds, cmap='nipy_spectral')
```




 



![](https://github.com/highjoule/Spectral-Read/blob/main/images/output_8_1.png)   


    



```python
resh = x.reshape(x.shape[0]*x.shape[1],x.shape[2])
```


```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=5, random_state=0).fit(resh)
clustered = kmeans.cluster_centers_[kmeans.labels_]
```


```python
clus = clustered.reshape(x.shape[0],img.shape[1],img.shape[2])
clus[:,:,149].shape
```




    (1181, 1230)




```python
import pickle

savedfile='model.sav'
pickle.dump(clustered,open(savedfile,'wb'))
```


```python
import random

rngw = x[0,0,:].shape
wv = random.randint(0,rngw[0])

stack = np.hstack((ldc[:,:,wv],x[:,:,wv]))
plt.imshow(stack,cmap='nipy_spectral')
plt.show()
print("clustered/original",wv)
```


![](https://github.com/highjoule/Spectral-Read/blob/main/images/output_13_0.png)   
    
    


    clustered/original 35
    


```python
savedfile = 'model.sav'
loadcluster = pickle.load(open(savedfile,'rb'))
loadclus = loadcluster.reshape(x.shape[0],img.shape[1],img.shape[2])
```


```python
loadclus[loadclus<0.001]=0.0
ldc=np.nan_to_num(loadclus,nan=0.0)
```


```python
import matplotlib.pyplot as plt
import random
from sklearn.metrics import r2_score

rng = x[:,:,0].shape
a,b = random.randint(0,rng[0]),random.randint(0, rng[1])



plt.plot(x[a,b,:],label='original',color='blue')
plt.plot(ldc[a,b,:],label='clustered',color='red')
plt.grid(b=True, which='major', color='#A79E9E', linestyle='-', linewidth=0.4)
plt.minorticks_on()
plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0)
plt.show()
print(a,b,r2_score(x[a,b,:],ldc[a,b,:]))
```


![](https://github.com/highjoule/Spectral-Read/blob/main/images/output_16_0.png)   
    
    


    1052 385 1.0
    


```python
ndvi_map
```




    array([[nan, nan, nan, ..., nan, nan, nan],
           [nan, nan, nan, ..., nan, nan, nan],
           [nan, nan, nan, ..., nan, nan, nan],
           ...,
           [nan, nan, nan, ..., nan, nan, nan],
           [nan, nan, nan, ..., nan, nan, nan],
           [nan, nan, nan, ..., nan, nan, nan]])




```python

ndvi_map = ndvi(ldc, 27, 47)
plt.imshow(ndvi_map,cmap='cividis')
plt.clim(np.nanmin(ndvi_map),np.nanmax(ndvi_map))
plt.colorbar()
```




   




 ![](https://github.com/highjoule/Spectral-Read/blob/main/images/output_18_1.png)   
   
    



```python
np.nanmin(ndvi_map),np.nanmax(ndvi_map)
```




    (0.06718798325012743, 0.0798744817883301)




```python
img.metadata.keys()
```




    dict_keys(['description', 'samples', 'lines', 'bands', 'data type', 'interleave', 'file type', 'header offset', 'byte order', 'map info', 'coordinate system string', 'band names', 'wavelength', 'fwhm', 'wavelength units', 'data ignore value', 'acquisition time'])




```python
import pandas as pd

info = pd.DataFrame(img.metadata["wavelength"],img.metadata["band names"],columns=['wavelenght (nm)'])
info
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>wavelenght (nm)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Band 1 (Band 1)</th>
      <td>402.440186</td>
    </tr>
    <tr>
      <th>Band 2 (Band 2)</th>
      <td>411.316376</td>
    </tr>
    <tr>
      <th>Band 3 (Band 3)</th>
      <td>419.372498</td>
    </tr>
    <tr>
      <th>Band 4 (Band 4)</th>
      <td>426.967438</td>
    </tr>
    <tr>
      <th>Band 5 (Band 5)</th>
      <td>434.308411</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>Band 230 (Band 169)</th>
      <td>2469.415527</td>
    </tr>
    <tr>
      <th>Band 231 (Band 170)</th>
      <td>2476.791260</td>
    </tr>
    <tr>
      <th>Band 232 (Band 171)</th>
      <td>2483.590576</td>
    </tr>
    <tr>
      <th>Band 233 (Band 172)</th>
      <td>2490.028076</td>
    </tr>
    <tr>
      <th>Band 234 (Band 173)</th>
      <td>2496.874023</td>
    </tr>
  </tbody>
</table>
<p>234 rows × 1 columns</p>
</div>




```python
info[50:100]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>wavelenght (nm)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Band 51 (Band 51)</th>
      <td>844.427124</td>
    </tr>
    <tr>
      <th>Band 52 (Band 52)</th>
      <td>855.182129</td>
    </tr>
    <tr>
      <th>Band 53 (Band 53)</th>
      <td>865.949402</td>
    </tr>
    <tr>
      <th>Band 54 (Band 54)</th>
      <td>876.640808</td>
    </tr>
    <tr>
      <th>Band 55 (Band 55)</th>
      <td>887.265869</td>
    </tr>
    <tr>
      <th>Band 56 (Band 56)</th>
      <td>897.986511</td>
    </tr>
    <tr>
      <th>Band 57 (Band 57)</th>
      <td>908.643982</td>
    </tr>
    <tr>
      <th>Band 58 (Band 58)</th>
      <td>919.176270</td>
    </tr>
    <tr>
      <th>Band 59 (Band 59)</th>
      <td>929.385498</td>
    </tr>
    <tr>
      <th>Band 60 (Band 60)</th>
      <td>939.860046</td>
    </tr>
    <tr>
      <th>Band 61 (Band 61)</th>
      <td>951.361511</td>
    </tr>
    <tr>
      <th>Band 62 (Band 62)</th>
      <td>962.264465</td>
    </tr>
    <tr>
      <th>Band 63 (Band 63)</th>
      <td>972.634644</td>
    </tr>
    <tr>
      <th>Band 64 (Band 3)</th>
      <td>942.957886</td>
    </tr>
    <tr>
      <th>Band 65 (Band 4)</th>
      <td>951.011414</td>
    </tr>
    <tr>
      <th>Band 66 (Band 5)</th>
      <td>959.522034</td>
    </tr>
    <tr>
      <th>Band 67 (Band 6)</th>
      <td>969.389954</td>
    </tr>
    <tr>
      <th>Band 68 (Band 7)</th>
      <td>978.737671</td>
    </tr>
    <tr>
      <th>Band 69 (Band 8)</th>
      <td>988.411560</td>
    </tr>
    <tr>
      <th>Band 70 (Band 9)</th>
      <td>998.374939</td>
    </tr>
    <tr>
      <th>Band 71 (Band 10)</th>
      <td>1008.168274</td>
    </tr>
    <tr>
      <th>Band 72 (Band 11)</th>
      <td>1017.984619</td>
    </tr>
    <tr>
      <th>Band 73 (Band 12)</th>
      <td>1028.806274</td>
    </tr>
    <tr>
      <th>Band 74 (Band 13)</th>
      <td>1037.764648</td>
    </tr>
    <tr>
      <th>Band 75 (Band 14)</th>
      <td>1047.431152</td>
    </tr>
    <tr>
      <th>Band 76 (Band 15)</th>
      <td>1057.380371</td>
    </tr>
    <tr>
      <th>Band 77 (Band 16)</th>
      <td>1067.609985</td>
    </tr>
    <tr>
      <th>Band 78 (Band 17)</th>
      <td>1078.039062</td>
    </tr>
    <tr>
      <th>Band 79 (Band 18)</th>
      <td>1088.589844</td>
    </tr>
    <tr>
      <th>Band 80 (Band 19)</th>
      <td>1099.108276</td>
    </tr>
    <tr>
      <th>Band 81 (Band 20)</th>
      <td>1109.740845</td>
    </tr>
    <tr>
      <th>Band 82 (Band 21)</th>
      <td>1120.494629</td>
    </tr>
    <tr>
      <th>Band 83 (Band 22)</th>
      <td>1131.143066</td>
    </tr>
    <tr>
      <th>Band 84 (Band 23)</th>
      <td>1141.873657</td>
    </tr>
    <tr>
      <th>Band 85 (Band 24)</th>
      <td>1152.471558</td>
    </tr>
    <tr>
      <th>Band 86 (Band 25)</th>
      <td>1163.484131</td>
    </tr>
    <tr>
      <th>Band 87 (Band 26)</th>
      <td>1174.525757</td>
    </tr>
    <tr>
      <th>Band 88 (Band 27)</th>
      <td>1185.397949</td>
    </tr>
    <tr>
      <th>Band 89 (Band 28)</th>
      <td>1196.174805</td>
    </tr>
    <tr>
      <th>Band 90 (Band 29)</th>
      <td>1207.114868</td>
    </tr>
    <tr>
      <th>Band 91 (Band 30)</th>
      <td>1217.695923</td>
    </tr>
    <tr>
      <th>Band 92 (Band 31)</th>
      <td>1228.998901</td>
    </tr>
    <tr>
      <th>Band 93 (Band 32)</th>
      <td>1240.056763</td>
    </tr>
    <tr>
      <th>Band 94 (Band 33)</th>
      <td>1250.795532</td>
    </tr>
    <tr>
      <th>Band 95 (Band 34)</th>
      <td>1262.334351</td>
    </tr>
    <tr>
      <th>Band 96 (Band 35)</th>
      <td>1273.306396</td>
    </tr>
    <tr>
      <th>Band 97 (Band 36)</th>
      <td>1284.279541</td>
    </tr>
    <tr>
      <th>Band 98 (Band 37)</th>
      <td>1295.196655</td>
    </tr>
    <tr>
      <th>Band 99 (Band 38)</th>
      <td>1306.047241</td>
    </tr>
    <tr>
      <th>Band 100 (Band 39)</th>
      <td>1317.017700</td>
    </tr>
  </tbody>
</table>
</div>




```python

```


```python

```
