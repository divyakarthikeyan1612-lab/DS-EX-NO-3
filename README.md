## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.

STEP 2:Clean the Data Set using Data Cleaning Process.

STEP 3:Apply Feature Encoding for the feature in the data set.

STEP 4:Apply Feature Transformation for the feature in the data set.

STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.

2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.

3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.

4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation

• Reciprocal Transformation

• Square Root Transformation

• Square Transformation

  # 2. POWER TRANSFORMATION
• Boxcox method

• Yeojohnson method

# CODING AND OUTPUT:
```
import pandas as pd
import numpy as np
from scipy import stats
df = pd.read_csv("Data.csv")
df
```
<img width="1402" height="566" alt="image" src="https://github.com/user-attachments/assets/674a443f-b436-428d-a1e1-67d28af218cb" />

```
print(df.head())
```
<img width="1394" height="191" alt="image" src="https://github.com/user-attachments/assets/c49e7b3a-0b2f-4286-97f7-a8d25e65b127" />

```
print(df.tail())
```

<img width="1377" height="199" alt="image" src="https://github.com/user-attachments/assets/38276542-fcb5-4da6-b058-49bfec860295" />

```
print(df.info())
```

<img width="1376" height="385" alt="image" src="https://github.com/user-attachments/assets/aa09c37b-a52a-4b39-9d93-f4f0e378cc76" />

```
print(df.describe())
```

<img width="1379" height="263" alt="image" src="https://github.com/user-attachments/assets/09f0ef0f-fb8a-4451-9a1c-34dfe80c873a" />

```
print("Missing values:\n", df.isnull().sum())
```

<img width="1375" height="253" alt="image" src="https://github.com/user-attachments/assets/4454cf6a-e3d9-4594-ab5c-b33a1add4d5e" />

```
from sklearn.preprocessing import OrdinalEncoder,LabelEncoder
climate = ['Cold','Warm','Hot','Very Hot']
ele = OrdinalEncoder(categories=[climate])
ele.fit_transform(df[["Ord_1"]])
```

<img width="1380" height="335" alt="image" src="https://github.com/user-attachments/assets/413e9621-2269-42a5-aa61-e5ba6dff1d92" />

```
df['bo2'] = ele.fit_transform(df[["Ord_1"]])
df

```
<img width="1380" height="437" alt="image" src="https://github.com/user-attachments/assets/2be96fc1-cbce-443b-9199-699448dbdba0" />

```
le = LabelEncoder()
df2 = df.copy()
df2['Ord_2'] = le.fit_transform(df2['Ord_2'])
df2
```


<img width="1375" height="493" alt="image" src="https://github.com/user-attachments/assets/4b99257b-733c-47c3-aef4-9f573db5c79e" />

```
df2['Ord_2'] = le.fit_transform(df2['Ord_2'])
df2
```


<img width="1384" height="455" alt="image" src="https://github.com/user-attachments/assets/bb9c9be8-2158-4436-ae0e-17960fd7e360" />


```
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
df3 = df.copy()
enc = pd.DataFrame(ohe.fit_transform(df2[["City"]]))
df2 = pd.concat([enc,df3],axis = 1)
df2
```



<img width="1384" height="538" alt="image" src="https://github.com/user-attachments/assets/60944394-db90-47f5-b66b-b355e9307327" />


```
pd.get_dummies(df,columns=['City'])
```


<img width="1380" height="415" alt="image" src="https://github.com/user-attachments/assets/0f983331-3b82-4497-b8a9-d248ffbd2a50" />

```
from category_encoders import BinaryEncoder
import pandas as pd
df = pd.read_csv('data.csv')
df
```


<img width="1378" height="484" alt="image" src="https://github.com/user-attachments/assets/9b2da12b-5e18-44e8-89d5-103a7ae02843" />

```
be = BinaryEncoder()
nd = be.fit_transform(df['Ord_2'])
df
```


<img width="1384" height="462" alt="image" src="https://github.com/user-attachments/assets/5ea25630-bd96-4ec2-905a-86eb941eb0dc" />

```
from category_encoders import TargetEncoder
te = TargetEncoder()
CC = df.copy()
new = te.fit_transform(CC["City"],y=CC["Target"])
CC = pd.concat([CC,new],axis = 1)
CC
```


<img width="1387" height="522" alt="image" src="https://github.com/user-attachments/assets/1dceaf18-da73-4df0-a987-99235aa7010b" />


```
import numpy as np
from scipy import stats
df = pd.read_csv('Data_to_Transform.csv')
df
```



<img width="1390" height="551" alt="image" src="https://github.com/user-attachments/assets/708e8ad3-e019-46a7-affa-d4bc3676e582" />


```
df.skew()
```


<img width="1381" height="176" alt="image" src="https://github.com/user-attachments/assets/c93d7e73-521a-4e4b-9d60-32aa82957df0" />


```
np.log(df["Highly Positive Skew"])
```


<img width="1384" height="323" alt="image" src="https://github.com/user-attachments/assets/58970fd1-a06e-48ad-a711-85e0797b09d2" />


```
np.reciprocal(df["Moderate Positive Skew"])
```


<img width="1381" height="325" alt="image" src="https://github.com/user-attachments/assets/9bb1a0f6-26a7-4113-8a45-4b41b0d5f4e0" />


```
np.sqrt(df["Highly Positive Skew"])
```


<img width="1385" height="320" alt="image" src="https://github.com/user-attachments/assets/0f5426f7-d4fb-4183-ae15-01adc38ce41d" />


```
np.square(df["Highly Positive Skew"])
```


<img width="1365" height="325" alt="image" src="https://github.com/user-attachments/assets/dc3f4c24-76ed-439b-a880-f3027c2c66b2" />


```
df["Highly Positive Skew_boxcox"], parameters = stats.boxcox(df["Highly Positive Skew"])
df
```

<img width="1389" height="520" alt="image" src="https://github.com/user-attachments/assets/6246115d-c816-4c9e-bd53-b77b6aedeacf" />


```
df["Moderate Negative Skew_yeojohnson"], parameters = stats.yeojohnson(df["Moderate Negative Skew"])
df
```


<img width="1383" height="539" alt="image" src="https://github.com/user-attachments/assets/e800d4d6-41f4-40f2-922c-ff74c9d440b4" />


```
from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution = 'normal')
df["Moderate Negative Skew_1"] = qt.fit_transform(df[["Moderate Negative Skew"]])
df
```

<img width="1383" height="564" alt="image" src="https://github.com/user-attachments/assets/23db7d78-084a-47e6-9805-a455db0c9f72" />

```
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats
sm.qqplot(df["Moderate Negative Skew"],line = '45')
plt.show()
```


<img width="1387" height="703" alt="image" src="https://github.com/user-attachments/assets/a7d75e89-96c8-4291-bff8-b47be18ad9de" />


```
sm.qqplot(df["Moderate Negative Skew_1"],line = '45')
plt.show()
```


<img width="1378" height="615" alt="image" src="https://github.com/user-attachments/assets/ced426d5-049a-4e40-a195-57c9c7417280" />


```
df["Highly Negative Skew_1"] = qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line = '45')
plt.show()
```


<img width="1384" height="646" alt="image" src="https://github.com/user-attachments/assets/2cf8203a-e4a2-4cf5-8f8b-4ea9fc2bdacb" />


```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew_1"]),line = '45')
plt.show()
```

<img width="1328" height="626" alt="image" src="https://github.com/user-attachments/assets/5248df48-3f19-47ff-a34b-55046216a0cb" />


```
sm.qqplot(df["Highly Negative Skew_1"],line = '45')
plt.show()
```

<img width="1381" height="621" alt="image" src="https://github.com/user-attachments/assets/474934a2-2a3c-4c8e-a70b-a9089f1fb955" />


```
sm.qqplot(np.abs(df["Highly Negative Skew_1"]),line = '45')
plt.show()
```

<img width="1394" height="614" alt="image" src="https://github.com/user-attachments/assets/9951d500-d4e2-4fb7-a1fd-b8e623647a4a" />


```
sm.qqplot(np.log(df["Highly Negative Skew_1"]),line = '45')
plt.show()
```

<img width="1362" height="693" alt="image" src="https://github.com/user-attachments/assets/a699ce2d-814d-4b37-b5bd-412956d86256" />


```
sm.qqplot(np.sqrt(df["Moderate Negative Skew_1"]),line='45')
plt.show()
```

<img width="1374" height="681" alt="image" src="https://github.com/user-attachments/assets/d733b748-3bd0-44be-aa44-b776cbd5d315" />


```
pd.concat([CC,new],axis = 1)
```

<img width="1386" height="416" alt="image" src="https://github.com/user-attachments/assets/30ab4914-64c7-41e5-a5c6-e2879b4cdc60" />


# RESULT:
       # INCLUDE YOUR RESULT HERE

       
