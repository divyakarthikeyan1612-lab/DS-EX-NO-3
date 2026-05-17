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




# RESULT:
       # INCLUDE YOUR RESULT HERE

       
