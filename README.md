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
df=pd.read_csv("/Encoding Data.csv")
df
```
<img width="387" height="379" alt="image" src="https://github.com/user-attachments/assets/f58150e2-4327-4da7-8c69-3fe60ebb3591" />

```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
<img width="122" height="176" alt="image" src="https://github.com/user-attachments/assets/bb31afa9-2952-4452-b8b0-c7adedf4c3bb" />

```
 df['bo2']=e1.fit_transform(df[["ord_2"]])
 df
```
<img width="351" height="347" alt="image" src="https://github.com/user-attachments/assets/324df7a6-e3bf-48c4-8b89-62cab509a321" />

```
le=LabelEncoder()
 dfc=df.copy()
 dfc['ord_2']=le.fit_transform(dfc['ord_2'])
 dfc
```
<img width="387" height="372" alt="image" src="https://github.com/user-attachments/assets/cbf2fcc9-4254-4731-8cfb-1c1312f4b784" />

```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder()
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]).toarray())
df2=pd.concat([df2,enc],axis=1)
df2
```
<img width="387" height="356" alt="image" src="https://github.com/user-attachments/assets/965fc401-edf7-4e23-b143-b384682b5e60" />

```
pd.get_dummies(df2,columns=["nom_0"])
```
<img width="627" height="344" alt="image" src="https://github.com/user-attachments/assets/c4b8068f-f64c-4900-8b40-45a0630f72f3" />

```
!pip install --upgrade category_encoders
```
```
from category_encoders import BinaryEncoder
df=pd.read_csv("/data.csv")
df
```

<img width="461" height="345" alt="image" src="https://github.com/user-attachments/assets/a1fabbb7-c0ea-4120-9c4b-33f578caec17" />

```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
df
```

<img width="432" height="368" alt="image" src="https://github.com/user-attachments/assets/e1d8acdf-0a75-4ed8-983c-335bc5969389" />

```
dfb=pd.concat([df,nd],axis=1)
dfb
```

<img width="620" height="372" alt="image" src="https://github.com/user-attachments/assets/7631e18f-8678-48aa-b59d-25bec199359f" />

```
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```

<img width="516" height="342" alt="image" src="https://github.com/user-attachments/assets/71121c61-b008-4eed-ae8d-decc4c02e70a" />

```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/Data_to_Transform.csv")
df
```

<img width="711" height="401" alt="image" src="https://github.com/user-attachments/assets/88d91d1d-f414-469d-b18b-9e9952315b31" />

```
df.skew()
```

<img width="255" height="156" alt="image" src="https://github.com/user-attachments/assets/809608a6-e7cf-477e-a881-95a505de1528" />

```
np.log(df["Highly Positive Skew"])
```

<img width="219" height="407" alt="image" src="https://github.com/user-attachments/assets/07689f3e-3771-4c12-8c1a-03aedc474fbd" />

```
np.reciprocal(df["Moderate Positive Skew"])
```

<img width="217" height="397" alt="image" src="https://github.com/user-attachments/assets/afa639e8-0639-4fda-8c32-389df36894af" />

```
np.sqrt(df["Highly Positive Skew"])
```

<img width="216" height="398" alt="image" src="https://github.com/user-attachments/assets/c380fb10-6221-4982-b2d0-0319317e4173" />

```
 np.square(df["Highly Positive Skew"])
```

<img width="223" height="394" alt="image" src="https://github.com/user-attachments/assets/a8da40ab-1b4d-42df-848f-f19d2ddc3ae2" />

```
 df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
 df
```

<img width="981" height="401" alt="image" src="https://github.com/user-attachments/assets/8a36850b-99bb-4d94-9247-f56d7b07aeee" />

```
df.skew()
```

<img width="326" height="185" alt="image" src="https://github.com/user-attachments/assets/43bfac8b-43b6-4158-a6dd-b51273e96dad" />

```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```

<img width="324" height="235" alt="image" src="https://github.com/user-attachments/assets/cd7d02af-0de9-4745-87b7-a7b6041ddfdc" />

```
 from sklearn.preprocessing import QuantileTransformer
 qt=QuantileTransformer(output_distribution='normal')
 df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
 df
```

<img width="1388" height="431" alt="image" src="https://github.com/user-attachments/assets/2a7a9e7f-b035-4888-9093-e30c065be80a" />

```
 import seaborn as sns
 import statsmodels.api as sm
 import matplotlib.pyplot as plt
 sm.qqplot(df["Moderate Negative Skew"],line='45')
 plt.show()
```
<img width="553" height="415" alt="image" src="https://github.com/user-attachments/assets/21418386-1c3f-4c5c-ad54-8ab7182c4352" />

```
 sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
 plt.show()

```

<img width="576" height="415" alt="image" src="https://github.com/user-attachments/assets/35a96e50-15c3-4461-a65d-1d44b5d6f744" />

```
from sklearn.preprocessing import QuantileTransformer
 qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
 df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
 sm.qqplot(df["Moderate Negative Skew"],line='45')
 plt.show()
```

<img width="557" height="425" alt="image" src="https://github.com/user-attachments/assets/30c27e04-0932-4c81-91ee-77e242368392" />

```
 df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
 sm.qqplot(df["Highly Negative Skew"],line='45')
 plt.show()
```

<img width="547" height="432" alt="image" src="https://github.com/user-attachments/assets/aafa393f-48fc-498c-9f63-bcf1d9008690" />

```
dt=pd.read_csv("/titanic_dataset.csv")
dt
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
dt["Age_1"]=qt.fit_transform(dt[["Age"]])
sm.qqplot(dt['Age'],line='45') 
plt.show()
```

<img width="572" height="412" alt="image" src="https://github.com/user-attachments/assets/0197fffe-d656-47d1-881c-e599f53b18c9" />

```
 sm.qqplot(df["Highly Negative Skew_1"],line='45')
 plt.show()
```

<img width="546" height="415" alt="image" src="https://github.com/user-attachments/assets/5abbe3b3-ebd1-42f4-b31b-850c1de5c2ee" />
       
# RESULT:
Thus the given data, Feature Encoding, Transformation process and save the data to a file
 was performed successfully

       
