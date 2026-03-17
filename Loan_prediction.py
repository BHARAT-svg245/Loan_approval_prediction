import pandas as pd
import numpy as np
import pickle
# Reading the data
df=pd.read_csv("loan_approval_dataset.csv")
df=df.drop("loan_id",axis=1)
y=df["loan_status"]
df=df.drop("loan_status",axis=1)
x=df
#Converting Target feature into categorical value.
from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
y=encoder.fit_transform(y)
# making the columns.
numerical_feature=list(df.select_dtypes(include=["int64","float64"]).columns)
categorical_feature=list(df.select_dtypes(include="object").columns)
from sklearn.preprocessing import OrdinalEncoder,StandardScaler,LabelEncoder
numerical_transformer=StandardScaler()
categorical_transformer=OrdinalEncoder()
# making the training and testing data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
# Making the pipeline and selecting the Classifier.
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
#classifier
model=RandomForestClassifier()
from sklearn.compose import ColumnTransformer
preprocessor=ColumnTransformer(transformers=[("num",numerical_transformer,numerical_feature),("cat",categorical_transformer,categorical_feature)])
# Making the pipeline
pipe=Pipeline([("preprocessor",preprocessor),("classifier",model)])
# fitting the data for training
pipe.fit(x_train,y_train)
#saving the model
# with open("Loan_Approval_Model.pkl","wb") as f:
#     pickle.dump(pipe,f)
#checking the model
df=pd.read_csv("secondloan2.csv")
df=df.drop("loan_id",axis=1)
with open("Loan_Approval_Model.pkl","rb") as f:
    loaded_model=pickle.load(f)
y_pred=loaded_model.predict(df)
print(y_pred)
