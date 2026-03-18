import streamlit as st
import pickle
import pandas as pd

#Loading the model 
with open("Loan_Approval_Model.pkl","rb") as f:
    predictor=pickle.load(f)
#Giving the title to the APP
st.title("Loan Approval")
#uploading the users data
upload_file=st.file_uploader("Upload file",type=["csv"])
if upload_file is not None:
  df=pd.read_csv(upload_file)
  if "loan_id" not in df.columns:
      st.error("loan_id not in dataframe")
  else:    
      loan_id=df["loan_id"]
      df=df.drop("loan_id",axis=1)
      st.write("preview of data")
      st.dataframe(df)

      if st.button("Predict"):
          results=predictor.predict(df)
          op_list=[]
          for target in results:
              if target ==0:
                  op_list.append("Rejected")
              else:
                  op_list.append("Approved") 

          loan_status=pd.DataFrame({"loan_id":loan_id,"status":op_list})
          st.dataframe(loan_status)
    
