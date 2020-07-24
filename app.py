import flask
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


#load models at top of app to load into memory only one time
with open('models/xgb_cv_final.pkl', 'rb') as f:
    clf_individual = pickle.load(f)
#load models at top of app to load into memory only one time
with open('models/gb_cv_compact_joint.pkl', 'rb') as f:
    clf_joint = pickle.load(f)

with open('models/knn_regression.pkl', 'rb') as f:
    knn = pickle.load(f)    
ss = StandardScaler()
#feature space
df_train_jl_scale = pd.read_csv('data/df_train_jl_scale.csv')
#load APR table
df_fico_apr = pd.read_csv('data/grade_to_apr.csv')

df_macro_mean  = pd.read_csv('data/df_macro_mean.csv', index_col=0, dtype=np.float64)

df_macro_std = pd.read_csv('data/df_macro_std.csv', index_col=0, dtype=np.float64)
drop_columns=['emp_length','purpose','revol_bal','grade','int_rate']
df_macro_mean=df_macro_mean.drop(columns=drop_columns)
df_macro_std =df_macro_std.drop(columns=drop_columns)


home_to_int = {'MORTGAGE': 4,
               'RENT': 3,
               'OWN': 5,         
               'ANY': 2,            
               'OTHER': 1,          
               'NONE':0 }

sub_grade_to_char={1:'A1',2:'A2',3:'A3',4:'A4',5:'A5',6:'B1',7:'B2',8:'B3',9:'B4',10:'B5'
                   ,11:'C1',12:'C2',13:'C3',14:'C4',15:'C5',16:'D1',17:'D2',18:'D3',19:'D4',20:'D5'
                   ,21:'E1',22:'E2',23:'E3',24:'E4',25:'E5',26:'F1',27:'F2',28:'F3',29:'F4',30:'F5'
                   ,31:'G1',32:'G2',33:'G3',34:'G4',35:'G5'}

app = flask.Flask(__name__, template_folder='templates')
@app.route('/')
def main():
    return (flask.render_template('index.html'))

@app.route('/report')
def report():
    return (flask.render_template('report.html'))

@app.route('/jointreport')
def jointreport():
    return (flask.render_template('jointreport.html'))


@app.route("/Individual", methods=['GET', 'POST'])
def Individual():
    
    if flask.request.method == 'GET':
        return (flask.render_template('Individual.html'))
    
    if flask.request.method =='POST':
        
        #get input
        
        #ask for first 2 digits of zip code as integer
        code = int(flask.request.form['code'])
        #fico score as integer
        fico_avg_score = int(flask.request.form['fico_avg_score'])
        #loan amount as integer
        loan_amnt = float(flask.request.form['loan_amnt'])
        #term as integer: 36 or 60
        term = int(flask.request.form['term'])
        #debt to income as float
        dti = float(flask.request.form['dti'])
        #home ownership as string
        home_ownership = flask.request.form['home_ownership']
        #number or mortgage accounts as integer
        mort_acc = int(flask.request.form['mort_acc'])
        #annual income as float
        annual_inc = float(flask.request.form['annual_inc'])
        #number of open accounts as integer
        open_acc = int(flask.request.form['open_acc'])
        #verification status as 0, 1, 2
        verification_status = int(flask.request.form['verification_status'])
        #revolving utilization as float
        revol_util = float(flask.request.form['revol_util'])
        #The total number of credit lines currently in the borrower's credit file
        total_acc = int(flask.request.form['total_acc'])
        #time since first credit line in months
        er_credit_open_date = pd.to_datetime(flask.request.form['er_credit_open_date'])
        issue_d = pd.to_datetime("today")
        credit_hist = issue_d - er_credit_open_date
        credit_line_ratio=open_acc/total_acc
        balance_annual_inc=loan_amnt/annual_inc
        #calculate grade from FICO
        sub_grade = knn.predict(np.reshape([fico_avg_score], (1,-1)))[0]
        #calculate grade
        grade = round(sub_grade/5) + 1
        #get interest rate
        apr_row = df_fico_apr[df_fico_apr['grade_num']==sub_grade]
        
        
        #use equal monthly installment formula
        if term==36:
            int_rate = apr_row['36_mo'].values[0]/100
            emi = int_rate/36
            installment = loan_amnt * (int_rate*(1+int_rate)**36) / ((1+int_rate)**36 - 1)
            term = 1
            
        else:
            int_rate = apr_row['60_mo'].values[0]/100
            emi = int_rate/60
            installment = loan_amnt * (int_rate*(1+int_rate)**36) / ((1+int_rate)**36 - 1)
            term = 2
            
        #make integer
        installment = int(installment)
        inst_amnt_ratio=installment/loan_amnt

        
        temp = pd.DataFrame(index=[1])
        #temp['fico'] = fico
        
        temp['term']=term
        temp['sub_grade']=sub_grade
        temp['home_ownership']=home_to_int[home_ownership.upper()]
        temp['annual_inc']=np.log(annual_inc)
        temp['verification_status']=verification_status
        temp['dti']=dti
        temp['revol_util']=revol_util
        temp['mort_acc'] = mort_acc
        temp['credit_hist']=credit_hist.days
        temp['credit_line_ratio']=credit_line_ratio
        temp['balance_annual_inc']=balance_annual_inc
        temp['fico_avg_score'] = fico_avg_score
        temp['inst_amnt_ratio']=inst_amnt_ratio
        
        #create original output dict
        output_dict= dict()
        output_dict['Provided Annual Income'] = annual_inc
        output_dict['Provided FICO Score'] = fico_avg_score
        output_dict['Predicted Interest Rate'] = int_rate * 100 #revert back to percentage
        output_dict['Estimated Installment Amount']=installment
        output_dict['Number of Payments'] = 36 if term==1 else 60
        output_dict['Sub Grade']= sub_grade_to_char[35-int(sub_grade)]
        output_dict['Loan Amount']=loan_amnt
        
        #create deep copy 
        scale = temp.copy()
        for feat in df_macro_mean.columns:
            scale[feat] = (scale[feat] - df_macro_mean.loc[code,feat]) / df_macro_std.loc[code,feat]
        
            
        #make prediction
        pred = clf_individual.predict(scale)
        
        if dti>43:
            res = 'Debt to income too high!'
        elif balance_annual_inc >=0.43:
            res= 'Debt to income too high!'
        elif revol_util>=90:
            res = 'Amount of credit used up too high!'
        elif pred ==1:
            res = 'Loan Denied'
        else:
            print(scale)
            res = 'Congratulations! Approved!'
 
        
        
        #render form again and add prediction
        return flask.render_template('Individual.html',
                                     original_input=output_dict,
                                     result=res,
                                     )
        
        
@app.route("/Joint", methods=['GET', 'POST'])
def Joint():
    
    if flask.request.method == 'GET':
        return (flask.render_template('Joint.html'))
    
    if flask.request.method =='POST':
        ['credit_hist', 'total_bal_il',
       'home_ownership', 'balance_annual_inc', 'sec_app_revol_util',
       'sec_balance_annual_inc', 'term', 'sec_app_inq_last_6mths',
       'sec_app_mort_acc', 'dti_joint', 'sec_fico_avg_score', 'sub_grade']
        #get input
        #fico score as integer
        sec_fico_avg_score = int(flask.request.form['sec_fico_avg_score'])
        fico_avg_score = int(flask.request.form['fico_avg_score'])
        #loan amount as integer
        loan_amnt = float(flask.request.form['loan_amnt'])
        #term as integer: 36 or 60
        term = int(flask.request.form['term'])
        #debt to income as float
        dti_joint = float(flask.request.form['dti_joint'])
        #home ownership as string
        home_ownership = flask.request.form['home_ownership']
        #number or mortgage accounts as integer
        sec_app_mort_acc = int(flask.request.form['sec_app_mort_acc'])
        #annual income as float
        annual_inc = float(flask.request.form['annual_inc'])
        #annual income as float
        sec_annual_inc = float(flask.request.form['sec_annual_inc'])
        #number of open accounts as integer
        sec_app_inq_last_6mths= int(flask.request.form['sec_app_inq_last_6mths'])
        #revolving utilization as float
        sec_app_revol_util = float(flask.request.form['sec_app_revol_util'])
        total_bal_il = float(flask.request.form['total_bal_il'])

        #time since first credit line in months
        er_credit_open_date = pd.to_datetime(flask.request.form['er_credit_open_date'])
      
        issue_d = pd.to_datetime("today")
        credit_hist = issue_d - er_credit_open_date
        balance_annual_inc=loan_amnt/annual_inc
        sec_balance_annual_inc=loan_amnt/sec_annual_inc
        
           #calculate grade from FICO
        sub_grade = knn.predict(np.reshape([fico_avg_score], (1,-1)))[0]
        #calculate grade
        grade = round(sub_grade/5) + 1
        #get interest rate
        apr_row = df_fico_apr[df_fico_apr['grade_num']==sub_grade]
        
        #use equal monthly installment formula
        if term==36:
            int_rate = apr_row['36_mo'].values[0]/100
            emi = int_rate/36
            installment = loan_amnt * (int_rate*(1+int_rate)**36) / ((1+int_rate)**36 - 1)
            term = 1
            
        else:
            int_rate = apr_row['60_mo'].values[0]/100
            emi = int_rate/60
            installment = loan_amnt * (int_rate*(1+int_rate)**36) / ((1+int_rate)**36 - 1)
            term = 2
            
        #make integer
        installment = int(installment)
        
        temp = pd.DataFrame(index=[1])
        #temp['fico'] = fico
        
        temp['term']=term
        temp['sub_grade']=sub_grade
        temp['home_ownership']=home_to_int[home_ownership.upper()]
        temp['dti_joint']=dti_joint
        temp['sec_app_revol_util']=sec_app_revol_util
        temp['sec_app_mort_acc'] = sec_app_mort_acc
        temp['credit_hist']=credit_hist.days
        temp['balance_annual_inc']=balance_annual_inc
        temp['sec_balance_annual_inc']=sec_balance_annual_inc
        temp['sec_fico_avg_score'] = sec_fico_avg_score
        temp['total_bal_il']=np.log(total_bal_il)
        temp['sec_app_inq_last_6mths']=sec_app_inq_last_6mths
       
        #create original output dict
        output_dict= dict()
        output_dict['Given Annual Income'] = annual_inc
        output_dict['Calculated Avg FICO'] = fico_avg_score
        output_dict['Predicted Interest Rate'] = int_rate * 100 #revert back to percentage
        output_dict['Predicted Installment']=installment
        output_dict['Number of Payments'] = 36 if term==1 else 60
        output_dict['Sub Grade']= sub_grade_to_char[35-int(sub_grade)]
        output_dict['Loan Amount']=loan_amnt
        
        #create deep copy 
        X_train = df_train_jl_scale[temp.columns]
        ss.fit(X_train)

        scale = temp.copy()
        scale = ss.transform(scale)
            
        #make prediction
        pred = clf_joint.predict(scale)
        
        if dti_joint>43:
            res = 'Debt to income too high for secondary applicant'
        elif balance_annual_inc >=0.43:
            res= 'Debt to income too high!'
        elif sec_app_revol_util>=90:
            res = 'Amount of credit used up too high for secondary applicant'
        elif pred ==1:
            res = 'Loan Denied!'
        else:
            res = 'Congratulations! Approved!'
 
        
        
        #render form again and add prediction
        return flask.render_template('Joint.html',
                                     original_input=output_dict,
                                     result=res,
                                     )        
if __name__ == '__main__':
    app.run(debug=True)