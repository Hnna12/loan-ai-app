import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- Page Configuration ---
st.set_page_config(page_title="Smart Loan Advisor", page_icon="💰", layout="wide")

# --- ၁။ AI Model Training & Data Loading ---
@st.cache_resource
def train_loan_model():
    df = pd.read_csv('train.csv')
    raw_df = df.copy() 
    
    # Missing Values ဖြည့်ခြင်း
    cols_to_fill = ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Credit_History', 'Loan_Amount_Term']
    for col in cols_to_fill:
        df[col] = df[col].fillna(df[col].mode()[0])
    df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())
    
    # Encoding (Training အတွက်)
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
    df['Married'] = df['Married'].map({'Yes': 1, 'No': 0})
    df['Education'] = df['Education'].map({'Graduate': 1, 'Not Graduate': 0})
    df['Self_Employed'] = df['Self_Employed'].map({'Yes': 1, 'No': 0})
    df['Property_Area'] = df['Property_Area'].map({'Urban': 2, 'Semiurban': 1, 'Rural': 0})
    df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})
    df['Dependents'] = df['Dependents'].replace('3+', 3).astype(int)
    
    X = df.drop(columns=['Loan_ID', 'Loan_Status'])
    y = df['Loan_Status']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, acc, raw_df

model, accuracy, raw_df = train_loan_model()

# --- ၂။ Sidebar UI (အရေးကြီးသော Insight များကို စုပြခြင်း) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2830/2830284.png", width=100)
    st.title("Loan Data Insights")
    st.metric(label="AI Model Accuracy", value=f"{accuracy*100:.2f}%")
    st.markdown("---")
    
    if st.checkbox("📊 အရေးကြီးသော Data Insights များ ကြည့်ရန်"):
        st.subheader("အဓိက ဆုံးဖြတ်ချက်အချက်အလက်များ")
        
        # ၁။ ဒေသအလိုက်
        st.write("📍 ၁။ ဒေသအလိုက် ရရှိမှု")
        area_chart = raw_df.groupby(['Property_Area', 'Loan_Status']).size().unstack()
        st.bar_chart(area_chart)
        
        # ၂။ အကြွေးမှတ်တမ်းအလိုက်
        st.write("💳 ၂။ အကြွေးမှတ်တမ်း အကျိုးသက်ရောက်မှု")
        cred_chart = raw_df.groupby(['Credit_History', 'Loan_Status']).size().unstack()
        st.bar_chart(cred_chart)
        
        # ၃။ ပညာအရည်အချင်းအလိုက်
        st.write("🎓 ၃။ ပညာအရည်အချင်းနှင့် ရလဒ်")
        edu_chart = raw_df.groupby(['Education', 'Loan_Status']).size().unstack()
        st.bar_chart(edu_chart)
        
        st.caption("မှတ်ချက် - အရောင်ရင့်သည် Approved ဖြစ်ပြီး အရောင်ဖျော့သည် Rejected ဖြစ်သည်။")

# --- ၃။ Main UI Layout ---
st.title("💰 Smart Loan Approval Advisor")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### လူကြီးမင်း၏ ချေးငွေရလဒ်ကို စစ်ဆေးပါ")
    st.write("ညာဘက်တွင် အချက်အလက်များကို ဖြည့်စွက်ပေးပါ။ AI မှ အတိတ်က ဒေတာများနှင့် တိုက်ဆိုင်စစ်ဆေးပေးပါမည်။")
    st.image("https://cdn-icons-png.flaticon.com/512/2830/2830284.png", width=350)

with col2:
    st.subheader("📝 အချက်အလက်များ ဖြည့်စွက်ပါ")
    
    g = st.selectbox("Gender", ["Male (ကျား)", "Female (မ)"], index=None, placeholder="ရွေးရန်")
    m = st.selectbox("Married Status", ["Yes (ရှိ)", "No (မရှိ)"], index=None, placeholder="ရွေးရန်")
    d = st.selectbox("Dependents", [0, 1, 2, 3], index=None, placeholder="ရွေးရန်")
    e = st.selectbox("Education", ["Graduate (ဘွဲ့ရ)", "Not Graduate (ဘွဲ့မရ)"], index=None, placeholder="ရွေးရန်")
    loan = st.number_input("Loan Amount (ကျပ်)", min_value=0, value=None, placeholder="ချေးလိုသောပမာဏ")
    
    ch = st.selectbox("Credit History", ["1.0 (ကောင်းမွန်သည်)", "0.0 (မကောင်းပါ)"], index=None)
    pa = st.selectbox("Location", ["Urban", "Semiurban", "Rural"], index=None)

    if st.button("Check Approval Status"):
        if None in [g, m, d, e, inc, loan, ch, pa]:
            st.warning("🚨 ကျေးဇူးပြု၍ အချက်အလက်အားလုံးကို ဖြည့်စွက်ပါ။")
        else:
            with st.spinner('AI မှ တွက်ချက်နေပါသည်...'):
                user_data = {
                    'Gender': 1 if "Male" in g else 0,
                    'Married': 1 if "Yes" in m else 0,
                    'Dependents': int(d),
                    'Education': 1 if "Graduate" in e and "Not" not in e else 0,
                    'Self_Employed': 0,
                    'ApplicantIncome': inc / 5000,
                    'CoapplicantIncome': 0,
                    'LoanAmount': loan / 5000,
                    'Loan_Amount_Term': 360,
                    'Credit_History': 1.0 if "1.0" in ch else 0.0,
                    'Property_Area': 2 if "Urban" in pa else 1 if "Semiurban" in pa else 0
                }
                
                res = model.predict(pd.DataFrame([user_data]))
                st.divider()
                
                if res[0] == 1:
                    st.success("### ✅ Approved! ချေးငွေရရှိရန် အလားအလာရှိပါသည်။")
                    st.balloons()
                else:
                    st.error("### ❌ Rejected! ချေးငွေရရှိရန် ခက်ခဲနိုင်ပါသည်။")
                    st.info("အကြံပြုချက် - Credit History နှင့် ဝင်ငွေအချိုးအစားကို ပြန်စစ်ပါ။")
    
    inc = st.number_input("Monthly Income (ကျပ်)", min_value=0, value=None, placeholder="ဝင်ငွေရိုက်ထည့်ပါ")
