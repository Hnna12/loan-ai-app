import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- Page Configuration ---
st.set_page_config(page_title="Smart Loan Advisor", page_icon="💰", layout="wide")

# --- Custom Styling ---
st.markdown("""
    <style>
    .stApp { background-color: #f8fafc; }
    div[data-testid="stVerticalBlock"] > div:has(div.stColumn) {
        background: white; padding: 30px; border-radius: 20px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.05); border: 1px solid #e2e8f0;
    }
    .stButton>button { width: 100%; border-radius: 10px; height: 3em; background-color: #007bff; color: white; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- ၁။ AI Model Training with Accuracy ---
@st.cache_resource
def train_loan_model():
    df = pd.read_csv('train.csv')
    cols_to_fill = ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Credit_History', 'Loan_Amount_Term']
    for col in cols_to_fill:
        df[col] = df[col].fillna(df[col].mode()[0])
    df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())
    
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
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return model, acc

model, accuracy = train_loan_model()

# --- ၂။ Sidebar UI ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2830/2830284.png", width=100)
    st.title("About AI Advisor")
    st.info("AI နည်းပညာဖြင့် ချေးငွေရရှိနိုင်ခြေကို ခန့်မှန်းပေးပါသည်။")
    st.metric(label="Model Accuracy (AI ၏ တိကျမှုနှုန်း)", value=f"{accuracy*100:.2f}%")
    st.markdown("---")
    st.warning("💡 Credit history ကောင်းမွန်ခြင်းသည် ချေးငွေရရှိရန် အရေးကြီးဆုံးဖြစ်ပါသည်။")

# --- ၃။ Main UI Layout ---
st.title("💰 Smart Loan Approval Advisor")

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown("### လူကြီးမင်း၏ ချေးငွေရလဒ်ကို စစ်ဆေးပါ")
    st.write("ညာဘက်ရှိ ဖောင်တွင် အချက်အလက်များကို မှန်ကန်စွာ ဖြည့်စွက်ပေးပါ။")
    st.image("https://cdn-icons-png.flaticon.com/512/2830/2830284.png", use_container_width=True)

with col_right:
    st.subheader("📝 အချက်အလက်များ ဖြည့်စွက်ပါ")
    
    gender = st.selectbox("Gender (ကျား/မ)", ["Male (ကျား)", "Female (မ)"], index=None, placeholder="ရွေးချယ်ရန်")
    married = st.selectbox("Married Status (အိမ်ထောင်ရှိ/မရှိ)", ["Yes (ရှိ)", "No (မရှိ)"], index=None, placeholder="ရွေးချယ်ရန်")
    dependents = st.selectbox("Number of Dependents (မှီခိုသူဦးရေ)", [0, 1, 2, 3], index=None, placeholder="ရွေးချယ်ရန်")
    education = st.selectbox("Education Level (ပညာအရည်အချင်း)", ["Graduate (ဘွဲ့ရ)", "Not Graduate (ဘွဲ့မရ)"], index=None, placeholder="ရွေးချယ်ရန်")
    
    income_mmk = st.number_input("Monthly Income (လစဉ်ဝင်ငွေ - ကျပ်)", min_value=0, value=None, placeholder="ဝင်ငွေရိုက်ထည့်ပါ (ဥပမာ- ၅၀၀,၀၀၀)")
    
    # Placeholder ထည့်သွင်းထားသော နေရာ
    loan_amount_mmk = st.number_input("Loan Amount (ချေးယူလိုသောပမာဏ - ကျပ်)", min_value=0, value=None, placeholder="ပမာဏရိုက်ထည့်ပါ (ဥပမာ- ၁,၀၀၀,၀၀၀)")
    credit_history = st.selectbox("Credit History Score (အကြွေးမှတ်တမ်း)", ["1.0 (ကောင်းမွန်သည်)", "0.0 (မကောင်းပါ)"], index=None, placeholder="ရွေးချယ်ရန်")
    property_area = st.selectbox("Property Location (နေထိုင်ရာဒေသ)", ["Urban (မြို့ပြ)", "Semiurban (မြို့ဆင်ခြေဖုံး)", "Rural (ကျေးလက်)"], index=None, placeholder="ရွေးချယ်ရန်")

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Analyze My Loan Status"):
        if None in [gender, married, dependents, education, income_mmk, loan_amount_mmk, credit_history, property_area]:
            st.warning("🚨 ကျေးဇူးပြု၍ အချက်အလက်အားလုံးကို ပြည့်စုံအောင် အရင်ဖြည့်ပါ။")
        else:
            with st.spinner('AI မှ တွက်ချက်နေပါသည်...'):
                import time
                time.sleep(1) 
                
                income_usd = income_mmk / 5000
                loan_usd = loan_amount_mmk / 5000

                user_input = {
                    'Gender': 1 if "Male" in gender else 0,
                    'Married': 1 if "Yes" in married else 0,
                    'Dependents': int(dependents),
                    'Education': 1 if "Graduate" in education and "Not" not in education else 0,
                    'Self_Employed': 0,
                    'ApplicantIncome': income_usd,
                    'CoapplicantIncome': 0,
                    'LoanAmount': loan_usd,
                    'Loan_Amount_Term': 360,
                    'Credit_History': 1.0 if "1.0" in credit_history else 0.0,
                    'Property_Area': 2 if "Urban" in property_area else 1 if "Semiurban" in property_area else 0
                }
                
                input_df = pd.DataFrame([user_input])
                prediction = model.predict(input_df)
                
                st.divider()
                if prediction[0] == 1:
                    st.balloons()
                    st.success(f"### 🎉 Congratulations! \n\n လူကြီးမင်း၏ ချေးငွေလျှောက်ထားမှုသည် Approved ဖြစ်နိုင်ခြေ များပါသည်။ \n\n (AI Confidence: {accuracy*100:.1f}%)")
                else:
                    st.error(f"### ❌ Sorry! \n\n လူကြီးမင်း၏ လက်ရှိ အချက်အလက်များအရ ချေးငွေရရှိရန် Rejected ဖြစ်နိုင်ခြေ များပါသည်။ \n\n (AI Confidence: {accuracy*100:.1f}%)")

