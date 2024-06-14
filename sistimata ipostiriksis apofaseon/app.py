import streamlit as st
import joblib
import numpy as np

model = joblib.load("random_forest_model.pkl")

def main():
    st.title("Credit Application Form")

    status_of_checking_account = st.selectbox("Status of existing checking account:",
                                              ["< 0 DM", "0 - 200 DM", "> 200 DM", "no checking account"])

    if status_of_checking_account == "< 0 DM":
        status_of_checking_account = 0
    elif status_of_checking_account == "0 - 200 DM":
        status_of_checking_account = 1
    elif status_of_checking_account == "> 200 DM":
        status_of_checking_account = 2
    elif status_of_checking_account == "no checking account":
        status_of_checking_account = 3

    duration_in_month = st.number_input("Duration in months:", min_value=1) 

    credit_history = st.selectbox("Credit history:",
                                  ["no credits taken/all credits paid back duly", "all credits at this bank paid back duly",
                                   "existing credits paid back duly till now", "delay in paying off in the past",
                                   "critical account/other credits existing (not at this bank)"])
    
    if credit_history == "no credits taken/all credits paid back duly":
        credit_history = 0
    elif credit_history == "all credits at this bank paid back duly":
        credit_history = 1
    elif credit_history == "existing credits paid back duly till now":
        credit_history = 2
    elif credit_history == "delay in paying off in the past":
        credit_history = 3
    elif credit_history == "critical account/other credits existing (not at this bank)":
        credit_history = 4

    purpose = st.selectbox("Purpose:",
                            ["car (new)", "car (used)", "furniture/equipment", "radio/television", "domestic appliances",
                            "repairs","education", "retraining", "business", "others"])
    
    if purpose == "car (new)":
        purpose = 0
    elif purpose == "car (used)":
        purpose = 1
    elif purpose == "furniture/equipment":
        purpose = 3
    elif purpose == "radio/television":
        purpose = 4
    elif purpose == "domestic appliances":
        purpose = 5
    elif purpose == "repairs":
        purpose = 6
    elif purpose == "education":
        purpose = 7
    elif purpose == "retraining":
        purpose = 8
    elif purpose == "business":
        purpose = 9
    elif purpose == "others":
        purpose = 2

    credit_amount = st.number_input("Credit amount:", min_value=0)

    savings_account_bonds = st.selectbox("Savings account/bonds:",
                                         ["< 100 DM", "100 - 500 DM", "500 - 1000 DM", "> 1000 DM",
                                          "unknown/no savings account"])

    if savings_account_bonds == "< 100 DM":
        savings_account_bonds = 0
    elif savings_account_bonds == "100 - 500 DM":
        savings_account_bonds = 1
    elif savings_account_bonds == "500 - 1000 DM":
        savings_account_bonds = 2
    elif savings_account_bonds == "> 1000 DM":
        savings_account_bonds = 3
    elif savings_account_bonds == "unknown/no savings account":
        savings_account_bonds = 4

    present_employment_since = st.selectbox("Present employment since:",
                                            ["unemployed", "< 1 year", "1 - 4 years", "4 - 7 years", "> 7 years"])

    if present_employment_since == "unemployed":
        present_employment_since = 0
    elif present_employment_since == "< 1 year":
        present_employment_since = 1
    elif present_employment_since == "1 - 4 years":
        present_employment_since = 2
    elif present_employment_since == "4 - 7 years":
        present_employment_since = 3
    elif present_employment_since == "> 7 years":
        present_employment_since = 4

    installment_rate = st.slider("Installment rate in percentage of disposable income:", min_value=0, max_value=100, value=50)

    personal_status_and_sex = st.selectbox("Personal status and sex:",
                                           ["male : divorced/separated", "female : divorced/separated/married",
                                            "male : single", "male : married/widowed",
                                            "female : single"])
    
    if personal_status_and_sex == "male : divorced/separated":
        personal_status_and_sex = 0
    elif personal_status_and_sex == "female : divorced/separated/married":
        personal_status_and_sex = 1
    elif personal_status_and_sex == "male : single":
        personal_status_and_sex = 2
    elif personal_status_and_sex == "male : married/widowed":
        personal_status_and_sex = 3
    elif personal_status_and_sex == "female : single":
        personal_status_and_sex = 4

    other_debtors_guarantors = st.selectbox("Other debtors/guarantors:", ["none", "co-applicant", "guarantor"])

    if other_debtors_guarantors == "none":
        other_debtors_guarantors = 0
    elif other_debtors_guarantors == "co-applicant":
        other_debtors_guarantors = 1
    elif other_debtors_guarantors == "guarantor":
        other_debtors_guarantors = 2

    present_residence_since = st.number_input("Present residence since:", min_value=0)

    property = st.selectbox("Property:", ["real estate", "building society savings agreement/life insurance",
                                          "car or other", "unknown/no property"])

    if property == "real estate":
        property = 0
    elif property == "building society savings agreement/life insurance":
        property = 1
    elif property == "car or other":
        property = 2
    elif property == "unknown/no property":
        property = 3

    age_in_years = st.number_input("Age in years:", min_value=18)

    other_installment_plans = st.selectbox("Other installment plans:", ["bank", "stores", "none"])

    if other_installment_plans == "bank":
        other_installment_plans = 0
    elif other_installment_plans == "stores":
        other_installment_plans = 1
    elif other_installment_plans == "none":
        other_installment_plans = 2

    housing = st.selectbox("Housing:", ["rent", "own", "for free"])

    if housing == "rent":
        housing = 0
    elif housing == "own":
        housing = 1
    elif housing == "for free":
        housing = 2

    num_existing_credits = st.number_input("Number of existing credits at this bank:", min_value=0)

    job = st.selectbox("Job:", ["unemployed/ unskilled  - non-resident", "unskilled - resident", 
                                "skilled employee / official","management/self-employed/highly qualified employee/officer"])

    if job == "unemployed/ unskilled  - non-resident":
        job = 0
    elif job == "unskilled - resident":
        job = 1
    elif job == "skilled employee / official":
        job = 2
    elif job == "management/self-employed/highly qualified employee/officer":
        job = 3

    num_people_maintenance = st.number_input("Number of people being liable to provide maintenance for:", min_value=0)

    telephone = st.selectbox("Telephone:", ["none", "yes, registered under the customers name"])

    if telephone == "none":
        telephone = 0
    elif telephone == "yes, registered under the customers name":
        telephone = 1

    foreign_worker = st.selectbox("Foreign worker:", ["yes", "no"])

    if foreign_worker == "yes":
        foreign_worker = 0
    elif foreign_worker == "no":
        foreign_worker = 1

    if st.button("Submit"):
        # Save the data into different variables
        data = [
            status_of_checking_account,
            duration_in_month,
            credit_history,
            purpose,
            credit_amount,
            savings_account_bonds,
            present_employment_since,
            installment_rate,
            personal_status_and_sex,
            other_debtors_guarantors,
            present_residence_since,
            property,
            age_in_years,
            other_installment_plans,
            housing,
            num_existing_credits,
            job,
            num_people_maintenance,
            telephone,
            foreign_worker
        ]
        # Convert data to numpy array and reshape for prediction
        data_array = np.array(data).reshape(1, -1)
        
        # Make a prediction using the loaded model
        prediction = model.predict(data_array)
        if prediction[0] == 1:
            st.title("You are eligible to take a loan from us")
        elif prediction[0] == 2:
            st.title("You are not eligible from taking a loan from us")

if __name__ == "__main__":
    main()
