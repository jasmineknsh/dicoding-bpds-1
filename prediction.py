import pandas as pd
import joblib

def predict_and_save_single():
    # === Load Model dan Kolom Encoding ===
    model = joblib.load("model/xgboost_model.pkl")
    encoded_columns = joblib.load("model/encoded_columns.pkl")

    # === Data Baru (1 Karyawan) ===
    data_baru = pd.DataFrame([{
        'BusinessTravel': 'Travel_Rarely',
        'EducationField': 'Life Sciences',
        'JobRole': 'Research Scientist',
        'MaritalStatus': 'Single',
        'OverTime': 'Yes',
        'MonthlyIncome': 5000,
        'TotalWorkingYears': 5,
        'YearsAtCompany': 3,
        'YearsWithCurrManager': 2,
        'Age': 28,
        'JobSatisfaction': 4,
        'EnvironmentSatisfaction': 3,
        'NumCompaniesWorked': 2,
        'YearsSinceLastPromotion': 1
    }])

    important_categorical = ['BusinessTravel', 'EducationField', 'JobRole', 'MaritalStatus', 'OverTime']
    important_numerical = [
        'MonthlyIncome',
        'TotalWorkingYears',
        'YearsAtCompany',
        'YearsWithCurrManager',
        'Age',
        'JobSatisfaction',
        'EnvironmentSatisfaction',
        'NumCompaniesWorked',
        'YearsSinceLastPromotion'
    ]

    # === Encoding Kategorikal (One-Hot Encoding) ===
    X = data_baru[important_categorical + important_numerical]
    X_encoded = pd.get_dummies(X, columns=important_categorical, drop_first=True)

    for col in encoded_columns:
        if col not in X_encoded.columns:
            X_encoded[col] = 0

    X_encoded = X_encoded[encoded_columns]

    y_pred = model.predict(X_encoded)[0]
    y_proba = model.predict_proba(X_encoded)[0][1]

    result_df = data_baru.copy()
    result_df['PredictedAttrition'] = y_pred
    result_df['Probability'] = round(y_proba, 4)

    print("=== Hasil Prediksi ===")
    if y_pred == 1:
        print(f"🚨 Karyawan diprediksi AKAN keluar. Probabilitas: {y_proba:.2%}")
    else:
        print(f"✅ Karyawan diprediksi TIDAK AKAN keluar. Probabilitas keluar: {y_proba:.2%}")

    return result_df

predict_and_save_single()
