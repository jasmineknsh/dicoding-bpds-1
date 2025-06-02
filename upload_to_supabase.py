import pandas as pd
from sqlalchemy import create_engine

# === Supabase PostgreSQL URL ===
URL = "postgresql://postgres.uwwsbuzkkfjgrbngqcat:jayamajujaya@aws-0-ap-southeast-1.pooler.supabase.com:6543/postgres"
engine = create_engine(URL)

# === Daftar file dan nama tabel ===
files_and_tables = {
    "data/employee.csv": "employee",
    "data/hasil_prediksi.csv": "hasil_prediksi",
    "data/top_10_important_features.csv": "top_10_important_features"
}

# === Proses upload tiap file ke tabel ===
for file_path, table_name in files_and_tables.items():
    df = pd.read_csv(file_path, encoding='windows-1252')
    df.to_sql(table_name, engine, index=False, if_exists='replace')
    print(f"✅ Berhasil upload ke tabel '{table_name}'")

print("🚀 Semua file berhasil diunggah ke Supabase!")
