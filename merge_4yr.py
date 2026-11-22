import glob
import os
import numpy as np
import pandas as pd


def find_project_root():
  user_profile = os.path.expanduser("~")
  candidates = [
      os.path.join(
          user_profile, "Start backup", "Documents", "smart_Ems_Project01"
      ),
      os.path.join(user_profile, "Documents", "smart_Ems_Project01"),
      os.path.join(
          r"C:\Users\TU-PINE\Documents", "smart_Ems_Project01"
      ),
      os.path.join(
          r"C:\Users\Maneerat\Documents", "smart_Ems_Project01"
      ),
  ]
  for path in candidates:
    if os.path.exists(path):
      return path
  return candidates[0]


# ==========================================
# 1. กำหนดพาธจากโฟลเดอร์ smart_Ems_Project01
# ==========================================
project_dir = find_project_root()
solar_folder = os.path.join(project_dir, "solar_4yr")
weather_file = os.path.join(
    project_dir, "weather1_4yr", "open-meteo-13.88N100.43E7m.csv"
)
output_dir = os.path.join(project_dir, "output_4yr_5min")
os.makedirs(output_dir, exist_ok=True)

print("โฟลเดอร์โปรเจกต์:", project_dir)
print("โฟลเดอร์โซลาร์:", solar_folder)
print("ไฟล์สภาพอากาศ:", weather_file)

# ==========================================
# 2. อ่านไฟล์โซลาร์ทั้งหมดและเรียงตามเวลาจริง
# ==========================================
solar_files = sorted(
    glob.glob(os.path.join(solar_folder, "**", "*.xlsx"), recursive=True)
)
solar_list = []

for f in solar_files:
  temp = pd.read_excel(f, sheet_name="Sheet1", header=1)
  temp["source_file"] = os.path.basename(f)
  solar_list.append(temp)

solar_df = pd.concat(solar_list, ignore_index=True)
solar_df["Statistical Period"] = pd.to_datetime(
    solar_df["Statistical Period"], errors="coerce"
)
solar_df = (
    solar_df.dropna(subset=["Statistical Period"])
    .sort_values("Statistical Period")
    .drop_duplicates(subset=["Statistical Period"])
    .set_index("Statistical Period")
)

# ==========================================
# 3. อ่านไฟล์สภาพอากาศย้อนหลัง 4 ปี (2022-2025)
# ==========================================
weather_df = pd.read_csv(weather_file, skiprows=3)
time_col = weather_df.columns[0]
weather_df[time_col] = pd.to_datetime(weather_df[time_col], errors="coerce")
weather_df = (
    weather_df.dropna(subset=[time_col])
    .sort_values(time_col)
    .drop_duplicates(subset=[time_col])
    .set_index(time_col)
)

# ==========================================
# 4. สร้างตารางเวลาทุก 5 นาที (2022-2025) และรวมข้อมูล
# ==========================================
grid_5min = pd.date_range(
    start="2022-01-01 00:00:00", end="2025-12-31 23:55:00", freq="5min"
)
df_5min = pd.DataFrame(index=grid_5min)
df_5min.index.name = "timestamp_5min"

weather_5min = weather_df.reindex(grid_5min).interpolate(method="time")
solar_5min = solar_df.reindex(grid_5min)
df_5min = pd.concat([solar_5min, weather_5min], axis=1)

# ==========================================
# 5. พารามิเตอร์แบตเตอรี่สมมติฐานเริ่มต้น (รออัปเดตในเทอม 2)
# ==========================================
dt_hr = 5.0 / 60.0
e_rated_kwh = 10.0
p_max_kw = 5.0
soc_min = 0.20
soc_max = 0.90
eta_c = 0.95
eta_d = 0.95
soc_init = 0.50

p_batt_kw = np.zeros(len(df_5min))
soc_series = np.zeros(len(df_5min))
soc_curr = soc_init

for i, p_val in enumerate(p_batt_kw):
  p_c = float(np.clip(p_val, -p_max_kw, p_max_kw))
  if p_c >= 0:
    delta_soc = (p_c * dt_hr * eta_c) / e_rated_kwh
  else:
    delta_soc = (p_c * dt_hr) / (eta_d * e_rated_kwh)
  soc_curr = float(np.clip(soc_curr + delta_soc, soc_min, soc_max))
  soc_series[i] = soc_curr

df_5min["P_batt_kW_sim"] = p_batt_kw
df_5min["SOC_sim"] = soc_series

# ==========================================
# 6. บันทึกไฟล์ผลลัพธ์ 4 ปี ทุก 5 นาที
# ==========================================
out_parquet = os.path.join(output_dir, "ems_4yr_5min.parquet")
out_csv = os.path.join(output_dir, "ems_4yr_5min.csv")

df_5min.to_parquet(out_parquet)
df_5min.to_csv(out_csv)

print("เสร็จแล้ว! อ่านไฟล์โซลาร์ได้:", len(solar_files), "ไฟล์")
print("จำนวนแถวทั้งหมด (ทุก 5 นาที):", f"{len(df_5min):,}")
print("บันทึกไฟล์ผลลัพธ์ไว้ที่:", out_parquet)
