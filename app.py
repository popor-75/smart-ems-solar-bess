 
import glob
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import streamlit as st

st.set_page_config(
    page_title="Smart EMS - Solar PV & BESS",
    page_icon="⚡",
    layout="wide",
)


@st.cache_data
def load_ems_data(base_dir):
  solar_folder = os.path.join(base_dir, "solar_4yr")
  weather_file = os.path.join(
      base_dir, "weather1_4yr", "open-meteo-13.88N100.43E7m.csv"
  )

 # 1. อ่านไฟล์โซลาร์จากโฟลเดอร์ solar_4yr หรือ solar_4yr_* ทั้งหมด
  solar_files = sorted(
      glob.glob(os.path.join(base_dir, "solar_4yr*", "**", "*.xlsx"), recursive=True)
  )
  solar_list = []
  for f in solar_files:
    try:
      temp = pd.read_excel(f, sheet_name="Sheet1", header=1)
      temp["source_file"] = os.path.basename(f)
      solar_list.append(temp)
    except Exception:
      pass

  if solar_list:
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
    s_num = solar_df.select_dtypes(include=[np.number]).copy()
    s_by_date = s_num.copy()
    s_by_date["date_key"] = s_by_date.index.date
    s_daily_mean = s_by_date.groupby("date_key").mean(numeric_only=True)
  else:
    s_daily_mean = pd.DataFrame()
  # 2. อ่านไฟล์สภาพอากาศ (ใช้คอลัมน์จริงตามที่ตรวจสอบ)
  weather_df = pd.read_csv(weather_file, skiprows=3)
  weather_df["time"] = pd.to_datetime(weather_df["time"], errors="coerce")
  weather_df = (
      weather_df.dropna(subset=["time"])
      .sort_values("time")
      .drop_duplicates(subset=["time"])
      .set_index("time")
  )

  # 3. สร้างตารางเวลาทุก 5 นาที (2022-2025)
  grid_5min = pd.date_range(
      start="2022-01-01 00:00:00", end="2025-12-31 23:55:00", freq="5min"
  )
  df = pd.DataFrame(index=grid_5min)
  df.index.name = "timestamp_5min"

  # อินเตอร์โพเลตข้อมูลสภาพอากาศรายชั่วโมง -> ทุก 5 นาที
  w_num = weather_df.apply(pd.to_numeric, errors="coerce")
  w_5min = w_num.reindex(grid_5min).interpolate(method="time").ffill().bfill()

  # จับคู่ข้อมูลโซลาร์ (ถ้าเป็นข้อมูลรายวัน ให้กระจายค่าเฉลี่ย/ค่าตามวันนั้น)
  s_num = solar_df.select_dtypes(include=[np.number]).copy()
  s_by_date = s_num.copy()
  s_by_date["date_key"] = s_by_date.index.date
  s_daily_mean = s_by_date.groupby("date_key").mean(numeric_only=True)

  df["date_key"] = df.index.date
  df = pd.concat([df, w_5min], axis=1)

  # เติมข้อมูลโซลาร์รายวันลงในแต่ละวัน
if not s_daily_mean.empty:
  for col in s_daily_mean.columns:
    df[f"solar_{col}"] = df["date_key"].map(s_daily_mean[col])

  ghi_col = "shortwave_radiation (W/m²)"
  temp_col = "temperature_2m (°C)"

  # คำนวณกำลังผลิต PV (kW) ตามสมการบทที่ 2.1 (P_dc_rated = 4.95 kWp)
  p_rated = 4.95
  gamma_p = -0.004
  g_val = df[ghi_col].fillna(0).clip(lower=0)
  t_val = df[temp_col].fillna(25.0)
  df["P_pv_kW"] = (
      p_rated * (g_val / 1000.0) * (1.0 + gamma_p * (t_val - 25.0))
  ).clip(lower=0, upper=p_rated)

  # โหลดอาคารสมมติฐานเบื้องต้น
  hours = df.index.hour + df.index.minute / 60.0
  base_load = 1.2 + 1.5 * np.exp(-0.5 * ((hours - 14.0) / 3.5) ** 2)
  df["P_load_kW"] = np.round(base_load, 3)

  return df, ghi_col, temp_col, len(solar_files)


def run_ems_simulation(
    df_sub, e_rated, p_max, soc_min, soc_max, eta_c, eta_d, soc_init, tou_on, tou_off
):
  dt = 5.0 / 60.0
  n = len(df_sub)
  p_pv = df_sub["P_pv_kW"].values
  p_load = df_sub["P_load_kW"].values
  hours = df_sub.index.hour.values
  weekdays = df_sub.index.weekday.values

  soc = np.zeros(n)
  p_charge = np.zeros(n)
  p_discharge = np.zeros(n)
  p_grid_import = np.zeros(n)
  p_grid_export = np.zeros(n)
  tariff_rate = np.zeros(n)

  curr_soc = soc_init
  for i in range(n):
    is_on_peak = (weekdays[i] < 5) and (9 <= hours[i] < 22)
    tariff_rate[i] = tou_on if is_on_peak else tou_off

    net_p = p_pv[i] - p_load[i]
    if net_p > 0:
      max_chg_energy = max(0.0, (soc_max - curr_soc) * e_rated)
      max_chg_power = min(p_max, max_chg_energy / (eta_c * dt))
      chg = min(net_p, max_chg_power)
      exp = max(0.0, net_p - chg)

      curr_soc += (eta_c * chg * dt) / e_rated
      p_charge[i] = chg
      p_grid_export[i] = exp
    else:
      deficit = -net_p
      if curr_soc > soc_min:
        max_dis_energy = max(0.0, (curr_soc - soc_min) * e_rated)
        max_dis_power = min(p_max, (max_dis_energy * eta_d) / dt)
        dis = min(deficit, max_dis_power)
        imp = max(0.0, deficit - dis)

        curr_soc -= (dis * dt) / (eta_d * e_rated)
        p_discharge[i] = dis
        p_grid_import[i] = imp
      else:
        p_grid_import[i] = deficit

    soc[i] = np.clip(curr_soc, soc_min, soc_max)

  res = df_sub.copy()
  res["SOC_%"] = soc * 100.0
  res["P_charge_kW"] = p_charge
  res["P_discharge_kW"] = p_discharge
  res["P_grid_import_kW"] = p_grid_import
  res["P_grid_export_kW"] = p_grid_export
  res["Tariff_THB"] = tariff_rate
  return res


st.title("⚡ ระบบบริหารจัดการพลังงานไฟฟ้าอัจฉริยะ (Smart EMS)")
st.caption(
    "ต้นแบบระบบควบคุม Solar PV (4.95 kWp) ร่วมกับ BESS | ฐานข้อมูลย้อนหลัง 4 ปี (2022–2025) ความละเอียด 5 นาที"
)

base_dir = os.path.dirname(os.path.abspath(__file__))

with st.sidebar:
  st.header("⚙️ ตั้งค่าพารามิเตอร์ระบบ")
  st.info("ค่า BESS ด้านล่างเป็นค่าสมมติฐานเริ่มต้น")

  e_rated = st.number_input("ความจุแบตเตอรี่ E_rated (kWh)", 1.0, 100.0, 10.0, 1.0)
  p_max = st.number_input("กำลังชาร์จ/คายประจุสูงสุด (kW)", 1.0, 50.0, 5.0, 0.5)
  soc_min_pct = st.slider("SOC ต่ำสุดปลอดภัย (%)", 5, 40, 20)
  soc_max_pct = st.slider("SOC สูงสุดปลอดภัย (%)", 70, 100, 95)
  soc_init_pct = st.slider("SOC เริ่มต้น (%)", 10, 90, 50)
  eta_pct = st.slider("ประสิทธิภาพชาร์จ/คายประจุ (%)", 80, 100, 95)

  st.divider()
  st.subheader("💰 อัตราค่าไฟฟ้า TOU (บาท/หน่วย)")
  tou_on = st.number_input("On-Peak (09:00-22:00 จ.-ศ.)", 1.0, 10.0, 5.79, 0.05)
  tou_off = st.number_input("Off-Peak / วันหยุด", 1.0, 10.0, 2.63, 0.05)

with st.spinner("กำลังโหลดและประมวลผลข้อมูล 4 ปี ทุก 5 นาที..."):
  df_all, ghi_col, temp_col, n_solar_files = load_ems_data(base_dir)

st.sidebar.success(f"โหลดไฟล์โซลาร์สำเร็จ {n_solar_files} ไฟล์")

col_f1, col_f2 = st.columns([1, 2])
with col_f1:
  selected_year = st.selectbox("เลือกปีที่ต้องการวิเคราะห์", [2022, 2023, 2024, 2025], index=3)
with col_f2:
  df_year = df_all[df_all.index.year == selected_year]
  avail_dates = sorted(df_year.index.date)
  selected_date = st.date_input(
      "เลือกวันที่เจาะลึกราย 5 นาที",
      value=avail_dates[min(15, len(avail_dates) - 1)],
      min_value=avail_dates[0],
      max_value=avail_dates[-1],
  )

df_day = df_year[df_year.index.date == selected_date]
sim_day = run_ems_simulation(
    df_day,
    e_rated,
    p_max,
    soc_min_pct / 100.0,
    soc_max_pct / 100.0,
    eta_pct / 100.0,
    eta_pct / 100.0,
    soc_init_pct / 100.0,
    tou_on,
    tou_off,
)

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 1. วิเคราะห์พลังงาน (Energy Analysis)",
    "🤖 2. พยากรณ์ด้วย AI (AI Prediction)",
    "🔋 3. จัดการแบตเตอรี่ (Battery & SOC)",
    "🧠 4. สมองกลจัดสรรพลังงาน & ความคุ้มค่า",
])

with tab1:
  st.subheader(f"สรุปผลการผลิตไฟฟ้าพลังงานแสงอาทิตย์ วันที่ {selected_date}")
  dt = 5.0 / 60.0
  total_pv_kwh = (sim_day["P_pv_kW"] * dt).sum()
  peak_pv_kw = sim_day["P_pv_kW"].max()
  avg_ghi = sim_day[ghi_col].mean()
  avg_temp = sim_day[temp_col].mean()

  m1, m2, m3, m4 = st.columns(4)
  m1.metric("พลังงานผลิตรวมจำลอง (PV Yield)", f"{total_pv_kwh:.2f} kWh")
  m2.metric("กำลังผลิตสูงสุดจำลอง (Peak Power)", f"{peak_pv_kw:.2f} kW")
  m3.metric("ความเข้มแสงเฉลี่ย (Shortwave Radiation)", f"{avg_ghi:.1f} W/m²")
  m4.metric("อุณหภูมิเฉลี่ย (2m)", f"{avg_temp:.1f} °C")

  fig1 = go.Figure()
  fig1.add_trace(
      go.Scatter(
          x=sim_day.index,
          y=sim_day["P_pv_kW"],
          name="กำลังผลิตโซลาร์ P_pv (kW)",
          fill="tozeroy",
          line=dict(color="#f59e0b"),
      )
  )
  fig1.add_trace(
      go.Scatter(
          x=sim_day.index,
          y=sim_day["P_load_kW"],
          name="ความต้องการใช้ไฟ P_load (kW)",
          line=dict(color="#3b82f6", dash="dash"),
      )
  )
  fig1.update_layout(
      title="โปรไฟล์กำลังผลิตไฟฟ้าโซลาร์เทียบกับโหลด (ทุก 5 นาที)",
      xaxis_title="เวลา",
      yaxis_title="กำลังไฟฟ้า (kW)",
      height=400,
  )
  st.plotly_chart(fig1, width="stretch")

  uploaded = st.file_uploader(
      "อัปโหลดไฟล์ข้อมูล FusionSolar (.xlsx) เพิ่มเติมเพื่อตรวจสอบ", type=["xlsx"]
  )
  if uploaded:
    up_df = pd.read_excel(uploaded, sheet_name="Sheet1", header=1)
    st.dataframe(up_df.head(20), width="stretch")

with tab2:
  st.subheader("โมเดลพยากรณ์กำลังผลิตไฟฟ้าล่วงหน้า 24 ชั่วโมง (Random Forest)")
  df_ml = df_year.copy()
  df_ml["hour"] = df_ml.index.hour
  df_ml["minute"] = df_ml.index.minute
  df_ml["dayofyear"] = df_ml.index.day_of_year
  features = ["hour", "minute", "dayofyear", ghi_col, temp_col]

  train_df = df_ml[df_ml.index.date != selected_date].dropna(
      subset=features + ["P_pv_kW"]
  )
  test_df = df_ml[df_ml.index.date == selected_date].copy()
  test_df[features] = test_df[features].apply(pd.to_numeric, errors="coerce").fillna(0)

  if len(train_df) > 100 and len(test_df) > 0:
    rf = RandomForestRegressor(
        n_estimators=40, max_depth=10, random_state=42, n_jobs=-1
    )
    rf.fit(train_df[features], train_df["P_pv_kW"])
    y_pred = rf.predict(test_df[features])
    y_true = test_df["P_pv_kW"].values

    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    c1, c2 = st.columns(2)
    c1.metric("ความแม่นยำ R² Score", f"{r2:.4f}")
    c2.metric("ค่าคลาดเคลื่อนเฉลี่ย (MAE)", f"{mae:.4f} kW")

    fig2 = go.Figure()
    fig2.add_trace(
        go.Scatter(
            x=test_df.index,
            y=y_true,
            name="ค่าจริง (Actual P_pv)",
            line=dict(color="#10b981"),
        )
    )
    fig2.add_trace(
        go.Scatter(
            x=test_df.index,
            y=y_pred,
            name="ค่าพยากรณ์ AI (Predicted)",
            line=dict(color="#ef4444", dash="dot"),
        )
    )
    fig2.update_layout(
        title="เปรียบเทียบค่ากำลังผลิตจริง vs ค่าพยากรณ์ AI ราย 5 นาที",
        xaxis_title="เวลา",
        yaxis_title="กำลังไฟฟ้า (kW)",
        height=400,
    )
    st.plotly_chart(fig2, width="stretch")

with tab3:
  st.subheader("สถานะพลังงานคงเหลือและพฤติกรรมการชาร์จ-คายประจุแบตเตอรี่")
  b1, b2, b3 = st.columns(3)
  b1.metric("พลังงานชาร์จเข้าสะสม", f"{(sim_day['P_charge_kW'] * dt).sum():.2f} kWh")
  b2.metric("พลังงานคายประจุสะสม", f"{(sim_day['P_discharge_kW'] * dt).sum():.2f} kWh")
  b3.metric("SOC สิ้นสุดวัน", f"{sim_day['SOC_%'].iloc[-1]:.1f} %")

  fig3 = go.Figure()
  fig3.add_trace(
      go.Scatter(
          x=sim_day.index,
          y=sim_day["SOC_%"],
          name="State of Charge (SOC %)",
          line=dict(color="#8b5cf6", width=3),
      )
  )
  fig3.add_hline(y=soc_max_pct, line_dash="dash", line_color="red", annotation_text="SOC Max")
  fig3.add_hline(y=soc_min_pct, line_dash="dash", line_color="orange", annotation_text="SOC Min")
  fig3.update_layout(
      title="กราฟการเปลี่ยนแปลงระดับพลังงานในแบตเตอรี่ (SOC %)",
      xaxis_title="เวลา",
      yaxis_title="SOC (%)",
      yaxis_range=[0, 105],
      height=380,
  )
  st.plotly_chart(fig3, width="stretch")

with tab4:
  st.subheader("ผลการจัดสรรพลังงาน (Power Dispatch) และความคุ้มค่าทางเศรษฐศาสตร์")
  cost_before = (sim_day["P_load_kW"] * dt * sim_day["Tariff_THB"]).sum()
  cost_after = (sim_day["P_grid_import_kW"] * dt * sim_day["Tariff_THB"]).sum()
  saving_thb = cost_before - cost_after
  saving_pct = (saving_thb / cost_before * 100.0) if cost_before > 0 else 0.0

  e1, e2, e3 = st.columns(3)
  e1.metric("ค่าไฟฟ้าก่อนใช้ระบบ (Grid Only)", f"{cost_before:.2f} บาท")
  e2.metric("ค่าไฟฟ้าหลังใช้ Smart EMS", f"{cost_after:.2f} บาท")
  e3.metric("ประหยัดค่าไฟฟ้าได้", f"{saving_thb:.2f} บาท ({saving_pct:.1f}%)")

  fig4 = go.Figure()
  fig4.add_trace(
      go.Bar(
          x=sim_day.index,
          y=sim_day["P_charge_kW"],
          name="ชาร์จเข้าแบต (P_charge)",
          marker_color="#22c55e",
      )
  )
  fig4.add_trace(
      go.Bar(
          x=sim_day.index,
          y=-sim_day["P_discharge_kW"],
          name="คายประจุช่วยโหลด (P_discharge)",
          marker_color="#a855f7",
      )
  )
  fig4.add_trace(
      go.Scatter(
          x=sim_day.index,
          y=sim_day["P_grid_import_kW"],
          name="ซื้อไฟจากกริด (P_grid_import)",
          line=dict(color="#ef4444"),
      )
  )
  fig4.update_layout(
      title="สมดุลการไหลของพลังงานราย 5 นาที (Power Dispatch)",
      xaxis_title="เวลา",
      yaxis_title="กำลังไฟฟ้า (kW)",
      barmode="relative",
      height=400,
  )
  st.plotly_chart(fig4, width="stretch")
