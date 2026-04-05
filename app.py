import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go

# 1. Cấu hình
st.set_page_config(page_title="AI Structural Analysis", layout="wide")
st.title("📊 Phân Tích Nội Lực Sàn Thông Minh (AI Heatmap)")

@st.cache_resource
def load_all():
    data = joblib.load('trained_model.pkl')
    return data['model'], data['encoders'], data['features']

model, encoders, feature_names = load_all()

# 2. Sidebar Input
st.sidebar.header("📥 Thông số thiết kế")
L = st.sidebar.number_input("Chiều dài sàn (m)", value=8.0)
B = st.sidebar.number_input("Chiều rộng sàn (m)", value=6.0)
H = st.sidebar.number_input("Bề dày sàn (m)", value=0.2)
mac_bt = st.sidebar.selectbox("Cấp độ bền", encoders['Cap_do_ben'].classes_)
q = st.sidebar.number_input("Tải trọng q (kN/m2)", value=12.0)

# 3. Logic tự động xác định loại kết cấu
st.subheader("🧐 Phân tích sơ bộ từ AI & Tiêu chuẩn")
col_info1, col_info2, col_info3 = st.columns(3)

# Xác định loại biên (Ví dụ minh họa dựa trên tỷ lệ nhịp)
loai_bien = "ngam_4_canh" if L/B < 1.5 else "ke_4_canh" 
# Xác định loại bê tông dựa trên nhịp
loai_bt = "PT (Dự ứng lực)" if max(L, B) > 7.0 else "RC (Bê tông thường)"
he_thong = "PT" if max(L, B) > 7.0 else "RC"

with col_info1:
    st.info(f"**Hệ thống đề xuất:** {loai_bt}")
with col_info2:
    st.info(f"**Loại biên dự kiến:** {loai_bien}")
with col_info3:
    st.info(f"**Tỷ lệ nhịp L/B:** {L/B:.2f}")

# 4. Xử lý dự đoán lưới điểm (Heatmap)
# Giả sử sàn được chia thành lưới 10x10 phần tử
num_points = 10 
points = []

# Tạo dữ liệu ảo cho 100 điểm trên sàn để AI dự đoán
for i in range(1, 101):
    points.append([
        he_thong, loai_bien, mac_bt, 30.0, L, B, H, q, 500.0, 0.1, i
    ])

df_mesh = pd.DataFrame(points, columns=feature_names)

# Encode dữ liệu chữ
for col in ['He_thong', 'Loai_san', 'Cap_do_ben']:
    df_mesh[col] = encoders[col].transform(df_mesh[col])

# Dự đoán toàn bộ lưới
all_predictions = model.predict(df_mesh)
m11_values = all_predictions[:, 0].reshape(num_points, num_points)
m22_values = all_predictions[:, 1].reshape(num_points, num_points)

# 5. Vẽ biểu đồ Heatmap tương tác với Plotly
st.subheader("🎨 Biểu đồ phân bố Momen (Di chuột để xem giá trị)")

tab1, tab2 = st.tabs(["Momen M11", "Momen M22"])

with tab1:
    fig1 = go.Figure(data=go.Heatmap(
        z=m11_values,
        x=np.linspace(0, L, num_points),
        y=np.linspace(0, B, num_points),
        colorscale='Viridis',
        hovertemplate='X: %{x}m<br>Y: %{y}m<br>M11: %{z:.2f} kNm<extra></extra>',
        zsmooth='best'  # <--- THÊM DÒNG NÀY ĐỂ LÀM MỊN HEATMAP
    ))
    fig1.update_layout(title="Phân bố Momen M11", xaxis_title="Chiều dài (m)", yaxis_title="Chiều rộng (m)")
    st.plotly_chart(fig1, use_container_width=True)

with tab2:
    fig2 = go.Figure(data=go.Heatmap(
        z=m22_values,
        x=np.linspace(0, L, num_points),
        y=np.linspace(0, B, num_points),
        colorscale='Hot',
        hovertemplate='X: %{x}m<br>Y: %{y}m<br>M22: %{z:.2f} kNm<extra></extra>',
        zsmooth='best'  # <--- THÊM DÒNG NÀY ĐỂ LÀM MỊN HEATMAP
    ))
    fig2.update_layout(title="Phân bố Momen M22", xaxis_title="Chiều dài (m)", yaxis_title="Chiều rộng (m)")
    st.plotly_chart(fig2, use_container_width=True)
