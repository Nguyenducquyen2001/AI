import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd 
import numpy as np
import keras 
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
with st.sidebar:
    selected = option_menu(
        menu_title = "",
        options = ["Home"],
        icons = ["house"],
        default_index = 0,
        styles={
                "container": {"padding": "0!important", "background-color": "#fafafa"},
                "icon": {"color": "blue", "font-size": "25px"},
                "nav-link": {
                    "font-size": "25px",
                    "text-align": "left",
                    "margin": "0px",
                    "--hover-color": "#eee",
                },
                "nav-link-selected": {"background-color": "#DDE0E6"},
            }
    )       



# Xử lý tùy chọn được chọn
if selected == "Home":
    st.write("", 
         "<span style='font-size: 24px; font-family: Arial;color: blue; text-align: center;'>DỰ ĐOÁN CHỨNG KHOÁN SÀN CHỨNG KHOÁN TÂN VIỆT</span>",
         unsafe_allow_html=True)
    st.divider()
    st.write("dữ liệu được quy đinh theo front như sau:")
    img='anh_front_csv.png'
    st.image(img)
    st.write()
    st.divider()
    #load model h5
    model_path = 'save_model.h5'
    # Upload file CSV từ giao diện người dùng
    uploaded_file = st.file_uploader("Chọn file CSV", type=["csv"])

    # Kiểm tra nếu file đã được tải lên
    if uploaded_file is not None:
        # Đọc dữ liệu từ file CSV chỉ với hai cột "Ngày" và "Đóng cửa"
        data = pd.read_csv(uploaded_file, encoding='latin1', usecols=["ngay", "dong cua"])
        
        # Chuyển đổi cột ngày thành định dạng thời gian
        data['ngay'] = pd.to_datetime(data['ngay'], format='%d/%m/%Y')
        # Kiểm tra xem dữ liệu có ít nhất 50 ngày
        if len(data) >= 50:
            st.write(data.tail(5))
            # Chọn cột ngày và đóng cửa
            dataf = pd.DataFrame(data, columns=['ngay', 'dong cua'])
            dataf = dataf.set_index('ngay')

            # Chuẩn hóa dữ liệu
            sc = MinMaxScaler(feature_range=(0, 1))
            data_scaled = sc.fit_transform(dataf.values)

            # Lấy 50 ngày gần nhất
            recent_data = data_scaled[-50:, :]

            # Chia dữ liệu thành chuỗi 50 ngày
            x_test = []
            x_test.append(recent_data[-50:, 0])
            x_test = np.array(x_test)
            x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

            # Tải mô hình đã lưu
            final_model = load_model(model_path)

            # Dự đoán ngày tiếp theo
            y_test_predict = final_model.predict(x_test)
            y_test_predict = sc.inverse_transform(y_test_predict)

            next_day = dataf.index[-1] + pd.DateOffset(days=1)  # Lấy ngày tiếp theo sau ngày cuối cùng trong dữ liệu
            next_day_prediction = y_test_predict[-1][0]  # Giá trị dự đoán của ngày tiếp theo

            # Hiển thị kết quả dự đoán
            st.write("Ngày tiếp theo:", next_day.date())
            st.write("Kết quả dự đoán:", next_day_prediction)
        else:
            st.write("Dữ liệu không đủ 30 ngày.")
    else:
        st.write("Chưa tải lên file CSV.")
    # tạo khoản cách tới cuối trang
    for i in range(5):
        st.markdown("")
    st.divider()    
    # Chia trang thành hai cột
    col1, col2 = st.columns(2)

    # Phần thông tin liên lạc
    with col1:
        st.markdown("<div style='background-color: #f2f2f2; padding: 10px;'>"
                    "<h2>sinh viên thực hiện </h2>"
                    "<ul>"
                    "<li>Họ và tên: Nguyễn Đức Quyền</li>"
                    "<li>MSSV:20146148</li>"
                    "<li>Ngành: CNKT Cơ Điện Tử</li>"
                    "</ul>"
                    "</div>", unsafe_allow_html=True)

    # Phần thông tin tác giả
    with col2:
        st.markdown("<div style='background-color: #f2f2f2; padding: 10px;'>"
                    "<h2>thông tin liên lạc </h2>"
                    "<ul>"
                    "<li>SĐT: (+84)377247668</li>"
                    "<li>Email: quyen2001@gmail.com</li>"
                    "<li>Github: <a href='https://github.com/Nguyenducquyen2001/AI'>github.AI</a></li>"
                    "</ul>"
                    "</div>", unsafe_allow_html=True)
