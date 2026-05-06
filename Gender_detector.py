import streamlit as st
import numpy as np
import cv2
import joblib


st.set_page_config(layout='wide', page_title="Gender Detector", page_icon="🇮🇳")

gender_pca = joblib.load("gender_pca.pkl")
gender_model = joblib.load("gender_model.pkl")
if "page" not in st.session_state:
    st.session_state.page = "home"


if st.session_state.page == "home":
    st.markdown("""
        <style>
            
        .hero-section {
            background-image: url("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTSfoDV_k3slxDl3wTps9laJDXkGjv-9sElaQ&s");
            background-size: cover;
            background-position: center;
            padding: clamp(30px, 8vw, 80px);
            border-radius: 20px;
            text-align: center;
            color: white;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin-bottom: 2rem;
            width: 100%;
        }

        .main-heading {
            font-size: clamp(2rem, 10vw, 4.5rem);
            font-weight: 800;
            margin: 0;
            letter-spacing: 1px;
            text-shadow: 2px 2px 15px rgba(0,0,0,0.5);
        }
        
        .sub-text {
            font-size: clamp(1rem, 3vw, 1.5rem);
            opacity: 0.9;
            margin-top: 10px;
        }

        </style>
        
        <div class="hero-section">
            <h1 class="main-heading">Gender Detector Model</h1>
            <p class="sub-text">AI-Powered Gender Detector System</p>
        </div>
        """, unsafe_allow_html=True)

    with st.sidebar:
        st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRUwEQVo6w2qXCPUOotLA2-lPhMzPaJFPz6hQ&s")
        
        st.divider()
        
        st.title("🚀 About Project")
        st.info("This System uses Natural Language Processing (NLP) to automatically Detected Gender.")
        
        st.title("🛠️ Technical Stack")
        st.write("**Streamlit:** Framework for creating the interactive web interface.")
        st.write("**Scikit-Learn:** Used for training and implementing the Gender Detector Model.")
        st.write("**Numpy:** For mathematical operations and array processing.")
        st.write("**Joblib:** To load the pre-trained Machine Learning model.")
        
        st.divider()
        
        st.title("📞 Contact Us")
        st.success("📍 **AI Engineers @ DUCAT**")
        st.write("📧 **Email:** uditpal518@gmail.com")
        st.write("📱 **Phone:** +91 99999-88888")
        st.write("🌐 **Website:** www.ducatindia.com")
    if st.button("Check Gender"):
        st.session_state.page = "check gender"


if st.session_state.page == "check gender":
    st.title("📸 Gender Detection")

    if st.button("⬅️ Back"):
        st.session_state.page = "home"

    cam_img = st.camera_input("Take Photo")
    if cam_img is not None:
        face_model = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        byte_file = np.asarray(bytearray(cam_img.read()), dtype = np.uint8)
        frame = cv2.imdecode(byte_file,1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face = face_model.detectMultiScale(gray, 1.3, 5)

        if len(face) > 0:
            for (x,y,w,h) in face:
                face_img = frame[y:y+h, x:x+w]
                face_img = cv2.resize(face_img,(100, 100))
                gray_img = cv2.cvtColor(face_img,cv2.COLOR_BGR2GRAY)
                flat_img = gray_img.flatten().astype('float32') / 255.0

                x_pca = gender_pca.transform([flat_img])
                gender = gender_model.predict(x_pca)[0]

                st.success(f"Recognition Gender: {gender}")

        else:
            st.warning("Face not detect! Please Try Again")
    else:
        st.info("Click on Take Photo ")