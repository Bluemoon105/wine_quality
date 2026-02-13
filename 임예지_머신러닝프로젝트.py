# 와인 품질 예측 Streamlit (아주 기본형)
import streamlit as st
import numpy as np
import joblib

# 1) 제목/설명
st.title('🍷 와인 품질 예측 시스템')
st.write('화학적 특성 값을 입력하고 "예측하기"를 눌러 품질 점수를 확인하세요.👽')

# 2) 사용자 입력 (와인 특성 7개)
alcohol = st.slider('alcohol(알코올 도수)', 8.0, 15.0, 9.4, 0.1)
volatile_acidity = st.slider('volatile acidity(휘발성 산도)', 0.10, 1.50, 0.70, 0.01)
citric_acid = st.slider('citric acid(구연산)', 0.00, 1.00, 0.00, 0.01)
sulphates = st.slider('sulphates(황산염)', 0.20, 2.00, 0.56, 0.01)
density = st.slider('density(밀도)', 0.9900, 1.0050, 0.9978, 0.0001)
total_sulfur_dioxide = st.slider('total sulfur dioxide(총 이산화황)', 6, 300, 34, 1)
fixed_acidity = st.slider('fixed acidity (고정 산도)', 4.0, 16.0, 7.4, 0.1)



# 3) 예측 버튼
if st.button('예측하기🎉'):
    model = joblib.load('wine_quality_model.pkl')

    X = np.array([[ 
        alcohol,
        volatile_acidity,
        citric_acid,
        sulphates,
        density,
        total_sulfur_dioxide,
        fixed_acidity
    ]])

    y_pred = model.predict(X)[0]

    # 결과 출력 (회귀→ 점수 표시)
    st.write(f'예측된 와인 품질 점수: **{y_pred:.2f}** 👽')
    st.balloons()  # 축하 풍선 효과
    st.snow()
    