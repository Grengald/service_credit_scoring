import joblib
import streamlit as st
import numpy as np

st.title("Кредитная карта Premium")

@st.cache_resource
def load_model(path="model.pkl"):
    return joblib.load(path)

model = load_model("model.pkl")

with st.form("Подать заявку"):
    age = st.number_input("Ваш возраст", min_value=18, value=30)
    income = st.number_input("Ваш доход в тысячах рублей", min_value=0.0, value=50.0)
    education = st.checkbox("У меня есть высшее образование")
    work = st.checkbox("У меня есть стабильная работа")
    car = st.checkbox("У меня есть автомобиль")
    submit = st.form_submit_button("Подать заявку")

if submit:
    features = [int(age), float(income), int(education), int(work), int(car)]
    try:
        pred = model.predict([features])[0]
        approved = bool(pred)
        st.success(f"Решение: {'Одобрено' if approved else 'Отказ'}")
    except Exception as e:
        st.error(f"Ошибка предсказания: {e}")
