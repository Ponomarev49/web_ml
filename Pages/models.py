import streamlit as st
import pandas as pd
from keras.models import load_model
from sklearn.model_selection import train_test_split
import pickle
import author
import sklearn.metrics._dist_metrics

X = author.data.drop(columns=['bomb_planted'])
y = author.data['bomb_planted']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Страница с инференсом моделей
def page_predictions():
    st.title("Предсказания моделей машинного обучения")

    # Виджет для загрузки файла
    uploaded_file = st.file_uploader("Загрузите ваш CSV файл", type="csv")

    # Интерактивный ввод данных, если файл не загружен
    if uploaded_file is None:
        st.subheader("Введите данные для предсказания:")

        # Интерактивные поля для ввода данных
        predict_input = pd.DataFrame(
            {'time_left': [st.number_input("time_left", min_value=1.0, max_value=180.0, step=5.0, value=90.0)],
             'ct_score': [author.data['ct_score'].mode()[0]],
             't_score': [author.data['t_score'].mode()[0]],
             'map': [author.data['map'].mode()[0]],
             'ct_health': [st.number_input("ct_health", min_value=1.0, max_value=500.0, step=20.0, value=100.0)],
             't_health': [st.number_input("t_health", min_value=1.0, max_value=500.0, step=20.0, value=100.0)],
             'ct_armor': [author.data['ct_armor'].mean()],
             't_armor': [author.data['t_armor'].mean()],
             'ct_money': [author.data['ct_money'].mode()[0]],
             't_money': [author.data['t_money'].mode()[0]],
             'ct_helmets': [author.data['ct_helmets'].mode()[0]],
             't_helmets': [author.data['t_helmets'].mode()[0]],
             'ct_defuse_kits': [author.data['ct_defuse_kits'].mode()[0]],
             'ct_players_alive': [st.number_input("ct_players_alive", min_value=1.0, max_value=5.0, step=1.0,
                                                  value=1.0)],
             't_players_alive': [st.number_input("t_players_alive", min_value=1.0, max_value=5.0, step=1.0,
                                                 value=1.0)]})
        if 'Unnamed: 0' in predict_input.columns:
            predict_input = predict_input.drop(['Unnamed: 0'], axis=1)
        for column in predict_input.columns:
            predict_input[column] = predict_input[column].astype(float)

        st.write(predict_input)

        if st.button('Сделать предсказание'):
            with open('Models/knn.pkl', 'rb') as file:
                knn = pickle.load(file)
            with open('Models/bagging.pkl', 'rb') as file:
                bagging_model = pickle.load(file)
            with open('Models/grad_boost.pkl', 'rb') as file:
                gradient_model = pickle.load(file)
            with open('Models/kmeans.pkl', 'rb') as file:
                kmeans_model = pickle.load(file)
            with open('Models/stacking.pkl', 'rb') as file:
                stacking_model = pickle.load(file)
            nn_model = load_model('Models/bin_class.h5')

            st.header("KNN:")
            pred = []
            knn_pred = knn.predict(predict_input)[0]
            pred.append(int(knn_pred))
            st.write(f"{knn_pred}")

            st.header("bagging:")
            bagging_pred = bagging_model.predict(predict_input)[0]
            pred.append(int(bagging_pred))
            st.write(f"{bagging_pred}")

            st.header("gradient:")
            gradient_pred = gradient_model.predict(predict_input)[0]
            pred.append(int(gradient_pred))
            st.write(f"{gradient_pred}")

            st.header("Binary Classification:")
            nn_pred = int(nn_model.predict(predict_input, verbose=None))
            pred.append(nn_pred)
            st.write(f"{nn_pred}")

            st.header("Stacking:")
            stacking_pred = stacking_model.predict(predict_input)[0]
            pred.append(int(stacking_pred))
            st.write(f"{stacking_model.predict(predict_input)[0]}")

            st.header("kmeans:")
            kmeans_pred = kmeans_model.predict(predict_input)[0]
            pred.append(int(kmeans_pred))
            st.write(f"{bagging_pred}")

# Загрузка моделей
# def models():
#     model1 = pickle.load(open('Models/knn.pkl', 'rb'))
#     model2 = pickle.load(open('Models/kmeans.pkl', 'rb'))
#     model3 = pickle.load(open('Models/grad_boost.pkl', 'rb'))
#     model4 = pickle.load(open('Models/bagging.pkl', 'rb'))
#     model5 = pickle.load(open('Models/stacking.pkl', 'rb'))
#     model6 = load_model('Models/bin_class.h5')
#     return model1, model2, model3, model4, model5, model6
