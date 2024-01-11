import streamlit as st


# Информаиця о нашем датасете
def page_dataset():
    st.markdown("""
       ## Описание датасета Раунды CS:GO 
       **Файл датасета:** `c_go.csv`

       Данный датасет содержит информацию о различных раундах в игре CS:GO. Содержит следующие столбцы:

       - `time_left`: время, прошедшее после начала раунда.
       - `ct_score`: количество выигранных раундов спецназовцами.
       - `t_score`: количество выигранных раундов террористами.
       - `map`: карта, на которой играют команды.
       - `ct_health`: текущее общее здоровье спецназовцев.
       - `t_health`: текущее общее здоровье террористов.
       - `ct_armor`: количество купленных бронежелетов спецназовцами.
       - `t_armor`: количество купленных бронежелетов террористами.
       - `ct_money`: количество денежных единиц спецназовцев.
       - `t_money`: количество денежных единиц спецназовцев.
       - `ct_helmets`: количество купленных шлемов спецназовцами.
       - `t_helmets`: количество купленных шлемов террористами.
       - `ct_defuse_kits`: количество купленных наборов для обезвреживания бомбы.
       - `ct_players_alive`: количество живых спецназовцев.
       - `t_players_alive`: количество живых террористов.
       
       **Особенности предобработки данных:**
       - Конвертация категориальных признаков.
       - Удаление дубликатов.
       - Удаление строк с типом NaN.
       """)
