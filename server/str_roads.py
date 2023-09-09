#!/usr/bin/env python3

import folium
import streamlit as st
from folium.plugins import Draw
import pandas as pd
from streamlit_folium import st_folium
import cv2
import base64
import io
from PIL import Image
from datetime import datetime
from ultralytics import YOLO
import csv
from io import BytesIO
import time

im = Image.open('logo.jpg')
st.sidebar.image(im)

st.write("## Добро пожаловать в сервис мониторинга дорожного покрытия! 👋")
st.markdown(
        """ **👈 Выберите действие сбоку слева**, чтобы посмотреть возможности сервиса! """
    )



login = st.sidebar.text_input('Логин', '')
password = st.sidebar.text_input('Пароль', '')	
if st.sidebar.button('Вход'):
    if login == '' or password == '':
        st.sidebar.write("**Введите логин и пароль**")
    else:
        st.sidebar.write("Добро пожаловать **" +  login + "!** ")
        

	
def loadTrack():
	uploaded_files = st.sidebar.file_uploader("Выберите a .db3 файл", accept_multiple_files=True)
	for uploaded_file in uploaded_files:
		bytes_data = uploaded_file.read()
		st.sidebar.write("Имя файла:", uploaded_file.name)

def currentRoadStatusOld():
    import streamlit as st
    import time
    import numpy as np

    st.markdown(f'# {list(page_names_to_funcs.keys())[1]}')
    st.write(
        """ Мониторинг состояния дорог. """
    )

    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()
    last_rows = np.random.randn(1, 1)
    chart = st.line_chart(last_rows, x = 'дни', y = 'количество найденных дефектов',)

    for i in range(1, 31):
        new_rows = last_rows[-1, :] + np.random.randn(5, 1).cumsum(axis=0)
        status_text.text("Анализ последних %i дней" % i)
        chart.add_rows(new_rows)
        progress_bar.progress(i)
        last_rows = new_rows
        time.sleep(0.05)

    progress_bar.empty()
    st.button("Пересчитать с учетом новых загруженных данных")
	
def currentRoadStatus():
    import streamlit as st
    import time
    import numpy as np

    st.write(""" Мониторинг состояния дорог. """)
    st.sidebar.write("Анализ последних 30 дней")
    chart_data = pd.DataFrame({ "Дни": [i for i in range(30)], 'Качество дорог, %': [50 + i + 2 * np.random.randn(1)[0] for i in range(30)]})
    #chart_data.index.name = 'Дни'
    st.line_chart(chart_data, x="Дни", y="Качество дорог, %")


page_names_to_funcs = {
     	"Текущее состояние дорог": currentRoadStatus,
	"Загрузить проезд": loadTrack,
}

demo_name = st.sidebar.selectbox("Выберите действие: ", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()

map_detect, cabinet = st.tabs(["maps", "information"])

model = YOLO('/home/enchar/test_image/best.pt')

df = pd.read_csv("my.csv", sep='\t')
last_clicked = None
uploaded_file = None
with map_detect:
    images = []
    dates = []
    for next_image in range(len(df)):
        dates.append(datetime.fromtimestamp(df['Name'].iloc[next_image]))
        images.append(f"data:image/png;base64,{df['image'].iloc[next_image]}")
    
    df['date'] = dates
    df['images'] = images
    images_base_64 = df['image']
    df = df.drop('image', axis=1)

    st.data_editor(
        df,
        column_config={
            "images": st.column_config.ImageColumn(
            "Preview Image", help="Streamlit app preview screenshots"
        ),
            # "image": st.column_config.ImageColumn(
            # "Preview Image", help="Streamlit app preview screenshots")
        },
        hide_index=True,
    )
    # st.write(df)

    

    

    location = df['latitude'].mean(), df['longitude'].mean()

    m = folium.Map(location=location, zoom_start=10)
    Draw(export=True).add_to(m)
    fg = folium.FeatureGroup(name="Markers")

    for i in range(0, len(df)):
        decoded = base64.b64decode(images_base_64.iloc[i])
        icon_url = io.BytesIO(decoded)
        icon = folium.features.CustomIcon(df['images'].iloc[i], icon_size=(50,50))
        folium.Marker([df['latitude'].iloc[i], df['longitude'].iloc[i]],popup=df['type'].iloc[i],
                    icon=icon).add_to(fg)
    # lol = folium.Marker([55.6811, 37.3347], icon=folium.Icon(color="red"))

    output = st_folium(m, feature_group_to_add=fg, width=900, height=600)

    if output['last_clicked'] is not None:
        last_clicked = output['last_clicked']
        uploaded_file = st.file_uploader("Upload your file here...", type=['png', 'jpeg', 'jpg'])
        folium.Marker([last_clicked['lat'], last_clicked['lng']], icon=folium.Icon(color="red")).add_to(fg)
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        if last_clicked is not None:
            results = model(image, device='cpu')
            uploaded_file = results[0].plot()
            st.image(uploaded_file)
            print(output['last_clicked'])

            with open('my.csv', 'a+', newline='\n') as csvfile:
                name = time.time()
                writer = csv.writer(csvfile, delimiter='\t')
                image_data = cv2.imencode('.jpg', uploaded_file)[1].tobytes()
                encoded_string = base64.b64encode(image_data).decode()
                writer.writerow([name, last_clicked['lat'], last_clicked['lng'], 'custom', encoded_string])
        
        uploaded_file = None


        

with cabinet:
    st.header("Information")
# c1, c2 = st.columns(2)
# with c1:
#     output = st_folium(m, width=700, height=500)
#
# with c2:
#     st.write(output)