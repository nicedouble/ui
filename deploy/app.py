#!/usr/bin/env python
# _*_coding:utf-8_*_

"""
@Time     : 2022/11/4 10:10
@Author   : ji hao ran
@File     : app.py
@Project  : ui
@Software : PyCharm
"""
import os
import time
import streamlit as st
from PIL import Image

if 'r1' not in st.session_state:
    st.session_state['r1'] = False

if 'r2' not in st.session_state:
    st.session_state['r2'] = False

current_path = os.path.abspath('../app')
tabs = st.tabs(['电池健康评价模型', '电池预警模型'])

with tabs[0]:
    n1 = 100
    btn1 = st.button('点击运行')
    ph1 = st.empty()
    if st.session_state['r1']:
        with ph1:
            st.image(Image.open(current_path + '/1.jpg'))
    if btn1:
        with ph1.container():
            st.write('正在计算...')
            bar1 = st.progress(0)
            for i in range(n1):
                time.sleep(0.1)
                bar1.progress(i + 1)
        ph1.empty()
        with ph1:
            st.image(Image.open(current_path + '/1.jpg'))
        st.session_state['r1'] = True

with tabs[1]:
    n2 = 100
    btn2 = st.button('点击运行', key='btn2')
    ph2 = st.empty()
    if st.session_state['r2']:
        with ph2:
            st.image(Image.open(current_path + '/2.jpg'))
    if btn2:
        with ph2.container():
            st.write('正在计算...')
            bar2 = st.progress(0)
            for i in range(n2):
                time.sleep(0.1)
                bar2.progress(i + 1)
        ph2.empty()
        with ph2:
            st.image(Image.open(current_path + '/2.jpg'))
        st.session_state['r2'] = True
