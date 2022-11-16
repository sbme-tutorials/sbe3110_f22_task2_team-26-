import streamlit as st
import plotly.graph_objects as go
from Code import *
import os.path


if 'start' not in st.session_state:
    st.session_state['start']=0
if 'size1' not in st.session_state:
    st.session_state['size1']=0
if 'lines' not in st.session_state:
    st.session_state['lines']=[]
if 'flag' not in st.session_state:
    st.session_state['flag'] = 0

st.set_page_config(page_title="Equalizer",layout='wide')
st.markdown("""
        <style>
               .css-18e3th9 {
                    padding-top: 2rem;
                    padding-bottom: 15rem;
                    padding-left: 2.5rem;
                    padding-right: 5rem;
                }
        </style>
        """, unsafe_allow_html=True)

file = st.sidebar.file_uploader('Upload a file')
col1, col2 = st.columns(2)
option = st.sidebar.selectbox('', (' ', 'Frequency', 'Instruments', 'Medical Signal', 'Vowels' ))
flag = 0 
if option == 'Frequency':
    sNumber = 10
    flag = 1
elif option == 'Instruments':
    sNumber = 3
    flag = 1
elif option == 'Medical Signal':
    sNumber = 4
    flag = 1
elif option == 'Vowels':
    sNumber = 6
    flag = 1
elif option == 'None':
    flag = 0



if file is not None:
    file_name=file.name
    ext = os.path.splitext(file_name)[1][1:]
    if ext=='csv':
        df = pd.read_csv(file)
        if option == 'Medical Signal':
            ECG_mode(df) 
    else:

        data, sr = loadAudio(file)
        frequencies, times, spectro = spectrogram(data, sr)
        fdata, freq, mag, phase, number_samples = fourierTransform(data, sr)
        if flag == 1:
            freq_axis_list, amplitude_axis_list,bin_max_frequency_value = bins_separation(freq, mag, sNumber)
            plotting = st.radio("Plot",['Time Domain', 'Spectogram'], label_visibility="hidden", horizontal= True)
            
            col1,col2 = st.columns(2)
            c1,c2,c3,c4 = st.columns(4)
            valueSlider = Sliders_generation(bin_max_frequency_value, sNumber)
            
            if option == 'Frequency':
                newMagnitudeList = frequencyFunction(valueSlider, amplitude_axis_list)
            elif option == 'Vowels':
                newMagnitudeList = vowlFunction(mag, freq, valueSlider) 
            elif option == 'Instruments':
                newMagnitudeList = musicFunction(mag, freq, valueSlider)
    
            else:
                newMagnitudeList = mag
            idata = inverseFourier(phase, newMagnitudeList)
            frequencies1, times1, spectro1 = spectrogram(idata, sr)
            audio = st.sidebar.audio(file, format='audio/wav')
            outputAudio(idata, sr)

            with col1:
                if plotting == 'Time Domain':
                    start_btn  = c1.button("▷")
                    pause_btn  = c2.button(label='Pause')
                    resume_btn = c3.button(label='resume')
                    default_btn = c4.button(label='Default')
                    plotShow(data,idata, start_btn,pause_btn,resume_btn,valueSlider,sr,default_btn)

            if plotting == 'Spectogram':
                flag = True
                with col1:
                    # plottingSpectrogram(times,frequencies,spectro)
                    plottingInfreqDomain(freq,mag)
                with col2:
                    # plottingSpectrogram(times1,frequencies1,spectro1)
                    plottingInfreqDomain(freq,newMagnitudeList)

