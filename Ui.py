import streamlit as st
import plotly.graph_objects as go
from Code import *

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

    data, sr = loadAudio(file)
    frequencies, times, spectro = spectrogram(data, sr)
    fdata, freq, mag, phase, number_samples = fourierTransform(data, sr)
    if flag == 1:
        freq_axis_list, amplitude_axis_list,bin_max_frequency_value = bins_separation(freq, mag, sNumber)
        col1,col2,col3,col4 = st.columns(4)
        start_btn  = col1.button("▷")
        pause_btn  = col2.button(label='Pause')
        resume_btn = col3.button(label='resume')
        spec1 = st.checkbox('Show Spectrogram')
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
            plotShow(data,idata, start_btn,pause_btn,resume_btn,valueSlider,sr)
            fig = make_subplots(1,2)
            plottingInfreqDomain(freq, newMagnitudeList)
            # plottingSpectrogram(frequencies1,times1,spectro1,spec1)
            
            # st.write(fig)
            
        
        # with col2:
            
            
            # plotShow(idata, start_btn)
            # plotShow(idata, start_btn)
    else:
        print("nothing")

