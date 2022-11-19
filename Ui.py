import streamlit as st
import plotly.graph_objects as go
from Code import *
import os.path
import copy


if "pause_play_flag" not in st.session_state:
    st.session_state.pause_play_flag = False
if 'start' not in st.session_state:
    st.session_state['start']=0
if 'size1' not in st.session_state:
    st.session_state['size1']=0
if 'i' not in st.session_state:
    st.session_state['i']=0
if 'lines' not in st.session_state:
    st.session_state['lines']=[]
if 'flag' not in st.session_state:
    st.session_state['flag'] = 1
if 'flagStrat' not in st.session_state:
    st.session_state['flagStart'] = 0
if 'startSize' not in st.session_state:
    st.session_state['startSize'] = 0
if 'fileName' not in st.session_state:
    st.session_state['fileName'] = ''

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
option = st.sidebar.selectbox('', ('Frequency', 'Instruments', 'Medical Signal', 'Vowels' ))
flag = 0 
if option == 'Frequency':
    sNumber, flag,nameFlag, names = settingvalues(10, 1,0, [])

elif option == 'Instruments':
    sNumber, flag,nameFlag, names = settingvalues(3, 1,1, ['Drums' , 'English horn', 'Glockenspiel'])

elif option == 'Medical Signal':
    sNumber, flag,nameFlag, names = settingvalues(4, 1,0, [])

elif option == 'Vowels':
    
    sNumber, flag,nameFlag, names = settingvalues(3, 1,1, ['sh',"s",'a'])



if file is not None:
    if st.session_state.fileName != file.name:
        initial()

    file_name=file.name
    st.session_state.fileName = file_name
    ext = os.path.splitext(file_name)[1][1:]

    if ext=='csv':
        df = pd.read_csv(file)
        if option == 'Medical Signal':
            ECG_mode(df) 
    else:
        if option == 'Medical Signal':
            st.write("")
        else:
            data, sr = loadAudio(file)
            frequencies, times, spectro = spectrogram(data, sr)
            fdata, freq, mag, phase, number_samples = fourierTransform(data, sr)
            if flag == 1:
                
                freq_axis_list, amplitude_axis_list, bin_max_frequency_value = bins_separation(freq, mag, sNumber)
                plotting = st.radio("Plot",['Time Domain', 'Spectogram'], label_visibility="hidden", horizontal= True)
                trymag = mag.copy()
                
                col1,col2 = st.columns(2)
                btn_col1,btn_col2,btn_col3,btn_col4,btn_col5,btn_col6,btn_col7,btn_col8 = st.columns(8)
                valueSlider = Sliders_generation(bin_max_frequency_value,freq, sNumber, names, nameFlag)
                
                if option == 'Frequency':
                    newMagnitudeList = frequencyFunction(valueSlider, amplitude_axis_list)
                elif option == 'Vowels':
                    newMagnitudeList = vowel_music_Function(mag, freq, valueSlider, 1) 
                elif option == 'Instruments':
                    newMagnitudeList = vowel_music_Function(mag, freq, valueSlider, 0)
                else:
                    print("")
                idata = inverseFourier(phase, newMagnitudeList)
                frequencies1, times1, spectro1 = spectrogram(idata, sr)
                audio = st.sidebar.audio(file, format='audio/wav')
                outputAudio(idata, sr)

                with col1:
                    if plotting == 'Time Domain':
                        # start_btn  = btn_col2.button("Play")
                        pause_btn  = btn_col4.button(label='Pause/Resume')
                        # resume_btn = btn_col3.button(label='Play')
                        plotShow(data,idata,pause_btn,valueSlider,sr)
                        

                if plotting == 'Spectogram':
                    flag = True
                    with col1:
                        plotSpectrogram(data,sr)
                    with col2:
                        plotSpectrogram(idata,sr)

