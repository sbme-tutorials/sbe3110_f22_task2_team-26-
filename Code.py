import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import librosa
from scipy import signal
import soundfile as sf
import librosa.display
import mpld3
from plotly.subplots import make_subplots
import  streamlit_vertical_slider  as svs
import streamlit.components.v1 as components
from streamlit.components.v1 import html
import math
from time import sleep
import pandas as pd
import altair as alt
import time
import plotly.graph_objects as go
from matplotlib.widgets import Slider
import copy

def initial():
    st.session_state.pause_play_flag = False
    st.session_state['start']=0
    st.session_state['size1']=0
    st.session_state['i']=0
    st.session_state['lines']=[]
    st.session_state['flag'] = 1
    st.session_state['flagStart'] = 0
    st.session_state['startSize'] = 0


def loadAudio(file):
    y, sr = librosa.load(file)
    return y, sr

def outputAudio(data, sr):
    sf.write('output.wav', data, sr)
    st.sidebar.audio('output.wav', format='audio/wav')

def fourierTransform(data, sr):
    N = len(data)
    freq = np.fft.rfftfreq(N, 1/sr)
    dataFFt = np.fft.rfft(data)
    phase = np.angle(dataFFt)
    mag = np.abs(dataFFt)
    return dataFFt, freq, mag, phase, N

def settingvalues(sNumber, flag,nameFlag, names):
    return sNumber, flag,nameFlag, names

def inverseFourier(phase, mag):
    newData = mag * np.exp(1j * phase)
    iFFt = np.float64(np.fft.irfft(newData))
    return iFFt

def spectrogram(data, sr):
    frequencies, times, spectro = signal.spectrogram(data, sr)
    return frequencies, times, spectro


def bins_separation(frequency, amplitude, sNumber):
    freq_axis_list = []
    amplitude_axis_list = []
    e = []
    bin_max_frequency_value = math.ceil(len(frequency)/sNumber)
    i = 0
    for i in range(0, sNumber):
        freq_axis_list.append(frequency[i*bin_max_frequency_value : (i+1)*bin_max_frequency_value])
        amplitude_axis_list.append(amplitude[i*bin_max_frequency_value:(i+1)*bin_max_frequency_value])

    return freq_axis_list, amplitude_axis_list,bin_max_frequency_value

def Sliders_generation(bin_max_frequency_value, frequency, sNumber, names, nameFlag):
        columns = st.columns(sNumber)
        values = []
        for i in range(0, sNumber):
            with columns[i]:
                value = svs.vertical_slider( key= i, default_value=0.0, step=0.1, min_value=-1.0, max_value=1.0)
                if value == None:
                    value = 0.0
                values.append(value)
                if nameFlag==1:
                    st.write(f"{names[i]}")
                else:
                    if i < 9:
                        freq_values = int(frequency[(i+1)*bin_max_frequency_value])
                    elif i ==9:
                        freq_values = int(max(frequency))
                    st.write(f"{freq_values}")
        return values

def frequencyFunction(values, amplitude_axis_list):
    flist =[]
    for i in range(0, 10):
            flist.append(amplitude_axis_list[i] * (1+values[i]))
            
    flat_list =[]
    for sublist in flist:
            for item in sublist:
                flat_list.append(item)

    return flat_list
       
def getindex(mag, freq, frequency_inHz):
    index = []
    for i in range(len(mag)):
        if  frequency_inHz - 0.5 < freq[i] < frequency_inHz + 0.5 :
            index.append(i)

    return(index[0])


def vowel_music_Function(mag, freq, value,vowelflag):
    filter = mag.copy()
    if vowelflag:       #sh(1900:4500)      s(4500:8400)
        vowel_music = [getindex(mag,freq,1900), getindex(mag,freq,4500), getindex(mag,freq,8400),getindex(mag,freq,10000)]
        print("vowel")
    else:              #drums(0:280)      english horn(280:1000)      glockenspiel(1000:7000)
        vowel_music = [getindex(mag,freq,0), getindex(mag,freq,280), getindex(mag,freq,1000), getindex(mag,freq,7000)]
        print("music")

    filter[vowel_music[0]:vowel_music[1]] = filter[vowel_music[0]:vowel_music[1]]*(1+value[0])
    filter[vowel_music[1]:vowel_music[2]] = filter[vowel_music[1]:vowel_music[2]]*(1+value[1])
    filter[vowel_music[2]:vowel_music[3]] = filter[vowel_music[2]:vowel_music[3]]*(1+value[2])

    return filter


def arrhythmia (arrhythmia,y_fourier):
    new_y=y_fourier
    df = pd.read_csv('arrhythmia_components.csv')
    sub=df['sub']
    abs_sub=df['abs_sub']
    result = [item * arrhythmia for item in abs_sub]
    new_y=np.add(y_fourier,result)
    return new_y

def ECG_mode(df):

    # ------------ECG Sliders  
    Arrhythmia  =st.slider('Arrhythmia mode', step=1, max_value=100 , min_value=0  ,value=100 )
    Arrhythmia/=100
    # Reading uploaded_file
    # df = pd.read_csv(uploaded_file)
    uploaded_xaxis=df['time']
    uploaded_yaxis=df['amp']
    # Slicing big data
    if (len(uploaded_xaxis)>1000):
        uploaded_xaxis=uploaded_xaxis[:2000]
    if (len(uploaded_yaxis)>1000):
        uploaded_yaxis=uploaded_yaxis[:2000]

    # fourier transorm
    y_fourier,freq, mag, phase, N = fourierTransform(uploaded_yaxis, 160)

    y_fourier=arrhythmia (Arrhythmia,y_fourier)
    new_mag= np.abs(y_fourier)
    y_inverse_fourier = inverseFourier(phase, new_mag)

    uploaded_fig,uploaded_ax = plt.subplots()
    uploaded_ax.set_title('ECG signal ')
    uploaded_ax.plot(uploaded_xaxis[50:950],y_inverse_fourier[50:950])  
    uploaded_ax.set_xlabel('Time ')
    uploaded_ax.set_ylabel('Amplitude (mv)')
    st.plotly_chart(uploaded_fig)

def plot_animation(df):
    brush = alt.selection_interval()
    chart1 = alt.Chart(df).mark_line().encode(
            x=alt.X('time', axis=alt.Axis(title='Time')),
        ).properties(
            width=500,
            height=300
        ).add_selection(
            brush).interactive()
    
    figure = chart1.encode(
                  y=alt.Y('amplitude',axis=alt.Axis(title='Amplitude')))| chart1.encode(
                  y=alt.Y('amplitude after processing',axis=alt.Axis(title='Amplitude after Processing'))).add_selection(
            brush)

    return figure

def currentState(df, size, N):
    if st.session_state.size1 == 0:
        step_df = df.iloc[0:N]
    if st.session_state.flagStart == 0:
        step_df = df.iloc[0:N]
    if st.session_state.flag == 0:
        step_df = df.iloc[st.session_state.i : st.session_state.size1 - 1]
    lines = plot_animation(step_df)
    line_plot = st.altair_chart(lines)
    line_plot = line_plot.altair_chart(lines)  #
    return line_plot

def plotRep(df, size, start, N, line_plot):
    for i in range(start, N - size):  #
            st.session_state.start=i 
            st.session_state.startSize = i-1
            step_df = df.iloc[i:size + i]
            st.session_state.size1 = size + i
            st.session_state.i = i
            lines = plot_animation(step_df)
            line_plot.altair_chart(lines)
            st.session_state.size1 = size + i
            time.sleep(.1)   #
    if st.session_state.size1 == N - 1:
        st.session_state.flag =1
        step_df = df.iloc[0:N]
        lines = plot_animation(step_df)
        line_plot.altair_chart(lines)

def plotShow(data, idata,pause_btn,value,sr):
    time1 = len(data)/(sr)
    if time1>1:
        time1 = int(time1)
    time1 = np.linspace(0,time1,len(data))   
    df = pd.DataFrame({'time': time1[::300], 
                        'amplitude': data[:: 300],
                        'amplitude after processing': idata[::300]}, columns=[
                        'time', 'amplitude','amplitude after processing'])
    N = df.shape[0]  # number of elements in the dataframe
    burst = 10      # number of elements (months) to add to the plot
    size = burst 
    line_plot = currentState(df, size, N)

    if pause_btn:
        st.session_state.flag = 0
        st.session_state.pause_play_flag = not(st.session_state.pause_play_flag)
        if st.session_state.pause_play_flag :
            plotRep(df, size, st.session_state.start, N, line_plot)
    
    if st.session_state.pause_play_flag:
        st.session_state.flag = 1
        plotRep(df, size, st.session_state.start, N, line_plot)
        


def plottingInfreqDomain(freq, data):
    fig = plt.figure(figsize=(6,3))
    plt.plot(freq, data)
    st.plotly_chart(fig)

def plotSpectrogram(data, sr):
    initial()
    fig, ax = plt.subplots(1, sharex=True, figsize=(15,10))
    ax.specgram(data, Fs=sr)

    ax.set_ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    st.pyplot(fig)


def plottingSpectrogram(inbins, infreqs, inPxx):
    trace = [go.Heatmap(x= inbins, y= infreqs, z= 10*np.log10(inPxx), colorscale='Jet'),]
    layout = go.Layout(height=430, width=600)
    fig = go.Figure(data = trace, layout=layout)
    fig.update_layout(hovermode='x unified')
    st.plotly_chart(fig)

