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
# from state import provide_state
if 'start' not in st.session_state:
    st.session_state['start']=0
if 'size1' not in st.session_state:
    st.session_state['size1']=0
if 'lines' not in st.session_state:
    st.session_state['lines']=[]
if 'flag' not in st.session_state:
    st.session_state['flag'] = 1

def loadAudio(file):
    y, sr = librosa.load(file)
    return y, sr

def outputAudio(data, sr):
    sf.write('output.wav', data, sr)
    st.sidebar.audio('output.wav', format='audio/wav')

def spectrogram(data, sr):
    frequencies, times, spectro = signal.spectrogram(data, sr)
    return frequencies, times, spectro

def fourierTransform(data, sr):
    N = len(data)
    freq = np.fft.rfftfreq(N, 1/sr)[:(N//2)]
    dataFFt = np.fft.fft(data)[:(N//2)]
    phase = np.angle(dataFFt)
    mag = np.abs(dataFFt)
    return dataFFt, freq, mag, phase, N


def inverseFourier(phase, mag):
    newData = mag * np.exp(1j * phase)
    iFFt = np.float64(np.fft.irfft(newData))
    return iFFt


def plottingInfreqDomain(freq, data):
    fig = plt.figure(figsize=(6,3))
    plt.plot(freq, data)
    st.plotly_chart(fig)

    
    
def plottingSpectrogram(inbins, infreqs, inPxx,flag):
    trace = [go.Heatmap(x= inbins, y= infreqs, z= 10*np.log10(inPxx), colorscale='Jet'),]
    layout = go.Layout(height=430, width=600)
    fig = go.Figure(data = trace, layout=layout)
    # fig.update_traces(showscale=False)
    fig.update_layout(hovermode='x unified')
    if flag:
        st.plotly_chart(fig)
    # return fig

def bins_separation(frequency, amplitude, sNumber):
    freq_axis_list = []
    amplitude_axis_list = []
    bin_max_frequency_value = math.ceil(len(frequency)/sNumber)
    i = 0
    for i in range(0, sNumber):
        freq_axis_list.append(frequency[i*bin_max_frequency_value : (i+1)*bin_max_frequency_value])
        amplitude_axis_list.append(amplitude[i*bin_max_frequency_value:(i+1)*bin_max_frequency_value])

    return freq_axis_list, amplitude_axis_list,bin_max_frequency_value

def Sliders_generation(bin_max_frequency_value, sNumber):
        columns = st.columns(sNumber)
        values = []
        for i in range(0, sNumber):
            with columns[i]:
                e = (i+1)*bin_max_frequency_value
                value = svs.vertical_slider( key= i, default_value=0.0, step=0.1, min_value=-1.0, max_value=1.0)
                if value == None:
                    value = 0.0
                values.append(value)
                st.write(f"{e}")
                
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

def vowlFunction(mag, freq, value): 
    # ahRange = [740, 1180, 2640]   1000-1250 2000-2800
    # iRange = [280, 2620, 3380]    2000-2800 3000-3450
    # aiRange = [360, 2220, 2960]   2000-2300 2850-3000
    # ʊRange = [380, 940, 2300]     800-1000  2000-2400
    # uRange = [320, 920, 2200]     800-1000  2000-2300
    filter = []
    arr =[]
    for i in range(len(mag)):
            if 1000 < freq[i] < 1250:             
                filter.append(mag[i])
            else:
                filter.append(mag[i])
    return filter

def musicFunction(mag, freq, value):
    filter = []
    for i in range(len(mag)):
        if 0 <= freq[i] < 279:    #Drums
            filter.append(mag[i] *(1 + value[0]))

        elif 280 < freq[i] < 1000:  #English horn
            filter.append(mag[i] * (1 + value[1]))

        elif 1000 <= freq[i] < 7000:  #Glockenspeil its actual range is from 784 4186
            filter.append(mag[i] * (1 + value[2]))

        else:
            filter.append(mag[i])
    return filter



def plot_animation(df):
    brush = alt.selection_interval()
    chart1 = alt.Chart(df).mark_line().encode(
            x=alt.X('time', axis=alt.Axis(title='Time')),
            # y=alt.Y('amplitude', axis=alt.Axis(title='Amplitude')),
        ).properties(
            width=500,
            height=300
        ).add_selection(
            brush).interactive()
    
    figure = chart1.encode(
                  y=alt.Y('amplitude',axis=alt.Axis(title='Amplitude')))| chart1.encode(
                  y=alt.Y('amplitude after processing',axis=alt.Axis(title='Amplitude after'))).add_selection(
            brush)

    return figure




def plotShow(data, idata,start_btn,pause_btn,resume_btn,value,sr):
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
    
    step_df = df.iloc[0:st.session_state.size1]
    lines = plot_animation(step_df)
    line_plot = st.altair_chart(lines)
    line_plot= line_plot.altair_chart(lines)

    # lines = plot_animation(df)
    # line_plot = st.altair_chart(lines)
    N = df.shape[0]  # number of elements in the dataframe
    burst = 10      # number of elements (months) to add to the plot
    size = burst    #   size of the current dataset
    if start_btn:
        st.session_state.flag = 1
        for i in range(1, N):
            st.session_state.start=i
            step_df = df.iloc[0:size]
            lines = plot_animation(step_df)
            line_plot = line_plot.altair_chart(lines)
            st.session_state.lines.append(line_plot)
            size = i + burst 
            st.session_state.size1 = size
            time.sleep(.1)

    elif resume_btn:
            st.session_state.flag = 1 
            for i in range( st.session_state.start,N):
                st.session_state.start =i 
                step_df = df.iloc[0:size]
                lines = plot_animation(step_df)
                line_plot = line_plot.altair_chart(lines)
                st.session_state.lines.append(line_plot)
                st.session_state.size1 = size
                size = i + burst
                time.sleep(.1)

    elif pause_btn:
            st.session_state.flag =0
            step_df = df.iloc[0:st.session_state.size1]
            lines = plot_animation(step_df)
            line_plot= line_plot.altair_chart(lines)
            s = st.session_state.lines.append(line_plot)


    if st.session_state.flag == 1:
        for i in range( st.session_state.start,N):
                st.session_state.start =i 
                step_df = df.iloc[0:size]
                lines = plot_animation(step_df)
                line_plot = line_plot.altair_chart(lines)
                st.session_state.lines.append(line_plot)
                st.session_state.size1 = size
                size = i + burst
                time.sleep(.1)

    
            
