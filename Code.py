import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import librosa
from scipy import signal
import soundfile as sf
import librosa.display
import mpld3
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
    st.session_state.start=0
if 'size1' not in st.session_state:
    st.session_state.size1=0

def loadAudio(file):
    y, sr = librosa.load(file)
    return y, sr

def outputAudio(data, sr):
    sf.write('output.wav', data, sr)
    st.sidebar.audio('output.wav', format='audio/wav')

# def spectrogram(data, sr):
#     frequencies, times, spectro = signal.spectrogram(data, sr)
#     return frequencies, times, spectro

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
    

<<<<<<< HEAD
# def plottingSpectrogram(data,idata,sr,flagToShow):
#     # xticks for first sample and second sample

#         # yticks for spectrograms
#         fig, ax = plt.subplots(1, 2,figsize=(15,30))
#         # fig.tight_layout(pad=10.0)
#         layout = go.Layout(height=430, width=600)

#         ax[0].specgram(data, Fs=sr)
#         ax[0].set_xlabel(xlabel='Time [sec]', size=25)
#         ax[0].set_ylabel(ylabel='Frequency Amplitude [rad/s]', size=25)
#         # ax[0].set_yticks(helper)
#         # ax[0].set_yticklabels(spec_yticks)
#         ax[0].set_title("First Channel", fontsize=30)
#         ax[0].tick_params(axis='both', which='both', labelsize=18)

#         ax[1].specgram(idata, Fs=sr)
#         ax[1].set_xlabel(xlabel='Time [sec]', size=25)
#         ax[1].set_ylabel(ylabel='Frequency Amplitude [rad/s]', size=25)
#         # ax[1].set_yticks(helper)
#         # ax[1].set_yticklabels(spec_yticks)
#         ax[1].set_title("Second Channel", fontsize=30)
#         ax[1].tick_params(axis='both', which='both', labelsize=18)
#         if flagToShow:
#             st.pyplot(fig)

    
    
def plottingSpectrogram(inbins, infreqs, inPxx):
    trace = [go.Heatmap(x= inbins, y= infreqs, z= 10*np.log10(inPxx), colorscale='Jet'),]
    layout = go.Layout(height=430, width=600)
    fig = go.Figure(data = trace, layout=layout)
    fig.update_traces(showscale=False)
    fig.update_layout(hovermode='x unified')
    st.plotly_chart(fig)
=======
def plottingSpectrogram(data,idata,sr,flagToShow):
    # xticks for first sample and second sample

        # yticks for spectrograms
        fig, ax = plt.subplots(1, 2,figsize=(6,2))
        # fig.tight_layout(pad=10.0)

        ax[0].specgram(data, Fs=sr)
        ax[0].set_xlabel(xlabel='Time [sec]', size=5)
        ax[0].set_ylabel(ylabel='Frequency Amplitude [rad/s]', size=5)
        # ax[0].set_yticks(helper)
        # ax[0].set_yticklabels(spec_yticks)
        ax[0].set_title("First Channel", fontsize=5)
        ax[0].tick_params(axis='both', which='both', labelsize=5)

        ax[1].specgram(idata, Fs=sr)
        ax[1].set_xlabel(xlabel='Time [sec]', size=5)
        ax[1].set_ylabel(ylabel='Frequency Amplitude [rad/s]', size=5)
        # ax[1].set_yticks(helper)
        # ax[1].set_yticklabels(spec_yticks)
        ax[1].set_title("Second Channel", fontsize=5)
        ax[1].tick_params(axis='both', which='both', labelsize=5)
        if flagToShow:
            st.pyplot(fig)

    
    
# def plottingSpectrogram(inbins, infreqs, inPxx):
#     trace = [go.Heatmap(x= inbins, y= infreqs, z= 10*np.log10(inPxx), colorscale='Jet'),]
#     layout = go.Layout(height=430, width=600)
#     fig = go.Figure(data = trace, layout=layout)
#     fig.update_traces(showscale=False)
#     fig.update_layout(hovermode='x unified')
#     st.plotly_chart(fig)
>>>>>>> 1b0de655e3bf4fd4d80a9cd0d42c1933a01a9d2b

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
                # e = (i+1)*bin_max_frequency_value
                # value = svs.vertical_slider( key= i, default_value=1, step=1, min_value=-1, max_value=1)
                # if value == None:
                #     value = 1
                # values.append(value)
                value = svs.vertical_slider( key= i, default_value=0.0, step=0.1, min_value=-1.0, max_value=1.0)
                if value == None:
                    value = 0.0
                values.append(value)
                # amplitude_axis_list[i] = value * amplitude_axis_list[i]
                
        return values

def frequencyFunction(values, amplitude_axis_list):
    flist =[]
    for i in range(0, 10):
            flist.append(amplitude_axis_list[i] * (values[i]))
            
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





def plotShow(data, idata,start_btn,pause_btn,resume_btn):
    time1 = np.linspace(0,2,len(data))
    df = pd.DataFrame({'time': time1[::500], 
                      'amplitude': data[:: 500],
                      'amplitude after processing': idata[::500]}, columns=[
                    'time', 'amplitude','amplitude after processing'])

    lines = plot_animation(df)
    line_plot = st.altair_chart(lines)
    N = df.shape[0]  # number of elements in the dataframe
    burst = 10      # number of elements (months) to add to the plot
    size = burst    #   size of the current dataset
    if start_btn:
        for i in range(1, N):
            st.session_state.start=i
            print(st.session_state.start)
            step_df = df.iloc[0:size]
            lines = plot_animation(step_df)
            line_plot = line_plot.altair_chart(lines)
            size = i + burst 
            st.session_state.size1 = size

            
            
            
    
            # step_df = df.iloc[0:size]
            # lines = plot_animation(step_df)
            # line_plot = line_plot.altair_chart(lines)
            # size = i + burst
            # if size >= N:
            #     size = N - 1
            time.sleep(.1)
    elif resume_btn: 
            print(st.session_state.start)
            for i in range( st.session_state.start,N):
                st.session_state.start =i 
                step_df = df.iloc[0:size]
                lines = plot_animation(step_df)
                line_plot = line_plot.altair_chart(lines)
                st.session_state.size1 = size
                size = i + burst
                time.sleep(.1)
                 
                # if st.session_state.size1 >=N:
                #     size = N - 1

    elif pause_btn:
            step_df = df.iloc[0:st.session_state.size1]
            lines = plot_animation(step_df)
            line_plot = line_plot.altair_chart(lines)
            # size = i + burst
            if pause_btn:
                print("pause")




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
    # frequencies, times, spectro = spectrogram(data, sr)
    fdata, freq, mag, phase, number_samples = fourierTransform(data, sr)
    if flag == 1:
        freq_axis_list, amplitude_axis_list,bin_max_frequency_value = bins_separation(freq, mag, sNumber)
        col1,col2,col3,col4 = st.columns(4)
        start_btn  = col1.button("▷")
        pause_btn  = col2.button(label='Pause')
        resume_btn = col3.button(label='resume')
        valueSlider = Sliders_generation(bin_max_frequency_value, sNumber)
        spec1 = st.checkbox('Show Spectrogram')
        if option == 'Frequency':
            newMagnitudeList = frequencyFunction(valueSlider, amplitude_axis_list)
        elif option == 'Vowels':
            newMagnitudeList = vowlFunction(mag, freq, valueSlider) 
        elif option == 'Instruments':
            newMagnitudeList = musicFunction(mag, freq, valueSlider)
        else:
            newMagnitudeList = mag
        idata = inverseFourier(phase, newMagnitudeList)
        audio = st.sidebar.audio(file, format='audio/wav')
        outputAudio(idata, sr)
        with col1:
            plotShow(data,idata, start_btn,pause_btn,resume_btn)
<<<<<<< HEAD
            plottingSpectrogram(frequencies,times,spectro)
=======
        plottingSpectrogram(data,idata,sr,spec1)
>>>>>>> 1b0de655e3bf4fd4d80a9cd0d42c1933a01a9d2b
        
        # with col2:
            # plotShow(idata, start_btn)
            # plotShow(idata, start_btn)
    else:
        print("nothing")

