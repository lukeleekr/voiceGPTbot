import streamlit as st
import openai
from audiorecorder import audiorecorder
from pydub import AudioSegment
import numpy as np
from io import BytesIO  # Import BytesIO for handling byte conversion
import os
from datetime import datetime
from gtts import gTTS
import base64

def STT(audio):
    filename = "input.mp3"
    audio.export(filename, format="wav")
    
    with open(filename, "rb") as audio_file:
        transcript = openai.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            # Specify any other parameters you might need
        )
    os.remove(filename)
    return transcript.text

def ask_gpt(prompt, model):
    response = openai.chat.completions.create(
        model = model,
        messages = prompt
    )
    system_message = response.choices[0].message
    return system_message.content

def TTS(response):
    #gTTS를 활용하여 음성파일 생성
    filename = "output.mp3"
    tts = gTTS(text=response, lang='ko')
    tts.save(filename)

    #음원파일 자동 재생
    with open(filename, "rb") as f:
        audio = f.read()
        b64 = base64.b64encode(audio).decode()
        md = f"""
            <audio controls>
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        st.markdown(md, unsafe_allow_html=True)
    #파일 삭제
    os.remove(filename)


def main():
    # Page Config 기본 설정 (페이지 제목, 레이아웃 등)
    st.set_page_config(
        page_title="음성 비서 프로그램",
        layout="wide"
    )
    st.header("음성 비서 프로그램")

    st.markdown("---")

    with st.expander("음성 비서 프로그램에 관하여", expanded=True):
        st.write(
            """
            - 음성 비서 프로그램의 UI는 스트림릿(Streamlit)을 사용하여 구현하였습니다.
            - STT (Speech-To-Text)는 OpenAI의 WhisperAI 를 사용했습니다.
            - 답변은 OpenAI의 GPT 모델을 활용했습니다.
            - TTS (Text-To-Speech)는 Google Translate TTS를 활용했습니다.
            """

        )
    st.markdown("")

    # Sidebar 설정 (API Key, GPT 모델 선택)
    with st.sidebar:
        openai.api_key = st.text_input(label = "OpenAI API Key", placeholder="API Key를 입력하세요.", value="", type="password")
        st.markdown("---")

        model = st.radio(label="GPT 모델",options=["gpt-4", "gpt-3.5-turbo"])
        st.markdown("---")

        if st.button(label = "초기화"):
            # Reset codes
            st.session_state["chat"] = []
            st.session_state["messages"] = [{"role": "system", "content": "You are a thoughtful assistant. \
                                             Respond to all input in 25 words and answer them in Korean."}]

    flag_start = False

    # 기능 구현
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("질문 입력")
        audio = audiorecorder("클릭하여 녹음하기", "녹음 중...")
        if len(audio) > 0 and not np.array_equal(audio, st.session_state["check_audio"]):
            # Convert audio to a bytes-like object using BytesIO
            buffer = BytesIO()
            audio.export(buffer, format="wav")
            buffer.seek(0)
            st.audio(buffer.read(), format='audio/wav')

            # 음원 파일에서 텍스트 추출
            question = STT(audio)

            # 채팅을 시각화 하기 위해 질문 내용 저장
            now = datetime.now().strftime("%H:%M")
            st.session_state["chat"] += [("user", now, question)]

            # 질문 내용 저장
            st.session_state["messages"] += [{"role": "user", "content": question}]
            st.session_state["check_audio"] = audio
            flag_start = True

    with col2:
        st.subheader("질문/답변")
        if flag_start == True:
            # 질문에 대한 답변
            response = ask_gpt(st.session_state["messages"], model)

            # GPT 모델에 넣을 프롬프트를 위해 답변 내용을 저장
            st.session_state["messages"] += [{"role": "system", "content": response}]
            st.write(response)

            # 채팅 시각화를 위한 답변 내용 저장
            now = datetime.now().strftime("%H:%M")
            st.session_state["chat"] += [("bot", now, response)]

            for sender, time, message in st.session_state["chat"]:
                if sender == "user":
                    st.write(f'''<div style="display:flex; align-items:center;">
                                <div style="background-color:#007AFF; color:white; border-radius:12px; 
                                    padding:8px 12px; margin-right:8px;">{message}</div>
                                <div style="font-size:0.8rem; color:gray;">{time}</div>
                                </div>''', unsafe_allow_html=True)

                    st.write("")
                    

                else:
                    st.write(f'''<div style="display:flex; align-items:center; justify-content:flex-end;">
                             <div style="background-color:lightgray; border-radius:12px; padding:8px 12px; margin-left: 
                             8px;">{message}</div><div style = "font-size:0.8rem; color: 
                             gray;">{time}</div></div>''', unsafe_allow_html=True)
                    st.write("")

            # TTS를 활용하여 음성파일 생성
            TTS(response)

    
    # session_state reset
    if "chat" not in st.session_state:
        st.session_state["chat"] = []
    
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "system", "content": "You are a thoughtful assistant. \
                                         Respond to all input in 25 words and answer them in Korean."}]
    if "check_audio" not in st.session_state:
        st.session_state["check_audio"] = []


if __name__ == "__main__":
    main()