# #Voice Assistant
# '''
# This chain creates a clone of ChatGPT with a few modifications to make it a voice assistant. It uses the pyttsx3 and speech_recognition
# libraries to convert text to speech and speech to text respectively. The prompt template is also changed to make it more suitable for
# voice assistant use.
# '''

# from langchain import OpenAI, ConversationChain, LLMChain, PromptTemplate
# from langchain.memory import ConversationBufferWindowMemory
# import os
# os.environ["OPENAI_API_KEY"] ="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"
# #serpapi_key="5e4b783d1e905b2992665d83235e27aaa73e103f239fb757b84be1cc2c75c57b"



# template = """Assistant is a large language model trained by OpenAI.

# Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

# Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

# Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.

# Assistant is aware that human input is being transcribed from audio and as such there may be some errors in the transcription. It will attempt to account for some words being swapped with similar-sounding words or phrases. Assistant will also keep responses concise, because human attention spans are more limited over the audio channel since it takes time to listen to a response.

# {history}
# Human: {human_input}
# Assistant:"""

# prompt = PromptTemplate(
#     input_variables=["history", "human_input"], 
#     template=template
# )


# chatgpt_chain = LLMChain(
#     llm=OpenAI(temperature=0), 
#     prompt=prompt, 
#     verbose=True, 
#     memory=ConversationBufferWindowMemory(k=2),
# )

# import speech_recognition as sr
# import pyttsx3
# engine = pyttsx3.init()


# def listen():
#     r = sr.Recognizer()
#     with sr.Microphone() as source:
#         print('Calibrating...')
#         r.adjust_for_ambient_noise(source, duration=5)
#         # optional parameters to adjust microphone sensitivity
#         # r.energy_threshold = 200
#         # r.pause_threshold=0.5    
        
#         print('Okay, go!')
#         while(1):
#             text = ''
#             print('listening now...')
#             try:
#                 audio = r.listen(source, timeout=5, phrase_time_limit=30)
#                 print('Recognizing...')
#                 # whisper model options are found here: https://github.com/openai/whisper#available-models-and-languages
#                 # other speech recognition models are also available.
#                 text = r.recognize_whisper(audio, model='medium.en', show_dict=True, )['text']
#             except Exception as e:
#                 unrecognized_speech_text = f'Sorry, I didn\'t catch that. Exception was: {e}s'
#                 text = unrecognized_speech_text
#             print(text)

            
#             response_text = chatgpt_chain.predict(human_input=text)
#             print(response_text)
#             engine.say(response_text)
#             engine.runAndWait()

# listen()
# #print(res)

from langchain import OpenAI, ConversationChain, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
import os
os.environ["OPENAI_API_KEY"] ="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"
#serpapi_key="5e4b783d1e905b2992665d83235e27aaa73e103f239fb757b84be1cc2c75c57b"



template = """Assistant is a large language model trained by OpenAI.

Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.

Assistant is aware that human input is being transcribed from audio and as such there may be some errors in the transcription. It will attempt to account for some words being swapped with similar-sounding words or phrases. Assistant will also keep responses concise, because human attention spans are more limited over the audio channel since it takes time to listen to a response.

{history}
Human: {human_input}
Assistant:"""

prompt = PromptTemplate(input_variables=["history", "human_input"], template=template)


chatgpt_chain = LLMChain(
    llm=OpenAI(temperature=0),
    prompt=prompt,
    verbose=True,
    memory=ConversationBufferWindowMemory(k=2),
)
import speech_recognition as sr
import pyttsx3

engine = pyttsx3.init()


def listen():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Calibrating...")
        r.adjust_for_ambient_noise(source, duration=5)
        # optional parameters to adjust microphone sensitivity
        # r.energy_threshold = 200
        # r.pause_threshold=0.5

        print("Okay, go!")
        while 1:
            text = ""
            print("listening now...")
            try:
                audio = r.listen(source, timeout=5, phrase_time_limit=30)
                print("Recognizing...")
                # whisper model options are found here: https://github.com/openai/whisper#available-models-and-languages
                # other speech recognition models are also available.
                text = r.recognize_whisper(
                    audio,
                    model="medium.en",
                    show_dict=True,
                )["text"]
            except Exception as e:
                unrecognized_speech_text = (
                    f"Sorry, I didn't catch that. Exception was: {e}s"
                )
                text = unrecognized_speech_text
            print(text)

            response_text = chatgpt_chain.predict(human_input=text)
            print(response_text)
            engine.say(response_text)
            engine.runAndWait()
listen()