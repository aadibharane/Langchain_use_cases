#Getting Started
'''
This notebook walks through how LangChain thinks about memory.

Memory involves keeping a concept of state around throughout a user’s interactions with an language model. 
A user’s interactions with a language model are captured in the concept of ChatMessages, so this boils down to ingesting,
capturing, transforming and extracting knowledge from a sequence of chat messages. There are many different ways to do this, 
each of which exists as its own memory type.
In general, for each type of memory there are two ways to understanding using memory. These are the standalone functions which 
extract information from a sequence of messages, and then there is the way you can use this type of memory in a chain
Memory can return multiple pieces of information (for example, the most recent N messages and a summary of all previous messages).
The returned information can either be a string or a list of messages.
In this notebook, we will walk through the simplest form of memory: “buffer” memory, which just involves keeping a buffer of all prior
messages. We will show how to use the modular utility functions here, then show how it can be used in a chain (both returning a 
string as well as a list of messages).
'''
#ChatMessageHistory
'''
One of the core utility classes underpinning most (if not all) memory modules is the ChatMessageHistory class. 
This is a super lightweight wrapper which exposes convenience methods for saving Human messages, AI messages, and then fetching them all.
'''
import os
os.environ["OPENAI_API_KEY"] ="OPENAI_API_KEY"
#You may want to use this class directly if you are managing memory outside of a chain.
from langchain.memory import ChatMessageHistory
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
import json

from langchain.memory import ChatMessageHistory
from langchain.schema import messages_from_dict, messages_to_dict



def conversation_memory():
    history = ChatMessageHistory()

    history.add_user_message("hi!")

    history.add_ai_message("whats up?")

    res=history.messages
    #print(res)

    #ConversationBufferMemory
    '''
    We now show how to use this simple concept in a chain. We first showcase ConversationBufferMemory which is just a wrapper around
    ChatMessageHistory that extracts the messages in a variable.
    '''
    #We can first extract it as a string.


    memory = ConversationBufferMemory()
    memory.chat_memory.add_user_message("hi!")
    memory.chat_memory.add_ai_message("whats up?")

    res=memory.load_memory_variables({})
    #print(res)

    #We can also get the history as a list of messages
    memory = ConversationBufferMemory(return_messages=True)
    memory.chat_memory.add_user_message("hi!")
    memory.chat_memory.add_ai_message("whats up?")
    res=memory.load_memory_variables({})
    #print(res)

    #Using in a chain
    #Finally, let’s take a look at using this in a chain (setting verbose=True so we can see the prompt).


    llm = OpenAI(temperature=0)
    conversation = ConversationChain(
        llm=llm, 
        verbose=True, 
        memory=ConversationBufferMemory()
    )
    res=conversation.predict(input="Hi there!")
    print(res)

    con=conversation.predict(input="I'm doing well! Just having a conversation with an AI.")
    print(con)

    con=conversation.predict(input="Tell me about yourself.")
    print(con)

    #Saving Message History
    '''
    You may often have to save messages, and then load them to use again. This can be done easily by first converting the messages
    to normal python dictionaries, saving those (as json or something) and then loading those. Here is an example of doing that.
    '''


    history = ChatMessageHistory()

    history.add_user_message("hi!")

    history.add_ai_message("whats up?")

    dicts = messages_to_dict(history.messages)
    print(dicts)
    new_messages = messages_from_dict(dicts)
    print(new_messages)
conversation_memory()