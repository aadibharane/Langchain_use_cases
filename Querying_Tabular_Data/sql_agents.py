from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.llms.openai import OpenAI
from langchain.agents import AgentExecutor
import os
os.environ["OPENAI_API_KEY"] ="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"
#serpapi_key="5e4b783d1e905b2992665d83235e27aaa73e103f239fb757b84be1cc2c75c57b"


def sql_agent():
    db = SQLDatabase.from_uri("sqlite:///C:/Program Files/SQLiteStudio/mydb.db")
    llm = OpenAI(temperature=0)
    toolkit = SQLDatabaseToolkit(db=db,llm=llm)

    agent_executor = create_sql_agent(
        llm=OpenAI(temperature=0),
        toolkit=toolkit,
        verbose=True
    )

    #Example: describing a table
    agent_executor.run("Describe the mydb table")

sql_agent()