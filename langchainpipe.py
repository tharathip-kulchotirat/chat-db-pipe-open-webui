"""
title: LangChain Pipe Function
author: Colby Sawyer @ Attollo LLC (mailto:colby.sawyer@attollodefense.com)
author_url: https://github.com/ColbySawyer7
version: 0.1.0

This module defines a Pipe class that utilizes LangChain
"""

from typing import Optional, Callable, Awaitable
from pydantic import BaseModel, Field
import os
import time

# import LangChain dependencies
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_community.llms import Ollama

# from langchain_ollama.llms import OllamaLLM
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_core.prompts import PromptTemplate
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from operator import itemgetter

# Uncomment to use OpenAI and FAISS
# from langchain_openai import ChatOpenAI
# from langchain_community.vectorstores import FAISS

DATABASE_URL = "postgresql://postgres:root@host.docker.internal:5432/test" # ปรับ url ตามความเหมาะสม


def SQLparse(data):
    return data.split("SQLQuery:")[1].strip()


class Pipe:
    class Valves(BaseModel):
        base_url: str = Field(default="http://host.docker.internal:11434") # ปรับ url ตามความเหมาะสม
        ollama_embed_model: str = Field(default="nomic-embed-text") # เลือก embedding model ตามความเหมาะสม
        ollama_model: str = Field(default="llama3.1:8b-instruct-q4_0") # เลือก LLM ตามความเหมาะสม
        openai_api_key: str = Field(default="...")
        openai_model: str = Field(default="gpt3.5-turbo")
        emit_interval: float = Field(
            default=2.0, description="Interval in seconds between status emissions"
        )
        enable_status_indicator: bool = Field(
            default=True, description="Enable or disable status indicator emissions"
        )

    def __init__(self):
        self.type = "pipe"
        self.id = "langchain_pipe"
        self.name = "LangChain Pipe"
        self.valves = self.Valves()
        self.last_emit_time = 0
        pass

    async def emit_status(
        self,
        __event_emitter__: Callable[[dict], Awaitable[None]],
        level: str,
        message: str,
        done: bool,
    ):
        current_time = time.time()
        if (
            __event_emitter__
            and self.valves.enable_status_indicator
            and (
                current_time - self.last_emit_time >= self.valves.emit_interval or done
            )
        ):
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "status": "complete" if done else "in_progress",
                        "level": level,
                        "description": message,
                        "done": done,
                    },
                }
            )
            self.last_emit_time = current_time

    async def pipe(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __event_emitter__: Callable[[dict], Awaitable[None]] = None,
        __event_call__: Callable[[dict], Awaitable[dict]] = None,
    ) -> Optional[dict]:
        await self.emit_status(__event_emitter__, "info", "/initiating Chain", False)

        # ======================================================================================================================================
        # MODEL SETUP
        # ======================================================================================================================================
        # Setup the model for generating responses
        # ==========================================================================
        # Ollama Usage
        _model = Ollama(
            model=self.valves.ollama_model, base_url=self.valves.base_url, temperature=0
        )
        # ==========================================================================
        # OpenAI Usage
        # _model = ChatOpenAI(
        #     openai_api_key=self.valves.openai_api_key,
        #     model=self.valves.openai_model
        # )
        # ==========================================================================

        # Example usage of FAISS for retreival
        # vectorstore = FAISS.from_texts(
        #     texts, embedding=OpenAIEmbeddings(openai_api_key=self.valves.openai_api_key)
        # )

        # ======================================================================================================================================
        # PROMPTS SETUP
        # ==========================================================================
        # _prompt = ChatPromptTemplate.from_messages([
        #     ("system", "You are a helpful bot"),
        #     ("human", "{question}")
        # ])
        answer_prompt = PromptTemplate.from_template(
            """Given the following user question, corresponding SQL query, and SQL result, answer the user question from the SQL Result. Also return the query result in markdown format and python code to visualize the result (if necessary) using matplotlib.
        Question: {question}
        SQL Query: {query}
        SQL Result: {result}
        Answer: """
        )
        # ======================================================================================================================================
        # CHAIN SETUP
        # ==========================================================================
        # Basic Chain
        db = SQLDatabase.from_uri(DATABASE_URL)
        execute_query = QuerySQLDataBaseTool(db=db)
        write_query = create_sql_query_chain(_model, db)
        chain = (
            RunnablePassthrough.assign(query=write_query).assign(
                result=itemgetter("query") | RunnableLambda(SQLparse) | execute_query
            )
            | answer_prompt
            | _model
            | StrOutputParser()
        )
        # ======================================================================================================================================
        # Langchain Calling
        # ======================================================================================================================================
        await self.emit_status(__event_emitter__, "info", "Starting Chain", False)
        messages = body.get("messages", [])
        # Verify a message is available
        if messages:
            question = messages[-1]["content"]
            try:
                # Invoke Chain
                response = chain.invoke({"question": question})
                # Set assitant message with chain reply
                body["messages"].append({"role": "assistant", "content": response})
            except Exception as e:
                await self.emit_status(
                    __event_emitter__,
                    "error",
                    f"Error during sequence execution: {str(e)}",
                    True,
                )
                return {"error": str(e)}
        # If no message is available alert user
        else:
            await self.emit_status(
                __event_emitter__,
                "error",
                "No messages found in the request body",
                True,
            )
            body["messages"].append(
                {
                    "role": "assistant",
                    "content": "No messages found in the request body",
                }
            )

        await self.emit_status(__event_emitter__, "info", "Complete", True)
        return response
