# Chat with DB Open Web UI Pipe
Allow you to chat with Database and Basic Analytics using langchain pipe and Open WebUI

# How To Use

### Define your database url
```python
# define your database url compatible with SQLAlchemy connection string
DATABASE_URL = "postgresql://postgres:root@localhost:5432/test"
```
### Define your defaults for langchain pipe
```python
# Define defaults for langchain pipe
base_url: str = Field(default="http://localhost:11434")
ollama_embed_model: str = Field(default="nomic-embed-text")
ollama_model: str = Field(default="llama3.1:8b-instruct-q4_0")
```

### Modify your langchain code
```python
# Prompt Template
answer_prompt = PromptTemplate.from_template(
    """Given the following user question, corresponding SQL query, and SQL result, answer the user question from the SQL Result. Also return the query result in markdown format and python code to visualize the result (if necessary) using matplotlib.
Question: {question}
SQL Query: {query}
SQL Result: {result}
Answer: """
)
```

```python
# Modify chain
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
```
