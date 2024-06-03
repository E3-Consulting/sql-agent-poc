import google.auth
from dotenv import load_dotenv
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_vertexai import ChatVertexAI
import os

CREDENTIALS, PROJECT_ID = google.auth.default()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")

# TODO: refine code to make it more efficiend

def init_db(user: str, password: str, host: str, port: str, database: str) -> SQLDatabase:
    db_uri = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"
    return SQLDatabase.from_uri(db_uri)

def get_sql_chain(llm, db):

    # few-shot learning check
    # DONE change role-playing description
    template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, write a SQL query that would answer the user's question. Take the conversation history into account.

    <SCHEMA >{schema}</SCHEMA>

    Conversation History: {chat_history}

    Surround the query with <sql> </sql>.

    For example:
    Question: Which actors have the first name Scarlett?
    SQL Query:  <sql>select * from actor where first_name = 'Scarlett';</sql>
    Question: Is 'Academy Dinosaur' available for rent from Store 1?
    SQL Query:  <sql>select inventory.inventory_id
                from inventory join store using (store_id)
                    join film using (film_id)
                    join rental using (inventory_id)
                where film.title = 'Academy Dinosaur'
                    and store.store_id = 1
                    and not exists (select * from rental
                                    where rental.inventory_id = inventory.inventory_id
                                    and rental.return_date is null);</sql>

    Your turn:

    Question: {question}
    SQL Query:
    """

    prompt = ChatPromptTemplate.from_template(template)
    
    # tool
    def get_schema(_):
        return db.get_table_info()
    
    return (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm
        | StrOutputParser()
        | (lambda x: x.replace("<sql>", "").replace("</sql>", ""))
    )

def get_response(user_query: str, db: SQLDatabase, chat_history: list, llm):
    sql_chain = get_sql_chain(llm = llm, db = db)

    template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema, question, sql query and sql response below, write a natural language response.

    <SCHEMA >{schema}</SCHEMA>

    Conversation History: {chat_history}
    SQL Query: <sql>{query}</sql>
    User question: {question}
    SQL Response: {response}
    """

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        RunnablePassthrough.assign(query = sql_chain).assign(
            schema = lambda _: db.get_table_info(),
            response = lambda vars: db.run(vars["query"]),
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain.invoke({"question": user_query, "chat_history": chat_history})

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = [
        AIMessage(content="Hello! I am your sales agent. How can I help you today?"),
    ]

load_dotenv()

st.set_page_config(page_title="Sales Agent", page_icon=":mortar_board:")

st.title("Sales Agent")

with st.sidebar:
    st.subheader("Settings")
    st.write("This is a simple chat application using MySQL. Connect to the database and start chatting!")
    
    st.text_input("Host", value="localhost", key="host")
    st.text_input("Port", value="3306", key="port")
    st.text_input("User", value="newuser", key="user")
    st.text_input("Password", type="password", value="password", key="password")
    st.text_input("Database", value="sakila", key="database")

    if st.button("Connect"):
        with st.spinner("Connecting to database..."):
            db = init_db(
                st.session_state["user"],
                st.session_state["password"],
                st.session_state["host"],
                st.session_state["port"],
                st.session_state["database"],
            )

            st.session_state["db"] = db

            with st.spinner("Establishing connections..."):
                if "llm" not in st.session_state:
                    st.session_state["llm"] = ChatVertexAI(
                    model_name="gemini-1.0-pro-002",
                    max_output_tokens="5048",
                    temperature=0.0,
                    credentials=CREDENTIALS,
                    project_id=PROJECT_ID,)

            st.success("Connected to database!")

for message in st.session_state["chat_history"]:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)

user_query =st.chat_input("Type query...")

if user_query is not None and user_query.strip() != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        with st.spinner("Querying database..."):
            response = get_response(
                user_query=user_query,
                db=st.session_state["db"],
                chat_history=st.session_state["chat_history"],
                llm=st.session_state["llm"],
            )
            st.markdown(response)

    st.session_state.chat_history.append(AIMessage(content=response))