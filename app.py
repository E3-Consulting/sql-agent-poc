import google.auth
from dotenv import load_dotenv
import streamlit as st
from langchain_core.globals import set_verbose
from langchain.sql_database import SQLDatabase
from langchain.tools import StructuredTool
from langchain.agents import create_sql_agent
from langchain.agents.agent_types import AgentType
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.prompts import PromptTemplate
from langchain.pydantic_v1 import BaseModel, Field
from langchain_google_vertexai.llms import VertexAI
from langchain_google_vertexai.model_garden import ChatAnthropicVertex
from langchain_openai import ChatOpenAI
import json

import os

CREDENTIALS, PROJECT_ID = google.auth.default()

set_verbose(True)

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")

# TODO: refine code to make it more efficiend
def agent_init(db: SQLDatabase, model):
    sql_tools = SQLDatabaseToolkit(db=db, llm=model)

    # Define Plotting Tool
    ## Plotting tool parameters
    class PlotParams(BaseModel):
        """Parameters for generating a plot."""
        params: str =  Field(default=None, description="""String of a json with the following schema:
                            {
                                "title": "PlotParams",
                                "type": "object",
                                "properties": {
                                    "x": {
                                    "type": "array",
                                    "items": {
                                        "type": "number"
                                    },
                                    "description": "This field represents the data points for the x-axis of the plot."
                                    },
                                    "y": {
                                    "type": "array",
                                    "items": {
                                        "type": "number"
                                    },
                                    "description": "This field represents the data points for the y-axis of the plot."
                                    },
                                    "title": {
                                    "type": "string",
                                    "description": "This field represents the title of the plot.",
                                    "default": null
                                    },
                                    "xlabel": {
                                    "type": "string",
                                    "description": "This field represents the label for the x-axis of the plot.",
                                    "default": null
                                    },
                                    "ylabel": {
                                    "type": "string",
                                    "description": "This field represents the label for the y-axis of the plot.",
                                    "default": null
                                    },
                                    "plotkind": {
                                    "type": "string",
                                    "description": "This field represents the kind of plot that will be generated.",
                                    "default": null
                                    }
                                },
                                "required": ["x", "y"]
                            }""")

    ## Plotting tool
    def generate_plot(params: PlotParams) -> str:
        """Generate a plot with the given parameters"""
        from matplotlib import pyplot as plt

        params = json.loads(params.params)

        #         ## Database Schema- use this information to not waste time querying the database more than you should.
        # CREATE TABLE IF NOT EXISTS Suppliers (
        # supplier_id INT AUTO_INCREMENT PRIMARY KEY,
        # name VARCHAR(255) NOT NULL
        # );
        # CREATE TABLE IF NOT EXISTS Categories (
        # category_id INT AUTO_INCREMENT PRIMARY KEY,
        # name VARCHAR(255) NOT NULL
        # );
        # CREATE TABLE IF NOT EXISTS Items (
        # item_id INT AUTO_INCREMENT PRIMARY KEY,
        # name VARCHAR(255) NOT NULL,
        # supplier_id INT,
        # category_id INT,
        # supplier_item_number VARCHAR(255),
        # universal_product_code VARCHAR(255),
        # unit_of_measure VARCHAR(50),
        # packing VARCHAR(50),
        # units FLOAT,
        # unit_price FLOAT,
        # total_packing_price FLOAT,
        # brand TEXT,
        # description TEXT,
        # FOREIGN KEY (supplier_id) REFERENCES Suppliers(supplier_id),
        # FOREIGN KEY (category_id) REFERENCES Categories(category_id)
        # );
        # CREATE TABLE IF NOT EXISTS Clients (
        # client_id INT AUTO_INCREMENT PRIMARY KEY,
        # name VARCHAR(255) NOT NULL,
        # company_type VARCHAR(100),
        # contact_info TEXT
        # );
        # CREATE TABLE IF NOT EXISTS Orders (
        # order_id INT AUTO_INCREMENT PRIMARY KEY,
        # client_id INT,
        # order_date DATE,
        # FOREIGN KEY (client_id) REFERENCES Clients(client_id)
        # );
        # CREATE TABLE IF NOT EXISTS OrderItems (
        # order_item_id INT AUTO_INCREMENT PRIMARY KEY,
        # order_id INT,
        # item_id INT,
        # quantity INT,
        # price_at_order FLOAT,
        # FOREIGN KEY (order_id) REFERENCES Orders(order_id),
        # FOREIGN KEY (item_id) REFERENCES Items(item_id)
        # );

        plt.figure()
        plt.plot(params.x, params.y, kind = params.plotkind)
        plt.title(params.title)
        plt.xlabel(params.xlabel)
        plt.ylabel(params.ylabel)
        plt.savefig('plot.png')
        return "Plot saved as plot.png"

    plot_tool = StructuredTool.from_function(
        func=generate_plot,
        name="generate_plot",
        schema=PlotParams
    )

    # TODO- safety to not allow it to filter other suppliers
    prompt_template = PromptTemplate(
    input_variables=["input", "agent_scratchpad", "tools", "tool_names", "supplier", "chat_history"],
    template="""
    # Requirements
        ## Instructions
        You are a helpful, friendly assistant for a client of an app called Allec. Allec is a marketplace for suppliers to sell their items. Their own clients can use Allec to purchase items. 
        Your task is to answer to the user's requests by using the following tools: {tools}
        You will be interacting with a user that works for a certain supplier. Address the user in second person. The supplier you are interacting with is {supplier}. You must filter the database based on the supplier. The supplier names should be formatted in initial case in the database.

        ## Output Specification
        Provide an informative, analytical, and accurate answer to the question based on your query results. Since you have to be analytical, provide extra detail. Do not include code block markers in your action input (e.g., ```sql ```, [SQL: ], etc.). 'Action' should only contain the name of the tool to be used.

    # Thought Process
        ## Structure
        Use the following structure for your thought process, do not reformat this:
        Human: user's input
        Thought: What do I need to do?
        Action: What tool should be utilized from the following: {tool_names}
        Action Input: Code or query to be executed in the desired tool
        Observation: Document the results of the action
        (Repeat Thought/Action/Observation as needed until you arrive at an adequate response for the user)
        Thought: I have gathered enough information to answer the question
        Final Answer: Provide your final answer here, using the information from your observations

        ## Examples
            ### Example 1- Human: Show me all the items that we supply.

            Thought: I need to list the tables to verify that the "Suppliers" table exists and contains the relevant data.
            Action: list_sql_database_tool
            Action Input: "Suppliers, Items"
            Observation: Both "Suppliers" and "Items" tables exist in the database.

            Thought: I need to get the schema and sample rows from the "Suppliers" table to understand its structure and verify the supplier's name.
            Action: info_sql_database_tool
            Action Input: "Suppliers"
            Observation: The "Suppliers" table has columns: supplier_id, name. Sample row: (1, "Tech Supplies Inc.")

            Thought: Now I need to find the items associated with the supplier "Tech Supplies Inc." by joining the "Items" and "Suppliers" tables.
            Action: query_sql_checker_tool
            Action Input: "SELECT i.* FROM Items i JOIN Suppliers s ON i.supplier_id = s.supplier_id WHERE s.name = 'Tech Supplies Inc.'"
            Observation: Query is correct.

            Thought: Execute the validated query to get the list of items supplied by "Tech Supplies Inc."
            Action: query_sql_database_tool
            Action Input: "SELECT i.* FROM Items i JOIN Suppliers s ON i.supplier_id = s.supplier_id WHERE s.name = 'Tech Supplies Inc.'"
            Observation: The query returned the following rows: (item_id: 101, name: "Laptop", supplier_id: 1, ...), (item_id: 102, name: "Mouse", supplier_id: 1, ...)

            Thought: I have gathered detailed information to answer the question.
            Final Answer: The items supplied by "Tech Supplies Inc." are:
            1. Laptop
            2. Mouse
            (Details of other columns as per the returned rows)


            ### Example 2- Human: What categories of items to we sell?

            Thought: I need to ensure the necessary tables are present.
            Action: list_sql_database_tool
            Action Input: "Suppliers, Items, Categories"
            Observation: The tables "Suppliers", "Items", and "Categories" exist in the database.

            Thought: Get the schema and sample rows from the "Suppliers" table to confirm the supplier's name.
            Action: info_sql_database_tool
            Action Input: "Suppliers"
            Observation: The "Suppliers" table has columns: supplier_id, name. Sample row: (2, "Office Supplies Co.")

            Thought: I need to find the category IDs of items provided by "Office Supplies Co." by joining the "Items" and "Suppliers" tables.
            Action: query_sql_checker_tool
            Action Input: "SELECT DISTINCT i.category_id FROM Items i JOIN Suppliers s ON i.supplier_id = s.supplier_id WHERE s.name = 'Office Supplies Co.'"
            Observation: Query is correct.

            Thought: Execute the validated query to get the distinct category IDs.
            Action: query_sql_database_tool
            Action Input: "SELECT DISTINCT i.category_id FROM Items i JOIN Suppliers s ON i.supplier_id = s.supplier_id WHERE s.name = 'Office Supplies Co.'"
            Observation: The query returned the following category IDs: (category_id: 201), (category_id: 202)

            Thought: I need to get the names of these categories by querying the "Categories" table.
            Action: query_sql_checker_tool
            Action Input: "SELECT name FROM Categories WHERE category_id IN (201, 202)"
            Observation: Query is correct.

            Thought: Execute the validated query to get the category names.
            Action: query_sql_database_tool
            Action Input: "SELECT name FROM Categories WHERE category_id IN (201, 202)"
            Observation: The query returned the following rows: (name: "Office Furniture"), (name: "Stationery")

            Thought: I have gathered detailed information to answer the question.
            Final Answer: The categories of items provided by "Office Supplies Co." are:
            1. Office Furniture
            2. Stationery
            
            ### Example 3- Human: List all clients who placed orders in the last month.
            
            Thought: First, ensure the "Orders" and "Clients" tables exist.
            Action: list_sql_database_tool
            Action Input: "Orders, Clients"
            Observation: The "Orders" and "Clients" tables exist in the database.

            Thought: Get the schema and sample rows from the "Orders" table to understand its structure and verify the date format.
            Action: info_sql_database_tool
            Action Input: "Orders"
            Observation: The "Orders" table has columns: order_id, client_id, order_date. Sample row: (1, 1001, '2024-05-10')

            Thought: I need to identify orders placed in the last month.
            Action: query_sql_checker_tool
            Action Input: "SELECT client_id FROM Orders WHERE order_date >= DATE_SUB(CURDATE(), INTERVAL 1 MONTH)"
            Observation: Query is correct.

            Thought: Execute the validated query to get the list of client IDs.
            Action: query_sql_database_tool
            Action Input: "SELECT client_id FROM Orders WHERE order_date >= DATE_SUB(CURDATE(), INTERVAL 1 MONTH)"
            Observation: The query returned the following client IDs: (client_id: 1001), (client_id: 1002)

            Thought: I need to get the names of these clients by querying the "Clients" table.
            Action: query_sql_checker_tool
            Action Input: "SELECT name FROM Clients WHERE client_id IN (1001, 1002)"
            Observation: Query is correct.

            Thought: Execute the validated query to get the client names.
            Action: query_sql_database_tool
            Action Input: "SELECT name FROM Clients WHERE client_id IN (1001, 1002)"
            Observation: The query returned the following rows: (name: "Client A"), (name: "Client B")

            Thought: I have gathered detailed information to answer the question.
            Final Answer: The clients who placed orders in the last month are:
            1. Client A
            2. Client B

    # Chat History- look back to this in case of follow-up questions
    {chat_history}

    # Input    
    Human: {input}

    # Your thought process and answer
    {agent_scratchpad}
    """)

    # Create the agent
    agent = create_sql_agent(
    llm=model,
    extra_tools=[plot_tool],
    prompt=prompt_template,
    agent_type= AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    agent_executor_kwargs={"handle_parsing_errors":"Check you output and make sure it conforms! Do not output an action and a final answer at the same time."},
    db=db,
    verbose=True
    )

    return agent



def init_db(user: str, password: str, host: str, port: str, database: str) -> SQLDatabase:
    db_uri = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"
    return SQLDatabase.from_uri(db_uri)

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = [
        AIMessage(content="Hello! I am your Allec Marketplace Agent. How can I help you today?")
    ]

load_dotenv()

st.set_page_config(page_title="Sales Agent", page_icon=":mortar_board:")
st.logo("allec_logo.webp")

st.title("Marketplace Agent")

with st.sidebar:
    st.subheader("Login")
    st.write("Hello! Enter your credentials to get started.")
    
    st.session_state["supplier"] = st.selectbox("Supplier", ['AMBROSIA', 'BALLESTER', 'CAJITA VALLEJO',
       'CC1 BEER DISTRIBUTOR, INC.', 'FINE WINE IMPORTS',
       'MENDEZ & COMPANY', 'OCEAN LAB BREWING CO',
       'PAN AMERICAN WINE & SPIRITS', 'PUERTO RICO SUPPLIES GROUP',
       'QUINTANA HERMANOS', 'SEA WORLD', 'SERRALLES', 'V. SUAREZ',
       'JOSE SANTIAGO', 'ASIA MARKET', 'B. FERNANDEZ & HNOS. INC.',
       'CARIBE COMPOSTABLES', 'DON PINA', 'DROUYN', 'KIKUET', 'LA CEBA',
       'LEODI', 'MAVE INC', 'MAYS OCHOA', 'NORTHWESTERN SELECTA',
       'PACKERS', 'PROGRESSIVE', 'QUALITY CLEANING PRODUCTS', 'TO RICO',
       'ULINE', 'BAGUETTES DE PR', 'BESPOKE', 'CAIMÁN',
       'CARDONA AGRO INDUSTRY', 'CHURRO LOOP', 'COMEX', 'DONA LOLA',
       'GFG', 'HL HERNANDEZ', 'IMPERIAL DADE', 'INPROMO', 'JUGOS HENRY',
       'Northwestern Selecta', 'PAPELERIA', 'PLAZA CELLARS',
       'PR SUPPLIES', 'TACONAZO', 'WEBSTAURANT STORE',
       'COCA-COLA PUERTO RICO BOTTLES', 'CR - DISTRIBUTIONS BALLESTER',
       'CR - DISTRIBUTIONS BE WAFFLED', 'CR - DISTRIBUTIONS GREEN VALLEY',
       'DADE PAPER', 'EL VIANDON', 'FRIGORIFICO', 'GBS',
       'NATURAL FOOD CENTER', 'VAQUERIA TRES MONJITAS', 'YC DEPOT',
       'BIO-WARE', 'BLESS PRODUCE', 'ELABORACION DE PASTELILLOS',
       'FINCA CAMAILA', 'FINCA LAHAM', 'HIDROPONICO LOS HERMANOS',
       'JESÚS CUEVAS', 'JUGOS LA BORINQUEÑA', 'LEVAIN',
       'LIQUIDACIONES FELICIANO', 'MEDALLA DISTRIBUTORS',
       'MONDA QUE MONDA', 'MR. SPECIAL', 'NORIS UNIFORMS- AGUADILLA',
       'PASTA ASORE', 'PEREZ OFFICE', 'PLATANO  HIO',
       "POPEYE'S ICE FACTORY", 'PR VERDE FOOD DISTRIBUITOR',
       'PRODUCTOS DON GADY', 'PRODUCTOS EL PLANTILLERO',
       'PRODUCTOS MI ENCANTO', 'QUESO DEL PAÍS LA ESPERANZA',
       'RINCÓN RUM INC', 'TU PLÁTANO', 'VITIN', 'WESTERN PAPER', 'WAHMEY',
       'Luxo Wine'])
    st.text_input("Username", key="username")
    st.text_input("Password", type="password", key="password")

    if st.button("Login"):
        with st.spinner("Connecting to database..."):
            db = init_db(
                "root",
                "",
                "localhost",
                "3306",
                "allecmarketplacesample",
            )

            st.session_state["db"] = db

            with st.spinner("Establishing connections..."):
                if "llm" not in st.session_state:
                    # st.session_state["llm"] = ChatAnthropicVertex(name="claude-3-opus@20240229", temperature=0, streaming=False)
                    st.session_state["llm"] = VertexAI(model_name="gemini-1.5-flash-001", temperature=0)
                    # st.session_state["llm"] = ChatOpenAI(model = "gpt-4o", temperature=0)
                if "agent" not in st.session_state:
                    st.session_state["agent"] = agent_init(
                        db=st.session_state["db"], 
                        model=st.session_state["llm"])

            st.success("Connected to Marketplace!")

user_query =st.chat_input("Type query...")

for message in st.session_state["chat_history"]:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)


if user_query is not None and user_query.strip() != "":
    st.session_state["chat_history"].append(HumanMessage(content=user_query))
    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        with st.spinner("Fetching answer..."):
            response = st.session_state["agent"].invoke({"input":user_query, "supplier":st.session_state["supplier"], "chat_history":st.session_state["chat_history"]})
            st.markdown(response["output"])

    st.session_state["chat_history"].append(AIMessage(content=response["output"]))