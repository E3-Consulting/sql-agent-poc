import google.auth
from dotenv import load_dotenv
import streamlit as st
from langchain_core.globals import set_verbose
from langchain.sql_database import SQLDatabase
from langchain.agents import create_sql_agent
from langchain.agents.agent_types import AgentType
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.utilities.sql_database import SQLDatabase
from langchain.prompts import PromptTemplate
from langchain_google_vertexai.llms import VertexAI
from langchain_core.output_parsers import JsonOutputParser
from custom_sql import CustomSQLDatabaseToolkit
import pandas as pd
import os

set_verbose(True)

CREDENTIALS, PROJECT_ID = google.auth.default()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")

st.set_page_config(page_title="Sales Agent", page_icon=":mortar_board:")
st.logo("allec_logo.webp")

st.title("Marketplace Agent")

load_dotenv()

# TODO: refine code to make it more efficient
def agent_init(db: SQLDatabase, model):
    # TODO- safety to not allow it to filter other suppliers
    prompt_template = PromptTemplate(
    input_variables=["input", "agent_scratchpad", "tools", "tool_names", "supplier", "chat_history"],
    template="""
    # Requirements
        ## Instructions
        You are a helpful, friendly assistant for a client of an app called Allec. Allec is a marketplace for suppliers to sell their items. Their own clients can use Allec to purchase items. 
        Your task is to answer to the user's requests by using the following tools: {tools}
        You will be interacting with a user that works for a certain supplier. Address the user in second person. The supplier you are interacting with is {supplier}. You must filter the database based on the supplier.
        If the query includes a request to plot, ignore it. This is beyond your capabilities.
        Finally, some details in the database may be written in spanish, however it must be noted that: **many columns will not be direct translations, list the values of column that is needed before making these assumptions**. Try to account for this as best as possible when constructing your queries.

        ## Output Specification
        Provide an informative, analytical, and accurate answer to the question based on your query results. Once you have a final answer, stop the thought process and provide your output after "Final Answer:". However, if you are stuck in a loop trying to find an answer to no avail, respond with "I do not have enough information to answer that".
        If you get None as a query result, respond with "There is no data available to answer that".

        ### Extra specifications:
        Do not include code block markers in your action input (e.g., ```sql ```, [SQL: ], etc.). 'Action' should only contain the name of the tool to be used.

        ## STOP CONDITION:
        Once a sql query is executed that returns a result that answers the question, return the final Thought and Final Answer.

    # Database Table Descriptions:
        * Categories- the names and ids of Categories that respond to the Items.
        * Clients- the names, ids, and company_types of Clients that respond to the Orders.
        * Items- information that responds to the items of all suppliers.
        * Suppliers- ids and names of Suppliers.
        * Orders- the date and ids of Orders, in addition to the client id associated with each.
        * OrderItems- for each order id, the items, qty, and price at order.

        ## Database Schema- use this information to understand the structure of the database.
            CREATE TABLE IF NOT EXISTS Suppliers (
            supplier_id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(255) NOT NULL
            );
            CREATE TABLE IF NOT EXISTS Categories (
            category_id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(255) NOT NULL
            );
            CREATE TABLE IF NOT EXISTS Items (
            item_id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            supplier_id INT,
            category_id INT,
            supplier_item_number VARCHAR(255),
            universal_product_code VARCHAR(255),
            unit_of_measure VARCHAR(50),
            packing VARCHAR(50),
            units FLOAT,
            unit_price FLOAT,
            total_packing_price FLOAT,
            brand TEXT,
            description TEXT,
            FOREIGN KEY (supplier_id) REFERENCES Suppliers(supplier_id),
            FOREIGN KEY (category_id) REFERENCES Categories(category_id)
            );
            CREATE TABLE IF NOT EXISTS Clients (
            client_id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            company_type VARCHAR(100),
            contact_info TEXT
            );
            CREATE TABLE IF NOT EXISTS Orders (
            order_id INT AUTO_INCREMENT PRIMARY KEY,
            client_id INT,
            order_date DATE,
            FOREIGN KEY (client_id) REFERENCES Clients(client_id)
            );
            CREATE TABLE IF NOT EXISTS OrderItems (
            order_item_id INT AUTO_INCREMENT PRIMARY KEY,
            order_id INT,
            item_id INT,
            quantity INT,
            price_at_order FLOAT,
            FOREIGN KEY (order_id) REFERENCES Orders(order_id),
            FOREIGN KEY (item_id) REFERENCES Items(item_id)
            );
        

    # Thought Process
        ## Structure
        Use the following structure for your thought process, do not reformat this:
        Human: user's input
        Thought: What do I need to do?
        Action: What tool should be utilized from the following: {tool_names}
        Action Input: Code or query to be executed in the desired tool
        Observation: Document the results of the action
        (Repeat Thought/Action/Observation until you arrive at an adequate sql query result that answers the user's question)
        Thought: The result of the executed query answers the user's query
        Final Answer: Natural language conversion of the result of the executed query

        ## Examples
            ### Example 1- Human: Show me all the items that we supply.

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
    toolkit = CustomSQLDatabaseToolkit(db = db, llm = model),
    prompt=prompt_template,
    agent_type= AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
    )

    return agent

def query_asks_for_plotting(text: str) -> bool:
    instructions = """
    You will be receiving a question from a user below. Your task is to check if the question includes a request to plot data. Only respond with "Yes" or "No".
    If the response contains "Agent stopped due to iteration limit or time limit.", repond with "No".
    
    Question: {text}
    """

    response = st.session_state["llm"].invoke(instructions.format(text=text)) 
    print(response)

    if "Yes" in response:
        return True
    else:
        return False

def llm_plotter(user_query: str, response: str):
    prompt =("""
            In the end of this prompt, there are two values: user query and response. The user query is a request to retrieve and plot data. The response is the retrieved data in natural language.
            We want to convert the retrieved data into a JSON format. Further, based on the user query, we want to plot as requested. If the request is vague, we want to assume the best fit from the retrieved data.
            There are four types of plots: table, bar, line, and histogram.

            ---
            For the following query, if it requires drawing a table, reply as follows:
            {"table": {"columns": ["column1", "column2", ...], "data": [[value1, value2, ...], [value1, value2, ...], ...]}}

            If the query requires creating a bar chart, reply as follows:
            {"bar": {"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...], "metadata": {"title": "Table Title", "xlabel": "X Label", "ylabel": "Y Label"}}

            If the query requires creating a line chart, reply as follows:
            {"line": {"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...], "metadata": {"title": "Table Title", "xlabel": "X Label", "ylabel": "Y Label"}}

            If the query requires creating a histogram, reply as follows:
            {"histogram": {"column": ["A"], "data": [25, 24, 10, ...]}, "metadata": {"title": "Table Title", "xlabel": "X Label", "ylabel": "Y Label"}}

            Return all output as a string.

            All strings in "columns" list and data list, should be in double quotes,

            For example: {"columns": ["title", "ratings_count"], "data": [["Gilead", 361], ["Spider's Web", 5164]]}

            ---
            Do not respond with backticks. For example, instead of responding like this:
             
             ```json {"bar": {"columns": ["\"Unius Olive Oil Arbequina 750ml\"", "\"Olive Oil Casanova Estates \\\"L\'Olio Toscano\\\" 2020 500ml\"", "\"Pic colomini d'Aragona Ex tra Virgin Olive Oil 2020 500ml\"", "\"Sol del Silenc io Premium Oil 500ml\"", "\"Ume Juice (Can) - Pack of 30 8.45oz\"", "\"Mik an Juice (Can) - Pack of 30 8.45oz\"", "\"Apple Juice (Can) - Pac k of 30 8.45oz\"", "\"Ume Juice - Pack of 24 8.45oz\"", "\"The 1 Water / Wine Glass (No Stem) - 6 Unit Presentation Gift Pack\"", "\"Mik an Juice - Pack of 24 8.45oz\"", "\"Apple Juice - Pac k of 24 8.45oz\"", "\"Deep Sea Water - Pack of 6 2L\"", "\"Young Wine Decanter\"", "\"Deep Sea Water - Pack of 24 17.5oz\"", "\"Polis hing Cloth\"", "\"Water Carafe (Pre-Order Only)\"", "\"Mature Wine Decanter\"", "\"Water / Wine Glass (Pre-Order Only)\"", "\"The 1 Glass - 2 Unit Presentation Gift Pack\"", "\"The 1 Glass\""], "data": [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], "metadata": {"title": "Top Selling Items by Revenue", "xlabel": "Item Name", "ylabel": "Revenue"}}} ```

            Respond like this:

            {"bar": {"columns": ["\"Unius Olive Oil Arbequina 750ml\"", "\"Olive Oil Casanova Estates \\\"L\'Olio Toscano\\\" 2020 500ml\"", "\"Pic colomini d'Aragona Ex tra Virgin Olive Oil 2020 500ml\"", "\"Sol del Silenc io Premium Oil 500ml\"", "\"Ume Juice (Can) - Pack of 30 8.45oz\"", "\"Mik an Juice (Can) - Pack of 30 8.45oz\"", "\"Apple Juice (Can) - Pac k of 30 8.45oz\"", "\"Ume Juice - Pack of 24 8.45oz\"", "\"The 1 Water / Wine Glass (No Stem) - 6 Unit Presentation Gift Pack\"", "\"Mik an Juice - Pack of 24 8.45oz\"", "\"Apple Juice - Pac k of 24 8.45oz\"", "\"Deep Sea Water - Pack of 6 2L\"", "\"Young Wine Decanter\"", "\"Deep Sea Water - Pack of 24 17.5oz\"", "\"Polis hing Cloth\"", "\"Water Carafe (Pre-Order Only)\"", "\"Mature Wine Decanter\"", "\"Water / Wine Glass (Pre-Order Only)\"", "\"The 1 Glass - 2 Unit Presentation Gift Pack\"", "\"The 1 Glass\""], "data": [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], "metadata": {"title": "Top Selling Items by Revenue", "xlabel": "Item Name", "ylabel": "Revenue"}}}
            ---

            Lets think step by step.

            User query: 
            """ + user_query +
            """
            
            Response: 
            """ + response)

    parser = JsonOutputParser()
    agent = st.session_state["llm"] | parser

    response_dict = agent.invoke(prompt)
    print(response_dict)

    # Check if the response is a bar chart.
    try:
        if "bar" in response_dict:
            data = response_dict["bar"]
            df = pd.DataFrame(data)
            df.set_index("columns", inplace=True)
            st.bar_chart(df)

            # Add title and labels
            st.title(response["metadata"]["title"])
            st.xlabel(response["metadata"]["xlabel"])
            st.ylabel(response["metadata"]["ylabel"])

        # Check if the response is a line chart.
        if "line" in response_dict:
            data = response_dict["line"]
            df = pd.DataFrame(data)
            df.set_index("columns", inplace=True)
            st.line_chart(df)

            # Add title and labels
            st.title(response["metadata"]["title"])
            st.xlabel(response["metadata"]["xlabel"])
            st.ylabel(response["metadata"]["ylabel"])

        # Check if the response is a histogram.
        if "histogram" in response_dict:
            data = response_dict["histogram"]
            df = pd.DataFrame(data)
            df.set_index("column", inplace=True)
            st.hist(df)

            # Add title and labels
            st.title(response["metadata"]["title"])
            st.xlabel(response["metadata"]["xlabel"])
            st.ylabel(response["metadata"]["ylabel"])

        # Check if the response is a table.
        if "table" in response_dict:
            data = response_dict["table"]
            df = pd.DataFrame(data["data"], columns=data["columns"])
            st.table(df)
    except:
        st.error("Plot creation unsuccessful.", icon = "🚨")

def improve_user_query(user_query: str) -> str:
    # ---
    # # Important Information for Context:
    # ## Database Table Descriptions:
    #     * Categories- the names and ids of Categories that respond to the Items.
    #     * Clients- the names, ids, and company_types of Clients that respond to the Orders.
    #     * Items- information that responds to the items of all suppliers.
    #     * Suppliers- ids and names of Suppliers.
    #     * Orders- the date and ids of Orders, in addition to the client id associated with each.
    #     * OrderItems- for each order id, the items, qty, and price at order.
    # ## Database Schema:
    #     CREATE TABLE IF NOT EXISTS Suppliers (
    #     supplier_id INT AUTO_INCREMENT PRIMARY KEY,
    #     name VARCHAR(255) NOT NULL
    #     );
    #     CREATE TABLE IF NOT EXISTS Categories (
    #     category_id INT AUTO_INCREMENT PRIMARY KEY,
    #     name VARCHAR(255) NOT NULL
    #     );
    #     CREATE TABLE IF NOT EXISTS Items (
    #     item_id INT AUTO_INCREMENT PRIMARY KEY,
    #     name VARCHAR(255) NOT NULL,
    #     supplier_id INT,
    #     category_id INT,
    #     supplier_item_number VARCHAR(255),
    #     universal_product_code VARCHAR(255),
    #     unit_of_measure VARCHAR(50),
    #     packing VARCHAR(50),
    #     units FLOAT,
    #     unit_price FLOAT,
    #     total_packing_price FLOAT,
    #     brand TEXT,
    #     description TEXT,
    #     FOREIGN KEY (supplier_id) REFERENCES Suppliers(supplier_id),
    #     FOREIGN KEY (category_id) REFERENCES Categories(category_id)
    #     );
    #     CREATE TABLE IF NOT EXISTS Clients (
    #     client_id INT AUTO_INCREMENT PRIMARY KEY,
    #     name VARCHAR(255) NOT NULL,
    #     company_type VARCHAR(100),
    #     contact_info TEXT
    #     );
    #     CREATE TABLE IF NOT EXISTS Orders (
    #     order_id INT AUTO_INCREMENT PRIMARY KEY,
    #     client_id INT,
    #     order_date DATE,
    #     FOREIGN KEY (client_id) REFERENCES Clients(client_id)
    #     );
    #     CREATE TABLE IF NOT EXISTS OrderItems (
    #     order_item_id INT AUTO_INCREMENT PRIMARY KEY,
    #     order_id INT,
    #     item_id INT,
    #     quantity INT,
    #     price_at_order FLOAT,
    #     FOREIGN KEY (order_id) REFERENCES Orders(order_id),
    #     FOREIGN KEY (item_id) REFERENCES Items(item_id)
    #     # );
    # ---

    instructions = ("""
    Regenerate the user query as best you can, if needed, so that it can be interpreted as best as possible by an llm. Make it more specific and clear. 
    Expand on the user query if needed. For example, if plotting the data seems necessary, change the response as such. 
    Or, if the user query is not clear, make it clearer. Or in another case, if it is too complex, simplify it. The goal is to have the query result in an insightful answer.
    Keep the response short and to the point, and only reply with the regenerated version of the user query. 
                    
    # Important Information for Context:
    ## Database Table Descriptions:
        * Categories- the names and ids of Categories that respond to the Items.
        * Clients- the names, ids, and company_types of Clients that respond to the Orders.
        * Items- information that responds to the items of all suppliers.
        * Suppliers- ids and names of Suppliers.
        * Orders- the date and ids of Orders, in addition to the client id associated with each.
        * OrderItems- for each order id, the items, qty, and price at order.

    Example #1:
        User Query: What are the top selling items? Plot them.
        Response: What are our top selling items? Plot them by revenue.
                    
    Example #2:
        User Query: What clients have ordered the most items?
        Response: What clients have ordered the most items? Include each client's number of items ordered, the total revenue, and the average revenue per order.
                    
    Example #3:
        User Query: What is the revenue by order categories?
        Response: What is the revenue per order category? Plot them on a bar chart.

    User Query: 
    """ + user_query)

    improved_query = st.session_state["llm"].invoke(instructions)
    print(improved_query)

    return improved_query

def init_db(user: str, password: str, host: str, port: str, database: str) -> SQLDatabase:
    db_uri = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"
    return SQLDatabase.from_uri(db_uri)

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = [
        AIMessage(content="Hello! I am your Allec Marketplace Agent. How can I help you today?")
    ]

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
                "marco",
                "marco1234",
                "34.148.197.141",
                "3306",
                "allecmarketplace",
            )

            st.session_state["db"] = db

            with st.spinner("Establishing connections..."):
                if "llm" not in st.session_state:
                    # st.session_state["llm"] = ChatAnthropicVertex(name="claude-3-opus@20240229", temperature=0, streaming=False)

                    st.session_state["llm"] = VertexAI(model_name="gemini-1.5-flash-001", temperature=0.0)
                    startup_response = st.session_state["llm"].invoke("This invocation is to assure faster responses!")

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
    improved_user_query = improve_user_query(user_query)

    st.session_state["chat_history"].append(HumanMessage(content=user_query))
    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        with st.spinner("Fetching answer..."):
            response = st.session_state["agent"].invoke({"input":improved_user_query, "supplier":st.session_state["supplier"], "chat_history":st.session_state["chat_history"]})
            st.markdown(response["output"])

        if query_asks_for_plotting(improved_user_query):
            with st.spinner("Generating plot..."):
                llm_plotter(improved_user_query, response["output"])

    st.session_state["chat_history"].append(AIMessage(content=response["output"]))