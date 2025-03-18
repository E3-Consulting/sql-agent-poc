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
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os

set_verbose(True)

CREDENTIALS, PROJECT_ID = google.auth.default()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")

st.set_page_config(page_title="Sales Agent", page_icon=":mortar_board:")
st.logo("Allec-New-Logo-M-Icon.png")

col1, col2 = st.columns([1, 3])
with col1:
    st.image("allec_logo.webp", width=150, output_format="PNG", use_column_width=True)
with col2:
    st.title("Marketplace Agent")

load_dotenv()


# TODO: refine code to make it more efficient
def agent_init(db: SQLDatabase, model):
    print(db.table_info)
    # TODO- safety to not allow it to filter other suppliers
    prompt_template = PromptTemplate(
        input_variables=[
            "input",
            "agent_scratchpad",
            "tools",
            "tool_names",
            "supplier",
            "supplier_id",
            "chat_history",
        ],
        template="""
    # Requirements
        ## Instructions
        You are a helpful, friendly assistant for a client of an app called Allec. Allec is a marketplace for suppliers to sell their items.
        Your task is to answer to the user's requests by using the following tools: {tools}
        The supplier you are interacting with is {supplier}. Its id in the table is {supplier_id}. You must filter the database based on the name of the supplier as it appared in the previous sentence in every query.
        If a query returns no data after checking it with the tools, respond with "There is no data available to answer that".
        Finally, some details in the database may be written in spanish, however it must be noted that: **many columns will not be direct translations, list the values of column that is needed before making these assumptions**.

        ## Output Specification
        Provide an informative, and accurate answer to the question based on your query results. Once you have a final answer, stop the thought process and provide your output after "Final Answer:".
        If you get None as a query result, respond with "There is no data available to answer that". Include all information from your final query in your natural language answer.
        If the query includes a request to plot data, append "Plotted data coming up!" to the end of the final answer after three new lines.
        If the query is a follow-up to plot data of the results of a previous query, simply respond "Plotted data coming up!" and don't use any tools.
        Try to format your response as a table if it makes sense.

        ### Extra specifications:
        Do not include code block markers in your action input (e.g., ```sql ```, [SQL: ], etc.). 'Action' should only contain the name of the tool to be used. The database is in MySQL.

        ## STOP CONDITION:
        Once a sql query is executed that returns a result that answers the question, return the Final Answer.

    # Database Table Descriptions:
        * Categories- the names and ids of Categories that respond to the Items.
        * Clients- the names, ids, and company_types of Clients that respond to the Orders.
        * Items- information that responds to the items of all suppliers.
        * Suppliers- ids and names of Suppliers.
        * Orders- the date and ids of Orders, in addition to the client id associated with each.
        * OrderItems- for each order id, the items, qty, and price at order.

        ## Database Schema- use this information to understand the structure of the database. 
            """
        + db.table_info
        + """

    # Thought Process
        ## Structure
        Use the following structure for your thought process, do not reformat this:
        Human: user's input
        Thought: What do I need to do?
        Action: What tool should be utilized from the following list {tool_names}
        Action Input: input for tool
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

    # Chat History- look back to this in case of follow-up questions. They will be referring to the last messages. Any dictionary included here represents a visualization.
    {chat_history}

    # Input    
    Human: {input}

    # Your thought process and answer
    {agent_scratchpad}
    """,
    )

    # Create the agent
    agent = create_sql_agent(
        llm=model,
        toolkit=CustomSQLDatabaseToolkit(db=db, llm=model),
        prompt=prompt_template,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
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
    # DO NOT RESPOND WITH BACKTICKS (i.e., ```json ... ```).

    #         Instead of responding like this:
    # Respond like this:

    #         {"bar": {"columns": ["Items", "Revenue"], "data": [["\"Unius Olive Oil Arbequina 750ml\"", "\"Olive Oil Casanova Estates \\\"L\'Olio Toscano\\\" 2020 500ml\"", "\"Pic colomini d'Aragona Ex tra Virgin Olive Oil 2020 500ml\"", "\"Sol del Silenc io Premium Oil 500ml\"", "\"Ume Juice (Can) - Pack of 30 8.45oz\"", "\"Mik an Juice (Can) - Pack of 30 8.45oz\"", "\"Apple Juice (Can) - Pac k of 30 8.45oz\"", "\"Ume Juice - Pack of 24 8.45oz\"", "\"The 1 Water / Wine Glass (No Stem) - 6 Unit Presentation Gift Pack\"", "\"Mik an Juice - Pack of 24 8.45oz\"", "\"Apple Juice - Pac k of 24 8.45oz\"", "\"Deep Sea Water - Pack of 6 2L\"", "\"Young Wine Decanter\"", "\"Deep Sea Water - Pack of 24 17.5oz\"", "\"Polis hing Cloth\"", "\"Water Carafe (Pre-Order Only)\"", "\"Mature Wine Decanter\"", "\"Water / Wine Glass (Pre-Order Only)\"", "\"The 1 Glass - 2 Unit Presentation Gift Pack\"", "\"The 1 Glass\""], [97.98,25.12,66.07,16.39,66.06,1.33,77.58,33.37,25.49,76.2,92.45,43.17,60.93,64.13,72.45,57.05,4.14,11.11,47.49,23.33 ]]}, "metadata": {"title": "Top Selling Items by Revenue", "xlabel": "Item Name", "ylabel": "Revenue"}}
    prompt = (
        """
            In the end of this prompt, there are two values: user query and result. The user query is a request to retrieve and plot data. The result is the retrieved data in natural language.
            We want to convert the retrieved data into a JSON format. Further, based on the user query, we want to plot as requested. If the request is vague, we want to assume the best fit from the retrieved data.
            The user query would be best to indicate what the column names are. For example, if the user asks for the top selling items, the column would be "Item Name" and "Quantity sold" or "Item Name" and "Revenue", depending on what the user specifies. However, if the user is not specific, feel free to guess for yourself based on the result value.
            There are four types of plots: table, bar, line, and histogram. If the user is not specific of the choice, please choose the best you can from the list.
            DO NOT RETURN AN EMPTY JSON.

            Example Response:
             
            ```json {"bar": {'columns': ['Item Name', 'Revenue'], 'data': [['"Unius Olive Oil Arbequina 750ml"', 26827.5], ['"Olive Oil Casanova Estates "L\'Olio Toscano" 2020 500ml"', 24575.0], ['"Pic colomini d\'Aragona Ex tra Virgin Olive Oil 2020 500ml"', 22348.75], ['"Sol del Silenc io Premium Oil 500ml"', 20250.0], ['"Ume Juice (Can) - Pack of 30 8.45oz"', 18317.5], ['"Mik an Juice (Can) - Pack of 30 8.45oz"', 16460.0], ['"Apple Juice (Can) - Pac k of 30 8.45oz"', 14643.75], ['"Ume Juice - Pack of 24 8.45oz"', 12950.0], ['"The 1 Water / Wine Glass (No Stem) - 6 Unit Presentation Gift Pack"', 12901.00048828125], ['"Mik an Juice - Pack of 24 8.45oz"', 11407.5], ['"Apple Juice - Pac k of 24 8.45oz"', 9945.0], ['"Deep Sea Water - Pack of 6 2L"', 9888.75], ['"Young Wine Decanter"', 7940.199890136719], ['"Deep Sea Water - Pack of 24 17.5oz"', 7610.0], ['"Polis hing Cloth"', 7297.5], ['"Water Carafe (Pre-Order Only)"', 6533.75], ['"Mature Wine Decanter"', 5807.790008544922], ['"Water / Wine Glass (Pre-Order Only)"', 5430.0], ['"The 1 Glass - 2 Unit Presentation Gift Pack"', 2985.6500244140625], ['"The 1 Glass"', 2497.0999755859375]]}, 'metadata': {'title': 'Revenue of Each Item', 'xlabel': 'Item Name', 'ylabel': 'Revenue'}} ```
            
            ---

            For the following query, if it requires drawing a table, reply as follows:
            {"table": {"columns": ["column1", "column2", ...], "data": [[value1, value2, ...], [value1, value2, ...], ...]}}

            If the query requires creating a bar chart, reply as follows:
            {"bar": {"columns": ["A", "B", "C", ...], "data": [[value1, value2, ...], [value1, value2, ...], ...]}, "metadata": {"title": "Table Title", "xlabel": "A", "ylabel": "B"}}

            If the query requires creating a line chart, reply as follows:
            {"line": {"columns": ["A", "B", "C", ...], "data": [[value1, value2, ...], [value1, value2, ...], ...]}, "metadata": {"title": "Table Title", "xlabel": "A", "ylabel": "B"}}

            If the query requires creating a histogram, reply as follows:
            {"histogram": {"columns": ["A"], "data": [25, 24, 10, ...]}, "metadata": {"title": "Table Title", "xlabel": "X Label", "ylabel": "Y Label"}}

            Return all output as a string.

            ---

            User query: 
            """
        + user_query
        + """
            
            Result: 
            """
        + response
    )

    parser = JsonOutputParser()

    # response_format_checker =

    agent = st.session_state["llm"] | (lambda x: x.replace("\n", "")) | parser
    response_dict = agent.invoke(prompt)
    print(response_dict)

    # print(response_dict["bar"]['columns'])

    # print(type(response_dict["bar"]['columns']))

    return response_dict


def generate_plot(response_dict: dict) -> None:
    try:
        # Check if the response is a bar chart.
        if "bar" in response_dict:
            print("Processing bar chart data...")
            data = response_dict["bar"]
            print("Bar data:", data)

            if "metadata" not in response_dict and "metadata" not in data:
                print("Metadata key not found in bar data")
            else:
                try:
                    metadata = data["metadata"]
                except:
                    metadata = response_dict["metadata"]
                print("Metadata:", metadata)

            df = pd.DataFrame(data["data"], columns=data["columns"])
            print("DataFrame:", df)

            # Add title and labels
            st.title(metadata["title"])
            st.bar_chart(data=df, x=metadata["xlabel"], y=metadata["ylabel"])

        # Check if the response is a line chart.
        if "line" in response_dict:
            print("Processing line chart data...")
            data = response_dict["line"]

            try:
                metadata = data["metadata"]
            except:
                metadata = response_dict["metadata"]

            df = pd.DataFrame(data["data"], columns=data["columns"])
            print("Line DataFrame:", df)
            # Add title and labels
            st.title(metadata["title"])
            st.line_chart(data=df, x=metadata["xlabel"], y=metadata["ylabel"])

        # Check if the response is a table.
        if "table" in response_dict:
            print("Processing table data...")
            data = response_dict["table"]
            df = pd.DataFrame(data["data"], columns=data["columns"])
            print("Table DataFrame:", df)
            st.table(df)

    except Exception as e:
        print(f"An error occurred: {e}")
        st.error("Plot creation unsuccessful.", icon="🚨")


def improve_user_query(user_query: str) -> str:
    instructions = (
        """
    Regenerate the user query as best you can, if needed, so that it can be interpreted as best as possible by an llm. Make it more specific and clear. 
    Expand on the user query if needed. For example, if plotting the data seems necessary, change the response as such. 
    Or, if the user query is not clear, make it clearer. Or in another case, if it is too complex, simplify it. The goal is to have the query result in an insightful answer.
    Keep the response short and to the point, and only reply with the regenerated version of the user query. 
    If it seems like a follow-up question, leave it as-is.

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
        Response: What is the revenue per order category?
                    
    # Chat History for context of query:
    """
        + str(st.session_state["chat_history"])
    ) + (
        """

    User Query: 
    """
        + user_query
    )

    improved_query = st.session_state["llm"].invoke(instructions)
    print(improved_query)

    return improved_query


def init_db(
    user: str, password: str, host: str, port: str, database: str
) -> SQLDatabase:
    db_uri = f"mysql://{user}:{password}@{host}:{port}/{database}"
    return SQLDatabase.from_uri(db_uri)


if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = [
        AIMessage(
            content="Hello! I am your Allec Marketplace Agent. How can I help you today?"
        )
    ]

with st.sidebar:
    st.subheader("Login")
    st.write("Hello! Enter your credentials to get started.")

    st.session_state["supplier"] = st.selectbox(
        "Supplier",
        [
            "AMBROSIA",
            "BALLESTER",
            "CAJITA VALLEJO",
            "CC1 BEER DISTRIBUTOR, INC.",
            "FINE WINE IMPORTS",
            "MENDEZ & COMPANY",
            "OCEAN LAB BREWING CO",
            "PAN AMERICAN WINE & SPIRITS",
            "PUERTO RICO SUPPLIES GROUP",
            "QUINTANA HERMANOS",
            "SEA WORLD",
            "SERRALLES",
            "V. SUAREZ",
            "JOSE SANTIAGO",
            "ASIA MARKET",
            "B. FERNANDEZ & HNOS. INC.",
            "CARIBE COMPOSTABLES",
            "DON PINA",
            "DROUYN",
            "KIKUET",
            "LA CEBA",
            "LEODI",
            "MAVE INC",
            "MAYS OCHOA",
            "NORTHWESTERN SELECTA",
            "PACKERS",
            "PROGRESSIVE",
            "QUALITY CLEANING PRODUCTS",
            "TO RICO",
            "ULINE",
            "BAGUETTES DE PR",
            "BESPOKE",
            "CAIMÁN",
            "CARDONA AGRO INDUSTRY",
            "CHURRO LOOP",
            "COMEX",
            "DONA LOLA",
            "GFG",
            "HL HERNANDEZ",
            "IMPERIAL DADE",
            "INPROMO",
            "JUGOS HENRY",
            "Northwestern Selecta",
            "PAPELERIA",
            "PLAZA CELLARS",
            "PR SUPPLIES",
            "TACONAZO",
            "WEBSTAURANT STORE",
            "COCA-COLA PUERTO RICO BOTTLES",
            "CR - DISTRIBUTIONS BALLESTER",
            "CR - DISTRIBUTIONS BE WAFFLED",
            "CR - DISTRIBUTIONS GREEN VALLEY",
            "DADE PAPER",
            "EL VIANDON",
            "FRIGORIFICO",
            "GBS",
            "NATURAL FOOD CENTER",
            "VAQUERIA TRES MONJITAS",
            "YC DEPOT",
            "BIO-WARE",
            "BLESS PRODUCE",
            "ELABORACION DE PASTELILLOS",
            "FINCA CAMAILA",
            "FINCA LAHAM",
            "HIDROPONICO LOS HERMANOS",
            "JESÚS CUEVAS",
            "JUGOS LA BORINQUEÑA",
            "LEVAIN",
            "LIQUIDACIONES FELICIANO",
            "MEDALLA DISTRIBUTORS",
            "MONDA QUE MONDA",
            "MR. SPECIAL",
            "NORIS UNIFORMS- AGUADILLA",
            "PASTA ASORE",
            "PEREZ OFFICE",
            "PLATANO  HIO",
            "POPEYE'S ICE FACTORY",
            "PR VERDE FOOD DISTRIBUITOR",
            "PRODUCTOS DON GADY",
            "PRODUCTOS EL PLANTILLERO",
            "PRODUCTOS MI ENCANTO",
            "QUESO DEL PAÍS LA ESPERANZA",
            "RINCÓN RUM INC",
            "TU PLÁTANO",
            "VITIN",
            "WESTERN PAPER",
            "WAHMEY",
            "Luxo Wine",
        ],
    )
    st.text_input("Username", key="username")
    st.text_input("Password", type="password", key="password")

    if st.button("Login"):
        with st.spinner("Connecting to database..."):
            db = init_db(
                "marco",
                "marco1234",
                "104.196.25.151",
                "3306",
                "allecmarketplace",
            )

            st.session_state["db"] = db
            st.session_state["engine"] = create_engine(
                "mysql://marco:marco1234@104.196.25.151:3306/allecmarketplace"
            )
            Session = sessionmaker(bind=st.session_state["engine"])
            st.session_state["session"] = Session()

            with st.spinner("Establishing connections..."):
                if "llm" not in st.session_state:
                    # st.session_state["llm"] = ChatAnthropicVertex(name="claude-3-opus@20240229", temperature=0, streaming=False)

                    st.session_state["llm"] = VertexAI(
                        model_name="gemini-1.5-flash-001", temperature=0.0
                    )
                    startup_response = st.session_state["llm"].invoke(
                        "This invocation is to establish an initial connection to the model!"
                    )

                if "agent" not in st.session_state:
                    st.session_state["agent"] = agent_init(
                        db=st.session_state["db"], model=st.session_state["llm"]
                    )

            st.success("Connected to Marketplace!")


suppliers = {
    "AMBROSIA": 1,
    "BALLESTER": 2,
    "CAJITA VALLEJO": 3,
    "CC1 BEER DISTRIBUTOR, INC.": 4,
    "FINE WINE IMPORTS": 5,
    "MENDEZ & COMPANY": 6,
    "OCEAN LAB BREWING CO": 7,
    "PAN AMERICAN WINE & SPIRITS": 8,
    "PUERTO RICO SUPPLIES GROUP": 9,
    "QUINTANA HERMANOS": 10,
    "SEA WORLD": 11,
    "SERRALLES": 12,
    "V. SUAREZ": 13,
    "JOSE SANTIAGO": 14,
    "ASIA MARKET": 15,
    "B. FERNANDEZ & HNOS. INC.": 16,
    "CARIBE COMPOSTABLES": 17,
    "DON PINA": 18,
    "DROUYN": 19,
    "KIKUET": 20,
    "LA CEBA": 21,
    "LEODI": 22,
    "MAVE INC": 23,
    "MAYS OCHOA": 24,
    "NORTHWESTERN SELECTA": 25,
    "PACKERS": 26,
    "PROGRESSIVE": 27,
    "QUALITY CLEANING PRODUCTS": 28,
    "TO RICO": 29,
    "ULINE": 30,
    "BAGUETTES DE PR": 31,
    "BESPOKE": 32,
    "CAIMÁN": 33,
    "CARDONA AGRO INDUSTRY": 34,
    "CHURRO LOOP": 35,
    "COMEX": 36,
    "DONA LOLA": 37,
    "GFG": 38,
    "HL HERNANDEZ": 39,
    "IMPERIAL DADE": 40,
    "INPROMO": 41,
    "JUGOS HENRY": 42,
    "Northwestern Selecta": 43,
    "PAPELERIA": 44,
    "PLAZA CELLARS": 45,
    "PR SUPPLIES": 46,
    "TACONAZO": 47,
    "WEBSTAURANT STORE": 48,
    "COCA-COLA PUERTO RICO BOTTLES": 49,
    "CR - DISTRIBUTIONS BALLESTER": 50,
    "CR - DISTRIBUTIONS BE WAFFLED": 51,
    "CR - DISTRIBUTIONS GREEN VALLEY": 52,
    "DADE PAPER": 53,
    "EL VIANDON": 54,
    "FRIGORIFICO": 55,
    "GBS": 56,
    "NATURAL FOOD CENTER": 57,
    "VAQUERIA TRES MONJITAS": 58,
    "YC DEPOT": 59,
    "BIO-WARE": 60,
    "BLESS PRODUCE": 61,
    "ELABORACION DE PASTELILLOS": 62,
    "FINCA CAMAILA": 63,
    "FINCA LAHAM": 64,
    "HIDROPONICO LOS HERMANOS": 65,
    "JESÚS CUEVAS": 66,
    "JUGOS LA BORINQUEÑA": 67,
    "LEVAIN": 68,
    "LIQUIDACIONES FELICIANO": 69,
    "MEDALLA DISTRIBUTORS": 70,
    "MONDA QUE MONDA": 71,
    "MR. SPECIAL": 72,
    "NORIS UNIFORMS- AGUADILLA": 73,
    "PASTA ASORE": 74,
    "PEREZ OFFICE": 75,
    "PLATANO HIO": 76,
    "POPEYE'S ICE FACTORY": 77,
    "PR VERDE FOOD DISTRIBUITOR": 78,
    "PRODUCTOS DON GADY": 79,
    "PRODUCTOS EL PLANTILLERO": 80,
    "PRODUCTOS MI ENCANTO": 81,
    "QUESO DEL PAÍS LA ESPERANZA": 82,
    "RINCÓN RUM INC": 83,
    "TU PLÁTANO": 84,
    "VITIN": 85,
    "WESTERN PAPER": 86,
    "WAHMEY": 87,
    "Luxo Wine": 88,
}

user_query = st.chat_input("Type query...")

for message in st.session_state["chat_history"]:
    if isinstance(message, AIMessage):
        with st.chat_message("AI", avatar="🤖"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human", avatar="👨‍💻"):
            st.markdown(message.content)
    elif isinstance(message, dict):
        with st.chat_message("AI", avatar="📊"):
            generate_plot(message)

if user_query is not None and user_query.strip() != "":
    improved_user_query = improve_user_query(user_query)

    st.session_state["chat_history"].append(HumanMessage(content=user_query))
    with st.chat_message("Human", avatar="👨‍💻"):
        st.markdown(user_query)

    with st.chat_message("AI", avatar="🤖"):
        with st.spinner("Fetching answer..."):
            response = st.session_state["agent"].invoke(
                {
                    "input": improved_user_query,
                    "supplier": st.session_state["supplier"],
                    "supplier_id": suppliers[st.session_state["supplier"]],
                    "chat_history": st.session_state["chat_history"],
                }
            )
            # Retry
            if (
                response["output"]
                == "Agent stopped due to iteration limit or time limit."
            ):
                st.markdown("Failed first attempt to fetch answer. Retrying...")
                response = st.session_state["agent"].invoke(
                    {
                        "input": improved_user_query,
                        "supplier": st.session_state["supplier"],
                        "supplier_id": suppliers[st.session_state["supplier"]],
                        "chat_history": st.session_state["chat_history"],
                    }
                )
                if (
                    response["output"]
                    == "Agent stopped due to iteration limit or time limit."
                ):
                    st.error(response["output"])
            # success
            if (
                response["output"]
                != "Agent stopped due to iteration limit or time limit."
            ):
                st.markdown(response["output"])

    if query_asks_for_plotting(improved_user_query):
        with st.chat_message("AI", avatar="📊"):
            with st.spinner("Generating plot..."):
                plot = llm_plotter(improved_user_query, response["output"])
                print(plot)
                generate_plot(response_dict=plot)
                st.session_state["plot"] = True
    else:
        st.session_state["plot"] = False

    st.session_state["chat_history"].append(AIMessage(content=response["output"]))

    # TODO- figure out how to not add plot to chat history every time.
    if st.session_state["plot"] == True:
        st.session_state["chat_history"].append(plot)
