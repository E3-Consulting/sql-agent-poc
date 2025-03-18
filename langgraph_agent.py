from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated
from langchain_core.pydantic_v1 import BaseModel, Field
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from dotenv import load_dotenv

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    supplier: str

class Response(BaseModel):
    """Final response to the user"""

    plot: dict = Field(description="""If the user did not request a plot, this will be an empty dictionary. If not, return a dictionary as instructed below:
                       
            You will return a dictionary defining a plot according to the user's request.

            Example Response:
             
            {"bar": {'columns': ['Item Name', 'Revenue'], 'data': [['"Unius Olive Oil Arbequina 750ml"', 26827.5], ['"Olive Oil Casanova Estates "L\'Olio Toscano" 2020 500ml"', 24575.0], ['"Pic colomini d\'Aragona Ex tra Virgin Olive Oil 2020 500ml"', 22348.75], ['"Sol del Silenc io Premium Oil 500ml"', 20250.0], ['"Ume Juice (Can) - Pack of 30 8.45oz"', 18317.5], ['"Mik an Juice (Can) - Pack of 30 8.45oz"', 16460.0], ['"Apple Juice (Can) - Pac k of 30 8.45oz"', 14643.75], ['"Ume Juice - Pack of 24 8.45oz"', 12950.0], ['"The 1 Water / Wine Glass (No Stem) - 6 Unit Presentation Gift Pack"', 12901.00048828125], ['"Mik an Juice - Pack of 24 8.45oz"', 11407.5], ['"Apple Juice - Pac k of 24 8.45oz"', 9945.0], ['"Deep Sea Water - Pack of 6 2L"', 9888.75], ['"Young Wine Decanter"', 7940.199890136719], ['"Deep Sea Water - Pack of 24 17.5oz"', 7610.0], ['"Polis hing Cloth"', 7297.5], ['"Water Carafe (Pre-Order Only)"', 6533.75], ['"Mature Wine Decanter"', 5807.790008544922], ['"Water / Wine Glass (Pre-Order Only)"', 5430.0], ['"The 1 Glass - 2 Unit Presentation Gift Pack"', 2985.6500244140625], ['"The 1 Glass"', 2497.0999755859375]]}, 'metadata': {'title': 'Revenue of Each Item', 'xlabel': 'Item Name', 'ylabel': 'Revenue'}}
            
            ---

            For the following query, if it requires drawing a table, reply as follows:
            {"table": {"columns": ["column1", "column2", ...], "data": [[value1, value2, ...], [value1, value2, ...], ...]}}

            If the query requires creating a bar chart, reply as follows:
            {"bar": {"columns": ["A", "B", "C", ...], "data": [[value1, value2, ...], [value1, value2, ...], ...]}, "metadata": {"title": "Table Title", "xlabel": "X Label", "ylabel": "Y Label"}}

            If the query requires creating a line chart, reply as follows:
            {"line": {"columns": ["A", "B", "C", ...], "data": [[value1, value2, ...], [value1, value2, ...], ...]}, "metadata": {"title": "Table Title", "xlabel": "X Label", "ylabel": "Y Label"}}

            If the query requires creating a histogram, reply as follows:
            {"histogram": {"columns": ["A"], "data": [25, 24, 10, ...]}, "metadata": {"title": "Table Title", "xlabel": "X Label", "ylabel": "Y Label"}}

            All strings in "columns" list, data list, and metadata, should be in double quotes,

            For example: {"columns": ["title", "ratings_count"], "data": [["Gilead", 361], ["Spider's Web", 5164]]}""")
    query_response: str = Field(description="Natural language response to the query")

class Agent:

    def __init__(self, model, tools, response, system=""):
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("improve_query", self.improve_query)
        graph.add_node("llm", self.call_model)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges(
            "llm",
            self.exists_action,
            {True: "action", False: END}
        )
        graph.add_edge("improve_query", "llm")
        graph.add_edge("action", "llm")
        graph.set_entry_point("improve_query")
        self.graph = graph.compile()
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools + [response], )

    def improve_query(self, state: AgentState):        
        user_query = state["messages"][-1].content

        prompt = ("""
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
        """) + state["messages"][:-1] + (
        """

        User Query: 
        """ + user_query)

        state["messages"][-1].content = self.model.invoke(prompt)


    def exists_action(self, state: AgentState):
        result = state['messages'][-1]
        return len(result.tool_calls) > 0

    def call_model(self, state: AgentState):
        messages = state['messages']
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        return {'messages': [message]}

    def take_action(self, state: AgentState):
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            print(f"Calling: {t}")
            if t['name'] not in self.tools:      # check for bad tool name from LLM
                print("\n ....bad tool name....")
                result = "bad tool name, retry"  # instruct LLM to retry if bad
            else:
                result = self.tools[t['name']].invoke(t['args'])
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        print("Back to the model!")
        return {'messages': results}