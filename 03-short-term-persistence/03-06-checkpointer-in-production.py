from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END, MessagesState
#from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.messages import HumanMessage

llm = ChatOpenAI(model="gpt-4o")

def chatbot(state: MessagesState):
    """Chatbot node that responds to messages"""

    # Send the entire chat hsitory, including the new user message
    response = llm.invoke(state["messages"])

    return {
        "messages": [response] # Add the new ai response to the message history
    }

DB_URI = "postgresql://postgres:root@localhost:5432/langgraph_stm?sslmode=disable"

with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup() # Call only the first time

    builder = StateGraph(MessagesState)

    builder.add_node("chatbot", chatbot)

    builder.add_edge(START, "chatbot")
    builder.add_edge("chatbot", END)

    #checkpointer = MemorySaver()

    graph = builder.compile(checkpointer=checkpointer)
    # graph = builder.compile() # Without a checkpointer

    # Start conversation

    # Create configuration to apply a thread_id to the session
    config = {"configurable": {"thread_id": "chat_session_1"}}

    message_1 = "Hi! My name Fikayo, I am an AI Engineer"

    input_1 = {
        "messages": [HumanMessage(content=message_1)]
    }

    result_1 = graph.invoke(input_1, config=config)

    print(f"User: {message_1}")
    print(f"AI: {result_1['messages'][-1].content}")

    # Second Turn
    message_2 = "What's my name?"

    input_2 = {
        "messages": [HumanMessage(content=message_2)]
    }

    result_2 = graph.invoke(input_2, config=config)

    print(f"User: {message_2}")
    print(f"AI: {result_2['messages'][-1].content}")