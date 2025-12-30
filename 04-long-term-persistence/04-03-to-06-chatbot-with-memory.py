from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.embeddings import init_embeddings
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
from typing import TypedDict, Annotated
from operator import add
from datetime import datetime

from langgraph.store.postgres import PostgresStore

llm = ChatOpenAI(model="gpt-4o")

class ChatMessagesState(TypedDict):
    """Extended state that includes memory context"""
    messages: Annotated[list, add]
    memory_context: str  # Retrieved memories will be stored here

# =================================================================
# NODE 1: MEMORY RETRIEVAL - RETRIEVE MEMORIES FROM STORE
# =================================================================

def retrieve_memories(state: ChatMessagesState, config: RunnableConfig, store: BaseStore):
    """
    Memory retrieval node that:
    1. Gets user_id from config
    2. Searches for user's memories in the store
    3. Formats memories into context string
    4. Stores context in state for chatbot to use
    """
    
    # Get user_id from config
    user_id = config["configurable"].get("user_id", "default_user")
    
    print(f"\n{'='*70}")
    print(f"📖 MEMORY RETRIEVAL NODE")
    print(f"{'='*70}")
    print(f"Retrieving memories for user: {user_id}")
    
    
    # SEARCH FOR USER'S MEMORIES
    
    print("\n🔍 SEARCHING STORE...")
    
    # Search for all memories about this user
    user_memories_namespace = (user_id, "memories")
    memories = store.search(
        user_memories_namespace,
        query="What are the facts about this user?"
    )
    
    # Build context from memories
    memory_context = ""
    if memories:
        print(f"✓ Found {len(memories)} memories")
        memory_texts = []
        for i, memory in enumerate(memories, 1):
            text = memory.value.get("text", "")
            print(f"{i}. {text}")
            memory_texts.append(text)
        
        # Format as context string
        memory_context = "\n".join([f"- {text}" for text in memory_texts])
    else:
        print("ℹ️  No memories found (new user)")
        memory_context = ""
    
    print(f"{'='*70}\n")
    
    # Return memory context in state
    return {"memory_context": memory_context}

# ================================================================
# NODE 2: CHATBOT - GENERATE RESPONSE USING RETRIEVED MEMORIES
# ================================================================

def chatbot(state: ChatMessagesState, config: RunnableConfig):
    """
    Chatbot node that:
    1. Reads memory context from state (retrieved by previous node)
    2. Generates a personalized response using the memory context
    
    Note: No store parameter needed - memories already in state!
    """
    
    # Get user_id from config
    user_id = config["configurable"].get("user_id", "default_user")
    
    print(f"\n{'='*70}")
    print(f"🤖 CHATBOT NODE")
    print(f"{'='*70}")
    print(f"Generating response for user: {user_id}")
    

    # USE MEMORY CONTEXT FROM STATE
   
    print("\n💭 USING MEMORY CONTEXT...")
    
    memory_context = state.get("memory_context", "")
    
    if memory_context:
        print("✓ Using retrieved memories for personalization")
    else:
        print("ℹ️  No memory context available")
    
    # -------------------------------------------------------------
    # GENERATE RESPONSE WITH MEMORY CONTEXT
    # -------------------------------------------------------------
    print("\n💬 GENERATING RESPONSE...")
    
    # Create system message with memory context
    if memory_context:
        system_prompt = f"""You are a helpful assistant with memory of past conversations.

        What you remember about this user:
        {memory_context}

        Use this information to personalize your response. Be natural and conversational."""
    else:
        system_prompt = "You are a helpful assistant. This is your first conversation with this user."
    
    # Build messages
    messages = [
        SystemMessage(content=system_prompt),
        *state["messages"]
    ]
    
    # Generate response
    response = llm.invoke(messages)
    print(f"   ✓ Response generated: {response.content[:80]}...")
    print(f"{'='*70}\n")
    
    return {"messages": [response]}


# ===============================================================
# NODE 3: MEMORY EXTRACTION - EXTRACT AND SAVE NEW MEMORIES
# ===============================================================

def extract_and_save_memories(state: ChatMessagesState, config: RunnableConfig, store: BaseStore):
    """
    Memory extraction node that:
    1. Analyzes the conversation
    2. Extracts facts worth remembering
    3. Saves new memories to the store
    """
    
    # Get user_id from config
    user_id = config["configurable"].get("user_id", "default_user")
    
    print(f"\n{'='*70}")
    print(f"🧠 MEMORY EXTRACTION NODE")
    print(f"{'='*70}")
    print(f"Extracting and saving memories for user: {user_id}")
    
    
    # GET RECENT CONVERSATION
   
    print("\n📝 ANALYZING CONVERSATION...")
    
    # Get the last two messages (user message and assistant response)
    if len(state["messages"]) >= 2:
        user_message = state["messages"][-2].content
        assistant_message = state["messages"][-1].content
    else:
        print("⚠️  Not enough messages to extract from")
        print(f"{'='*70}\n")
        return state
    
    print(f"User said: {user_message[:60]}...")
    print(f"Assistant said: {assistant_message[:60]}...")
    
    # EXTRACT MEMORABLE FACTS
    
    print("\n🔍 EXTRACTING FACTS...")
    
    # Ask LLM to extract memorable facts
    extract_prompt = f"""Look at this conversation and extract any facts worth remembering about the user.

    User: {user_message}
    Assistant: {assistant_message}

    List each fact on a new line starting with a dash (-).
    Only include clear, factual information about the USER (not about the assistant).
    If there are no facts to remember, respond with: NONE

    Examples of good facts:
    - User's name is Alice
    - User works as a teacher
    - User enjoys hiking
    - User is learning Python

    Examples of bad facts (don't include these):
    - The assistant was helpful
    - We had a conversation
    - The user asked a question"""
    
    extraction = llm.invoke(extract_prompt).content
    print(f"Extraction result: {extraction[:80]}...")
    
    # SAVE EXTRACTED FACTS TO STORE
    
    print("\n💾 SAVING TO STORE...")
    
    # Save extracted facts
    if "NONE" not in extraction.upper():
        lines = [line.strip() for line in extraction.split("\n") if line.strip().startswith("-")]
        
        saved_count = 0
        for line in lines:
            fact = line[1:].strip()  # Remove the dash
            if fact and len(fact) > 5:  # Basic validation
                # Generate a unique key based on timestamp
                memory_key = f"memory_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
                
                # Save to store
                store.put(
                    namespace=(user_id, "memories"),
                    key=memory_key,
                    value={
                        "text": fact,
                        "timestamp": datetime.now().isoformat(),
                        "source": "conversation"
                    }
                )
                print(f"✓ Saved: {fact}")
                saved_count += 1
        
        if saved_count == 0:
            print("ℹ️  No valid facts to save")
    else:
        print("ℹ️  No new facts to save")
    
    print(f"{'='*70}\n")
    
    return state


# ================================================================
# BUILD THE GRAPH
# ================================================================

# Create graph
builder = StateGraph(ChatMessagesState)

# Add nodes
builder.add_node("retrieve_memories", retrieve_memories)
builder.add_node("chatbot", chatbot)
builder.add_node("extract_memories", extract_and_save_memories)

# Define edges: retrieve -> chatbot -> extract -> END
builder.add_edge(START, "retrieve_memories")
builder.add_edge("retrieve_memories", "chatbot")
builder.add_edge("chatbot", "extract_memories")
builder.add_edge("extract_memories", END)

# Create memory systems
checkpointer = MemorySaver() # For conversation history

store_embeddings_model = init_embeddings("openai:text-embedding-3-small")
""" store = InMemoryStore(
    index={
        "embed": store_embeddings_model,  # Embedding provider
        "dims": 1536,  # Embedding dimensions
        "fields": ["text", "$"]  # Fields to embed ("text" + $ = ALL other fields)
    }
) # For long-term memory """

DB_URI = "postgresql://postgres:root@localhost:5432/langgraph_ltm?sslmode=disable"

with PostgresStore.from_conn_string(
    DB_URI,
    index={
        "dims": 1536,
        "embed": store_embeddings_model,
        "fields": ["text", "$"]  
    }
) as store:

    #store.setup() # Do this once to run migrations

    # Compile graph
    graph = builder.compile(
        checkpointer=checkpointer,
        store=store
    )


    # ==================================================================
    # DEMONSTRATION 1: FIRST CONVERSATION
    # ==================================================================

    print("\n" + "="*70)
    print("DEMONSTRATION 1: First Conversation with Sarah")
    print("="*70)

    # Configuration for Sarah
    config = {
        "configurable": {
            "thread_id": "chat_001",
            "user_id": "sarah"
        }
    }

    # Turn 1: Introduction
    print("\n" + "-"*70)
    print("TURN 1: Introduction")
    print("-"*70)

    sarah_message_1 = "Hi! My name is Sarah and I'm a data scientist."

    result = graph.invoke(
        {"messages": [HumanMessage(content=sarah_message_1)]},
        config=config
    )

    print(f"\n📨 USER: {sarah_message_1}")
    print(f"📤 ASSISTANT: {result['messages'][-1].content}")

    # Turn 2: Share work info
    print("\n" + "-"*70)
    print("TURN 2: Sharing work information")
    print("-"*70)

    sarah_message_2 = "I'm currently working on a machine learning project using Python and TensorFlow."

    result = graph.invoke(
        {"messages": [HumanMessage(content=sarah_message_2)]},
        config=config
    )

    print(f"\n📨 USER: {sarah_message_2}")
    print(f"📤 ASSISTANT: {result['messages'][-1].content}")

    # Turn 3: Share hobbies
    print("\n" + "-"*70)
    print("TURN 3: Sharing hobbies")
    print("-"*70)

    sarah_message_3 = "In my free time, I love playing guitar and going on weekend hikes."

    result = graph.invoke(
        {"messages": [HumanMessage(content=sarah_message_3)]},
        config=config
    )

    print(f"\n📨 USER: {sarah_message_3}")
    print(f"📤 ASSISTANT: {result['messages'][-1].content}")

    # Turn 4: Share dietary preference
    print("\n" + "-"*70)
    print("TURN 4: Sharing preferences")
    print("-"*70)

    sarah_message_4 = "I'm vegetarian and I prefer coffee over tea."

    result = graph.invoke(
        {"messages": [HumanMessage(content=sarah_message_4)]},
        config=config
    )

    print(f"\n📨 USER: {sarah_message_4}")
    print(f"📤 ASSISTANT: {result['messages'][-1].content}")

    # =================================================================
    # INSPECT STORED MEMORIES
    # =================================================================

    print("\n\n" + "="*70)
    print("INSPECTING STORED MEMORIES")
    print("="*70)

    # Search all memories for Sarah
    memories = store.search(
        ("sarah", "memories"),
        query="What are the facts about this user?"
    )

    print(f"\n📊 Total memories stored for Sarah: {len(memories)}\n")

    for i, memory in enumerate(memories, 1):
        print(f"{i}. {memory.value['text']}")
        print(f"Key: {memory.key}")
        print(f"Timestamp: {memory.value['timestamp']}")
        print(f"Source: {memory.value['source']}")
        print(f"Created at: {memory.created_at}")
        print()


    # =============================================================
    # DEMONSTRATION 2: SARAH RETURNS (New Thread, Same User)
    # =============================================================

    print("="*70)
    print("DEMONSTRATION 2: Sarah Returns (Different Day)")
    print("="*70)
    print("New conversation thread - memories should persist!\n")

    # New thread, same user
    new_config = {
        "configurable": {
            "thread_id": "chat_005",  # Different thread
            "user_id": "sarah"         # Same user
        }
    }

    # Turn 1: Sarah returns
    print("-"*70)
    print("DAY 2 - First Message")
    print("-"*70)

    sarah_day2_message_1 = "Good morning! What do you remember about me?"

    result = graph.invoke(
        {"messages": [HumanMessage(content=sarah_day2_message_1)]},
        new_config
    )

    print(f"\n📨 USER: {sarah_day2_message_1}")
    print(f"📤 ASSISTANT: {result['messages'][-1].content}")

    # Turn 2: Reference stored info
    print("\n" + "-"*70)
    print("DAY 2 - Second Message")
    print("-"*70)

    sarah_day2_message_2 = "Can you suggest a lunch place considering my dietary preferences?"

    result = graph.invoke(
        {"messages": [HumanMessage(content=sarah_day2_message_2)]},
        new_config
    )

    print(f"\n📨 USER: {sarah_day2_message_2}")
    print(f"📤 ASSISTANT: {result['messages'][-1].content}")


    # ==============================================================
    # DEMONSTRATION 3: DIFFERENT USER (Memory Isolation)
    # ==============================================================

    print("\n\n" + "="*70)
    print("DEMONSTRATION 3: Different User - John")
    print("="*70)
    print("Testing memory isolation between users\n")

    # Different user
    john_config = {
        "configurable": {
            "thread_id": "chat_010",
            "user_id": "john"
        }
    }

    # Turn 1: John queries chatbot about himself
    print("-"*70)
    print("JOHN'S FIRST MESSAGE")
    print("-"*70)

    john_message_1 = "Hey! I am John, What do you know about me?"

    result = graph.invoke(
        {"messages": [HumanMessage(content=john_message_1)]},
        john_config
    )

    print(f"\n📨 USER (John): {john_message_1}")
    print(f"📤 ASSISTANT: {result['messages'][-1].content}")