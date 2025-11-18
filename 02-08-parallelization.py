from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

# Define state types
class OverallState(TypedDict):
    """Main state that flows through the graph"""
    topic: str
    instagram_post: str
    twitter_post: str
    linkedin_post: str
    final_output: str

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4")

# Node 1: Generate Instagram post
def generate_instagram(state: OverallState) -> OverallState:
    """Generate an engaging Instagram post with emojis and hashtags"""
    print("📸 Instagram Generator: Creating post...")
    
    prompt = f"""
    Create an Instagram post about: {state['topic']}

    Requirements:
    - Engaging and visual language
    - 2-3 short paragraphs (150-200 words max)
    - Include relevant emojis
    - End with 5-8 relevant hashtags
    - Casual, friendly tone
    - Call-to-action to engage with the post

    Make it perfect for Instagram's audience."""
    
    instagram_post = llm.invoke(prompt).content
    
    print("✓ Instagram Generator: Complete\n")
    
    return {"instagram_post": instagram_post}


# Node 2: Generate Twitter post
def generate_twitter(state: OverallState) -> OverallState:
    """Generate a concise Twitter post"""
    print("🐦 Twitter Generator: Creating post...")
    
    prompt = f"""
    Create a Twitter post about: {state['topic']}

    Requirements:
    - Maximum 280 characters (this is crucial!)
    - Punchy and attention-grabbing
    - Include 2-3 relevant hashtags
    - Conversational tone
    - Can use emojis sparingly
    - Should spark engagement/replies

    Make it perfect for Twitter's fast-paced environment."""
    
    twitter_post = llm.invoke(prompt).content
    
    print("✓ Twitter Generator: Complete\n")
    
    return {"twitter_post": twitter_post}


# Node 3: Generate LinkedIn post
def generate_linkedin(state: OverallState) -> OverallState:
    """Generate a professional LinkedIn post"""
    print("💼 LinkedIn Generator: Creating post...")
    
    prompt = f"""
    Create a LinkedIn post about: {state['topic']}

    Requirements:
    - Professional yet engaging tone
    - 3-4 paragraphs (200-300 words)
    - Include insights or lessons learned
    - Use line breaks for readability
    - Add 3-5 professional hashtags
    - Include a thought-provoking question at the end
    - Focus on value and professional development

    Make it perfect for LinkedIn's professional audience."""
    
    linkedin_post = llm.invoke(prompt).content
    
    print("✓ LinkedIn Generator: Complete\n")
    
    return {"linkedin_post": linkedin_post}

# Aggregator node: Combine all posts
def aggregate_posts(state: OverallState) -> OverallState:
    """Combine all platform posts into a formatted final output"""
    print("📋 Aggregator: Combining all posts...\n")
    
    final_output = f"""
    {'='*70}
    SOCIAL MEDIA CONTENT PACKAGE
    {'='*70}
    Topic: {state['topic']}

    {'='*70}
    📸 INSTAGRAM POST
    {'='*70}

    {state['instagram_post']}

    {'='*70}
    🐦 TWITTER POST
    {'='*70}

    {state['twitter_post']}

    {'='*70}
    💼 LINKEDIN POST
    {'='*70}

    {state['linkedin_post']}

    {'='*70}
    CONTENT PACKAGE COMPLETE ✓
    {'='*70}
    """
    
    return {"final_output": final_output}


# Build the graph
builder = StateGraph(OverallState)

# Add all nodes
builder.add_node("generate_instagram", generate_instagram)
builder.add_node("generate_twitter", generate_twitter)
builder.add_node("generate_linkedin", generate_linkedin)
builder.add_node("aggregate_posts", aggregate_posts)

# Define parallel execution from START
# All three content generators run simultaneously
builder.add_edge(START, "generate_instagram")
builder.add_edge(START, "generate_twitter")
builder.add_edge(START, "generate_linkedin")

# All generators flow to the aggregator
builder.add_edge("generate_instagram", "aggregate_posts")
builder.add_edge("generate_twitter", "aggregate_posts")
builder.add_edge("generate_linkedin", "aggregate_posts")

# Aggregator flows to END
builder.add_edge("aggregate_posts", END)

# Compile the graph
graph = builder.compile()


# Test out the Graph
topic = "The impact of AI on workplace productivity"

print(f"\n🎯 Topic: {topic}\n")

result = graph.invoke({
    "topic": topic,
    "instagram_post": "",
    "twitter_post": "",
    "linkedin_post": "",
    "final_output": ""
})

print(result["final_output"])

