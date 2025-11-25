from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List, Literal
from pydantic import BaseModel, Field

# Define the state
class OptimizationState(TypedDict):
    """State for evaluator-optimizer workflow"""
    product_name: str # Name of the product
    product_features: List[str] # List of its features
    target_audience: str # Audience we want to sell to
    current_description: str # The current generated product details
    evaluation_result: dict # A breakdown of evaluation results
    feedback: str # Main feedback from the Evaluator
    iteration_count: int # Number of times the E-O loop has ran
    max_iterations: int # Total number of times the E-O loop should run
    is_approved: bool # If the Evaluator has approved the product details
    iteration_history: List[dict] # All the evaluations made in order


# Structured outputs

class ProductDescription(BaseModel):
    """Generated product description"""
    headline: str = Field(description="Catchy headline (max 10 words)")
    description: str = Field(description="Main product description (100-150 words)")
    key_benefits: List[str] = Field(description="3-5 key benefits as bullet points")
    call_to_action: str = Field(description="Compelling call-to-action")


class Evaluation(BaseModel):
    """Evaluation of product description"""
    overall_score: int = Field(description="Overall quality score 1-10", ge=1, le=10)
    clarity_score: int = Field(description="Clarity score 1-10", ge=1, le=10)
    persuasiveness_score: int = Field(description="Persuasiveness score 1-10", ge=1, le=10)
    audience_fit_score: int = Field(description="Target audience fit 1-10", ge=1, le=10)
    is_approved: bool = Field(description="Whether description meets standards (score >= 8)")
    strengths: List[str] = Field(description="What works well")
    weaknesses: List[str] = Field(description="What needs improvement")
    specific_feedback: str = Field(description="Detailed, actionable feedback for revision")


llm = ChatOpenAI(model="gpt-4o-mini") # gpt-4 does not support structured outputs

gemini_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Optimizer Node: Generate or refine description
def generate_description(state: OptimizationState) -> OptimizationState:
    """Optimizer creates or improves product description"""
    iteration = state["iteration_count"]
    
    print(f"\n{'='*70}")
    print(f"✍️  OPTIMIZER: Iteration {iteration}")
    print(f"{'='*70}")
    
    optimizer_llm = llm.with_structured_output(ProductDescription)
    
    # First iteration - generate from scratch
    if iteration == 1:
        print("Creating initial product description...\n")
        
        prompt = f"""Create a compelling product description for:

        Product: {state['product_name']}
        Features: {', '.join(state['product_features'])}
        Target Audience: {state['target_audience']}

        Requirements:
        - Headline: Catchy and concise (max 10 words)
        - Description: Engaging and informative (100-150 words)
        - Key Benefits: 3-5 clear, compelling benefits
        - Call-to-Action: Strong, action-oriented CTA

        Make it persuasive and tailored to the target audience."""
    
    # Subsequent iterations - refine based on feedback
    else:
        print(f"Refining description based on evaluation feedback...\n")
        
        prompt = f"""Improve this product description based on feedback:

        Product: {state['product_name']}
        Target Audience: {state['target_audience']}

        CURRENT DESCRIPTION:
        {state['current_description']}

        EVALUATION SCORES:
        - Overall: {state['evaluation_result'].get('overall_score', 0)}/10
        - Clarity: {state['evaluation_result'].get('clarity_score', 0)}/10
        - Persuasiveness: {state['evaluation_result'].get('persuasiveness_score', 0)}/10
        - Audience Fit: {state['evaluation_result'].get('audience_fit_score', 0)}/10

        FEEDBACK TO ADDRESS:
        {state['feedback']}

        CRITICAL: Focus on the specific weaknesses mentioned. Make targeted improvements to:
        1. Address each point in the feedback
        2. Maintain the strengths that were working
        3. Increase scores in weak areas

        Generate an improved version that addresses all feedback."""
    

    description = optimizer_llm.invoke(prompt)
    
    # Format for display and storage
    formatted_description = f"""
        HEADLINE: {description.headline}
        
        DESCRIPTION:
        {description.description}
        
        KEY BENEFITS:
        
        {chr(10).join([f"• {benefit}" for benefit in description.key_benefits])}
        
        CALL-TO-ACTION:
        {description.call_to_action}"""
    
    print("Generated Description:")
    print("-" * 70)
    print(formatted_description)
    print()
    
    return {
        "current_description": formatted_description,
        "iteration_count": iteration + 1
    }


# Evaluator Node: Assess the description
def evaluate_description(state: OptimizationState) -> OptimizationState:
    """Evaluator assesses product description quality"""
    print(f"{'='*70}")
    print(f"🔍 EVALUATOR: Reviewing description")
    print(f"{'='*70}\n")
    
    evaluator_llm = gemini_llm.with_structured_output(Evaluation)
    
    prompt = f"""Evaluate this product description objectively:

        Product: {state['product_name']}
        Target Audience: {state['target_audience']}

        DESCRIPTION TO EVALUATE:
        {state['current_description']}

        Evaluate on these criteria (1-10 scale):
        1. CLARITY: Is it clear and easy to understand?
        2. PERSUASIVENESS: Does it effectively sell the product?
        3. AUDIENCE FIT: Does it resonate with the target audience?

        APPROVAL CRITERIA: Overall score must be 8 or higher to approve.

        Provide:
        - Scores for each criterion
        - Overall score (average of criteria)
        - Whether it's approved (score >= 8)
        - Specific strengths (what's working well)
        - Specific weaknesses (what needs improvement)
        - Actionable feedback for the next iteration

        Be objective and constructive."""
    
    evaluation = evaluator_llm.invoke(prompt)
    
    print(f"Evaluation Results:")
    print("-" * 70)
    print(f"Overall Score: {evaluation.overall_score}/10")
    print(f"Clarity: {evaluation.clarity_score}/10")
    print(f"Persuasiveness: {evaluation.persuasiveness_score}/10")
    print(f"Audience Fit: {evaluation.audience_fit_score}/10")
    print(f"Status: {'✅ APPROVED' if evaluation.is_approved else '❌ NEEDS REVISION'}")
    print(f"\nStrengths:")
    for strength in evaluation.strengths:
        print(f"  ✓ {strength}")
    print(f"\nWeaknesses:")
    for weakness in evaluation.weaknesses:
        print(f"  ✗ {weakness}")
    print(f"\nFeedback: {evaluation.specific_feedback}")
    print()
    
    # Store iteration in history
    iteration_record = {
        "iteration": state["iteration_count"] - 1,
        "description": state["current_description"],
        "scores": {
            "overall": evaluation.overall_score,
            "clarity": evaluation.clarity_score,
            "persuasiveness": evaluation.persuasiveness_score,
            "audience_fit": evaluation.audience_fit_score
        },
        "approved": evaluation.is_approved,
        "feedback": evaluation.specific_feedback
    }
    
    history = state.get("iteration_history", [])
    history.append(iteration_record)
    
    return {
        "evaluation_result": evaluation.model_dump(),
        "feedback": evaluation.specific_feedback,
        "is_approved": evaluation.is_approved,
        "iteration_history": history
    }


# Decision: Continue or finish? (Conditional Edge function)
def should_continue(state: OptimizationState) -> Literal["optimizer", "end"]:
    """Decide whether to continue optimizing or finish"""
    
    # If approved, we're done
    if state["is_approved"]:
        print(f"{'='*70}")
        print(f"✅ SUCCESS: Description approved!")
        print(f"{'='*70}\n")

        return "end"
    
    # If we've hit max iterations, stop
    if state["iteration_count"] > state["max_iterations"]:
        print(f"{'='*70}")
        print(f"⚠️  MAX ITERATIONS REACHED: Stopping at iteration {state['iteration_count'] - 1}")
        print(f"{'='*70}\n")

        return "end"
    
    # Otherwise, continue optimizing
    print(f"{'='*70}")
    print(f"🔄 CONTINUING: Routing back to optimizer for iteration {state['iteration_count']}")
    print(f"{'='*70}\n")

    return "optimizer"


# Build the graph
builder = StateGraph(OptimizationState)

# Add nodes
builder.add_node("optimizer", generate_description)
builder.add_node("evaluator", evaluate_description)

# Define the flow
builder.add_edge(START, "optimizer")
builder.add_edge("optimizer", "evaluator")

# Conditional edge: continue or end
builder.add_conditional_edges(
    "evaluator",
    should_continue,
    {
        "optimizer": "optimizer",  # Loop back for another iteration
        "end": END
    }
)


# Compile the graph
graph = builder.compile()

# Example 1: Fitness Tracker
print("="*70)
print("EVALUATOR-OPTIMIZER PATTERN: PRODUCT DESCRIPTION")
print("="*70)
print("Example: Fitness Tracker")
print("="*70)

result = graph.invoke({
    "product_name": "FitPulse Pro Smart Watch",
    "product_features": [
        "Heart rate monitoring",
        "GPS tracking",
        "Sleep analysis",
        "Waterproof to 50m",
        "7-day battery life",
        "Smartphone notifications"
    ],
    "target_audience": "Health-conscious professionals aged 25-45",
    "current_description": "",
    "evaluation_result": {},
    "feedback": "",
    "iteration_count": 1,
    "max_iterations": 5,
    "is_approved": False,
    "iteration_history": []
})

print("\n" + "="*70)
print("FINAL APPROVED DESCRIPTION")
print("="*70)
print(result["current_description"])


print("\n" + "="*70)
print("OPTIMIZATION JOURNEY")
print("="*70)
for record in result["iteration_history"]:
    print(f"\nIteration {record['iteration']}:")
    print(f"  Scores: Overall={record['scores']['overall']}/10, "
          f"Clarity={record['scores']['clarity']}/10, "
          f"Persuasiveness={record['scores']['persuasiveness']}/10, "
          f"Audience Fit={record['scores']['audience_fit']}/10")
    print(f"  Status: {'✅ Approved' if record['approved'] else '❌ Needs work'}")
    if not record['approved']:
        print(f"  Feedback: {record['feedback'][:100]}...")