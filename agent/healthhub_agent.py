"""
KVS HealthHub AI Agent
Agentic RAG system for healthcare information, prescriptions, and appointments
"""
import os
import sqlite3
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
import sys
import base64
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag.vector_store import initialize_vector_store


load_dotenv()

# Initialize LangSmith tracing
os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING", "true")
os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "cvs-healthhub-ai")

# Initialize LLM with OpenRouter (using ChatOpenAI with custom base URL)
llm = ChatOpenAI(
    model=os.getenv("OPENROUTER_MODEL", "anthropic/claude-3.5-sonnet"),
    temperature=0.7,
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),  # Your OpenRouter key here
    openai_api_base=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    default_headers={
        "HTTP-Referer": "https://cvs-healthhub-ai.com",
        "X-Title": "CVS HealthHub AI Assistant"
    }
)

# Initialize multimodal LLM for vision tasks (NVIDIA Nemotron)
vision_llm = ChatOpenAI(
    model=os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-nano-12b-v2-vl:free"),
    temperature=0.3,
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    default_headers={
        "HTTP-Referer": "https://cvs-healthhub-ai.com",
        "X-Title": "CVS HealthHub AI - Vision Analysis"
    }
)

# Initialize vector store
vector_store = initialize_vector_store()


@tool
def search_medication_info(query: str) -> str:
    """
    Search for medication information including uses, side effects, dosage, and interactions.
    Use this when user asks about specific medications, drugs, or prescriptions.

    Args:
        query: Medication name or question about medication

    Returns:
        Relevant medication information from knowledge base
    """
    try:
        results = vector_store.similarity_search(
            query=query,
            k=3,
            filter_dict={"category": "medication"}
        )

        if not results:
            return "No medication information found. Please consult a pharmacist or doctor."

        context = "Medication Information:\n\n"
        for idx, result in enumerate(results, 1):
            context += f"{idx}. {result['content']}\n\n"

        context += "\nNote: Always consult with a healthcare provider before starting any medication."
        return context

    except Exception as e:
        return f"Error searching medication info: {str(e)}"


@tool
def check_drug_interactions(medications: str) -> str:
    """
    Check for potential drug interactions between multiple medications.
    Use when user mentions taking multiple drugs or asks about medication safety.

    Args:
        medications: Comma-separated list of medication names

    Returns:
        Information about potential drug interactions
    """
    try:
        query = f"drug interactions between {medications}"
        results = vector_store.similarity_search(
            query=query,
            k=2,
            filter_dict={"category": "interactions"}
        )

        if not results:
            return f"No interaction data found for: {medications}. Please consult your pharmacist."

        context = f"Drug Interaction Check for: {medications}\n\n"
        for result in results:
            context += f"{result['content']}\n\n"

        context += "\n⚠️ Always verify with a licensed pharmacist before combining medications."
        return context

    except Exception as e:
        return f"Error checking interactions: {str(e)}"


@tool
def find_vaccines() -> str:
    """
    Find available vaccines, scheduling information, and vaccine guidelines.
    Use when user asks about flu shots, COVID vaccines, or immunizations.

    Returns:
        Available vaccine information
    """
    try:
        results = vector_store.similarity_search(
            query="vaccine availability schedule appointments",
            k=3,
            filter_dict={"category": "vaccines"}
        )

        if not results:
            return "Vaccine information currently unavailable. Please call your local CVS pharmacy."

        context = "Available Vaccines:\n\n"
        for result in results:
            context += f"• {result['content']}\n\n"

        context += "\nTo schedule: Call (555) 123-4567 or visit CVS.com/vaccines"
        return context

    except Exception as e:
        return f"Error fetching vaccine info: {str(e)}"


@tool
def get_store_services() -> str:
    """
    Get information about CVS pharmacy services including hours, locations, and offerings.
    Use when user asks about store hours, services, or pharmacy capabilities.

    Returns:
        CVS pharmacy services information
    """
    try:
        results = vector_store.similarity_search(
            query="CVS pharmacy services hours location",
            k=3,
            filter_dict={"category": "services"}
        )

        if not results:
            # Fallback to basic info
            return """CVS Pharmacy Services:

• Prescription Filling & Refills
• Immunizations (Flu, COVID-19, Shingles, etc.)
• Health Screenings
• Medication Therapy Management
• Specialty Pharmacy Services
• MinuteClinic (select locations)

Hours: Most locations open 8 AM - 10 PM
Find your nearest CVS: CVS.com/store-locator"""

        context = "CVS Services:\n\n"
        for result in results:
            context += f"{result['content']}\n\n"
        return context

    except Exception as e:
        return f"Error fetching services: {str(e)}"


@tool
def check_insurance_coverage(medication_or_service: str) -> str:
    """
    Check insurance coverage for medications or services.
    Use when user asks about copays, insurance, or coverage.

    Args:
        medication_or_service: Name of medication or service to check

    Returns:
        Insurance coverage information
    """
    try:
        query = f"insurance coverage copay {medication_or_service}"
        results = vector_store.similarity_search(
            query=query,
            k=2,
            filter_dict={"category": "insurance"}
        )

        if not results:
            return f"""Insurance coverage information for '{medication_or_service}':

Coverage varies by insurance plan. Please:
1. Contact your insurance provider
2. Call CVS Pharmacy at (555) 123-4567
3. Log in to CVS.com to check your specific coverage

Most Medicare and major insurance plans are accepted at CVS."""

        context = f"Insurance Coverage - {medication_or_service}:\n\n"
        for result in results:
            context += f"{result['content']}\n\n"
        return context

    except Exception as e:
        return f"Error checking coverage: {str(e)}"


def analyze_prescription_image(image_data: str, image_type: str = "base64") -> Dict[str, Any]:
    """
    Analyze prescription or medication images using NVIDIA's multimodal model.
    Extracts medication names, dosages, instructions, and provides safety information.

    This is NOT a tool for the agent, but a standalone function called from API/UI.

    Args:
        image_data: Base64 encoded image or file path
        image_type: Type of input ("base64" or "file")

    Returns:
        Dictionary with analysis results including medications found, warnings, and recommendations
    """
    try:
        # Prepare image for vision model
        if image_type == "file":
            with open(image_data, "rb") as img_file:
                image_b64 = base64.b64encode(img_file.read()).decode()
        else:
            image_b64 = image_data

        # Create vision prompt
        vision_prompt = """You are a pharmaceutical image analysis expert for CVS HealthHub AI.

Analyze this prescription or medication image and extract:

1. **Medication Names**: All medications visible (brand and generic names)
2. **Dosage Information**: Strength, quantity, frequency
3. **Instructions**: Usage directions if visible
4. **Patient Information**: Any visible patient details (for verification)
5. **Prescriber Information**: Doctor/pharmacy details if present
6. **Warnings**: Any special warnings or precautions visible

Then provide:
- **Safety Check**: Key drug interaction warnings
- **Next Steps**: What the patient should do (verify with pharmacist, etc.)
- **CVS Services**: Relevant CVS services (prescription transfer, auto-refill, etc.)

Format your response clearly with headers. If text is unclear, indicate that.
If this is NOT a prescription/medication image, politely state that."""

        # Call vision model
        message = HumanMessage(
            content=[
                {"type": "text", "text": vision_prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_b64}"
                    }
                }
            ]
        )

        response = vision_llm.invoke([message])
        analysis_text = response.content

        # Extract medication names for additional context
        medications_mentioned = []
        # Simple extraction - look for common medication patterns
        import re
        med_pattern = r'\b[A-Z][a-z]+(?:ol|il|am|in|ox|ine|ide|ate)\b'
        medications_mentioned = list(set(re.findall(med_pattern, analysis_text)))

        # Get additional info from knowledge base for mentioned medications
        additional_info = []
        if medications_mentioned:
            for med in medications_mentioned[:3]:  # Top 3 to avoid overload
                try:
                    results = vector_store.similarity_search(
                        query=f"{med} medication information",
                        k=1,
                        filter_dict={"category": "medication"}
                    )
                    if results:
                        additional_info.append(f"\n📚 **{med} Info from Database:**\n{results[0]['content']}\n")
                except:
                    pass

        # Combine analysis with knowledge base info
        full_response = analysis_text
        if additional_info:
            full_response += "\n\n---\n## Additional Information from CVS Knowledge Base\n"
            full_response += "\n".join(additional_info)

        full_response += "\n\n⚠️ **Disclaimer**: This is an automated analysis. Always verify prescription details with your CVS pharmacist before taking any medication."

        return {
            "success": True,
            "analysis": full_response,
            "medications_detected": medications_mentioned,
            "has_additional_info": len(additional_info) > 0
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Image analysis failed: {str(e)}",
            "analysis": "Unable to analyze image. Please ensure the image is clear and try again.",
            "medications_detected": []
        }


# Agent tools
tools = [
    search_medication_info,
    check_drug_interactions,
    find_vaccines,
    get_store_services,
    check_insurance_coverage
]

# System prompt for healthcare agent
system_prompt = """You are the CVS HealthHub AI Assistant, a helpful healthcare information agent.

Your capabilities:
- Provide medication information (uses, side effects, dosage)
- Check drug interactions for patient safety
- Help find and schedule vaccines
- Explain CVS pharmacy services and hours
- Assist with insurance coverage questions

Guidelines:
1. **Safety First**: Always recommend consulting healthcare providers for medical decisions
2. **Use Tools**: Call the appropriate tool based on user's question
3. **Be Clear**: Provide accurate, easy-to-understand information
4. **Be Helpful**: Guide users to next steps (call pharmacy, visit website, etc.)
5. **Compliance**: Remind users this is informational only, not medical advice

When user asks about:
- Medications/drugs → use search_medication_info
- Multiple medications → use check_drug_interactions
- Vaccines/immunizations → use find_vaccines
- Store hours/services → use get_store_services
- Insurance/copays → use check_insurance_coverage

Always prioritize patient safety and accurate information."""

# Create agent
agent = create_react_agent(llm, tools, prompt=system_prompt)


def chat(user_input: str, chat_history: List[Dict] = None) -> Dict[str, Any]:
    """
    Process user input through healthcare agent with conversation history

    Args:
        user_input: User's question or request
        chat_history: Previous conversation history (list of message dicts)

    Returns:
        Response with agent's answer and updated history
    """
    if chat_history is None:
        chat_history = []

    try:
        # Build messages list with history
        messages = []

        # Add previous messages (skip welcome message, keep only real conversation)
        for msg in chat_history:
            if msg["role"] in ["user", "assistant"] and "Welcome to CVS" not in msg["content"]:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

        # Add current user message
        messages.append({"role": "user", "content": user_input})

        # Invoke agent with full conversation context
        result = agent.invoke({
            "messages": messages
        })

        # Extract response
        response = result["messages"][-1].content if result.get("messages") else "I apologize, I couldn't process that request."

        return {
            "response": response,
            "chat_history": chat_history,
            "tool_calls": len([m for m in result.get("messages", []) if hasattr(m, 'tool_calls')])
        }

    except Exception as e:
        return {
            "response": f"I encountered an error: {str(e)}. Please try rephrasing your question.",
            "chat_history": chat_history,
            "tool_calls": 0
        }


if __name__ == "__main__":
    # Test agent
    print("CVS HealthHub AI Agent initialized!")
    print(f"Vector Store: {vector_store.get_collection_stats()}")

    # Test query
    test_response = chat("What vaccines are available?")
    print(f"\nTest Response: {test_response['response']}")
