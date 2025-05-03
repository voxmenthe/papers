import os
from google import genai
from rich.console import Console

console = Console()

# Configure and initialize the client at the module level
api_key = os.getenv("GEMINI_API_KEY")
genai_client = None
if not api_key:
    console.print("[bold red]Error: GEMINI_API_KEY environment variable not set.[/]")
else:
    try:
        # The new SDK uses the Client object directly
        genai_client = genai.Client(api_key=api_key)
        # Optional: Test connection or list models here if needed
        # models = genai_client.models.list()
        console.log("Google GenAI Client configured successfully.")
    except Exception as e:
        console.print(f"[bold red]Error configuring Google GenAI Client: {e}[/]")
        genai_client = None

def call_llm(prompt: str, model_name: str = 'gemini-1.5-flash-latest') -> str:
    """Calls the configured Gemini model with the given prompt using the new SDK pattern."""
    if not genai_client:
        return "Error: Google GenAI Client not initialized. Check API key (GEMINI_API_KEY) and configuration."

    try:
        console.log(f"Sending prompt to Gemini (model: {model_name}, length: {len(prompt)} chars)...")
        # Use the client.models.generate_content method
        response = genai_client.models.generate_content(
            model=model_name,  # Specify the model here
            contents=prompt    # Pass the prompt string directly to contents
        )

        # Access the text directly from the response object
        if hasattr(response, 'text'):
            result_text = response.text
            console.log("Received response from Gemini.")
            return result_text
        else:
            # Log the full response object for debugging if text attribute is missing
            console.print("[bold yellow]Warning: Response object missing 'text' attribute:[/]")
            try:
                # Try dumping the model to see structure
                console.print(response.model_dump_json(exclude_none=True, indent=2))
            except AttributeError:
                 console.print(response) # Fallback to printing the object itself
            return "Error: Received unexpected response structure from Gemini (missing text)."

    except Exception as e:
        console.print(f"[bold red]Error during Gemini API call: {e}[/]")
        # import traceback
        # console.print(traceback.format_exc())
        return f"Error calling Gemini API: {e}"
