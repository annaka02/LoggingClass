from openai import OpenAI
from dotenv import load_dotenv
import google.generativeai as genai
import os
import json
import uuid
import logging
from datetime import datetime

load_dotenv() 

def setup_logging():
    """Configure logging to save logs in JSON format"""
    logger = logging.getLogger("Chatbot")

    file_handler = logging.FileHandler("chatbot_logs.json")
    formatter = logging.Formatter("%(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(console_handler)
    return logger

def initialize_client(use_ollama: bool = True, use_gemini: bool = False):
    """Initialize client for OpenAI, Ollama, or Gemini"""
    if use_gemini:
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        return genai.GenerativeModel('gemini-2.5-flash')
    if use_ollama:
        return OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama"                                                           
            )
    return OpenAI(api_key=os.getenv("OPEN_API_KEY"))

class Chatbot:
    def __init__(self, use_ollama: bool = True, use_gemini: bool = False):
        self.logger = setup_logging()
        self.session_id = str(uuid.uuid4())
        self.user_id = str(uuid.uuid4())
        self.client = initialize_client(use_ollama, use_gemini)
        self.use_ollama = use_ollama
        self.use_gemini = use_gemini
        self.model_name = "gemini-1.5-flash" if use_gemini else ("gpt-4o-mini" if not use_ollama else "llama3.2:8b")

        self.message = [
            {
                "role": "system",
                "content": "You are a helpful assistant that can answer questions and help with task."}
        ]

    def chat(self, user_input : str):
        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "level":"INFO",
                "type":"user_input",
                "metadata": {
                    "user_id": self.user_id,
                    "session_id": self.session_id,
                    "model": self.model_name
                }
            }
            self.logger.info(json.dumps(log_entry))

            self.message.append({
                "role": "user",
                "content": user_input
            })

            start_time = datetime.now()
            
            if self.use_gemini:
                chat = self.client.start_chat(history=[])
                response = chat.send_message(user_input)
                assistant_response = response.text
            else:
                response = self.client.chat.completions.create(
                    model=self.model_name, messages=self.message
                )
                assistant_response = response.choices[0].message.content
            
            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds()

            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "level": "INFO",
                "type": "model_response",
                "model_response": assistant_response,
                "metadata": {
                    "session_id": self.session_id,
                    "model": self.model_name,
                    "response_time": response_time,
                    "tokens_used": (response.usage.total_tokens
                                    if not self.use_gemini and hasattr(response, 'usage')
                                    else None
                                    ),
                }
            }
            self.logger.info(json.dumps(log_entry))

            self.message.append(
                {
                    "role": "assistant",
                    "content": assistant_response
                }
            )
            return assistant_response
        
        except Exception as e:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "level": "ERROR",
                "type": "error",
                "error_message": str(e),
                "metadata": {
                    "session_id": self.session_id,
                    "model": self.model_name,
                     "user_id": self.user_id,
                }
            }
            self.logger.error(json.dumps(log_entry))
            return f"Sorry, something evil happened in the universe: {str(e)}"


def main():
    print("\nSelect Model Type:")
    print("1. OpenAI GPT-4")
    print("2. Ollama Llama 3.2")
    print("3. Google Gemini")

    while True:
        choice = input("enter choice (1, 2, or 3): ").strip()
        if choice in ["1", "2", "3"]:
            break
        print("Please enter 1, 2, or 3")
        
    use_ollama = (choice == "2")
    use_gemini = (choice == "3")

    chatbot = Chatbot(use_ollama, use_gemini)
    print("\n===Chat Session Started===")
    model_type = "Gemini" if use_gemini else ("Ollama" if use_ollama else "OpenAI")
    print(f"Using {model_type} model")
    print("Type 'exit' to end the conversation\n")   
    print(f"Session ID: {chatbot.session_id}")
    
    while True:
        user_input = input("You: ").strip()

        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        if not user_input:
           continue

        response = chatbot.chat(user_input)
        print(f"Bot: {response}\n")
if __name__ == "__main__":
    try:
        main()   
    except KeyboardInterrupt:
        print("\nChat session ended by user.")




    





        
    


