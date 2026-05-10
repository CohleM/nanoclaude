"""NanoClaude — a code-act agent with a Claude Code-style terminal interface.

Decorators handle spinner and tool logging; they no-op gracefully when
``headless=True`` (no printer attached).  No inheritance — one class, two modes.
"""

import json
import os
import traceback
from pydantic import BaseModel
from litellm import completion

class Config(BaseModel):
    model: str = "deepseek/deepseek-v4-pro"

class Content(BaseModel):
    type: str
    text: str

class Message(BaseModel):
    role: str
    content: list[Content]

# TOOLS
get_weather = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"]
        }
    }
}
class NanoClaude:

    def __init__(self):
        # ── agent state ─────────────────────────────────────────────────
        self.history = [{"role": "system", "content": "You're NanoClaude Code, a god tier coding agent"}]
        
        # ADD TOOLs
        self.tools = [get_weather]
    def step(self):
        params = {
            **Config().model_dump(),
            "thinking": {"type": "enabled"},
            "api_key": os.environ['LLM_API_KEY']
        }
        params["messages"] = self.history
        params["tools"] = self.tools # Add tools
        return completion(**params)

    def execute(self, user_message):
        
        self.history.append(user_message.dict())

        # print(self.history)

        while True:
            response = self.step()
            print(response)
            assistant_msg = response.choices[0].message

            if assistant_msg.content:
                print('='*50)
                print('ASSISTANT\n', assistant_msg.content, )
                print('='*50)

            # ADD TOOLS
            if hasattr(assistant_msg, "tool_calls") and assistant_msg.tool_calls:
                self.history.append(assistant_msg.model_dump())
                for tool in assistant_msg.tool_calls:
                    if tool.function.name == 'get_weather':
                        self.history.append({
                        "role": "tool",
                        "content": '{"weather": "Sunny", "temp": "28C"}',
                        "tool_call_id": tool.id
                    })
                        

            # natural stop
            if response.choices[0].finish_reason == "stop":
                print('='*50)
                print("Execution Finished")
                print('='*50)
                break

    # ── interactive repl ────────────────────────────────────────────────
    def repl(self):
        try:
            while True:
                print('='*50) 
                print('Enter your prompt')
                print('='*50) 
                instruction = input().strip()

                if instruction.lower() in ("quit", "exit", "q"):
                    break

                if not instruction:
                    continue

                content = Content(type="text", text=instruction)
                user_message = Message(content=[content], role="user")

                self.execute(user_message=user_message)

        except KeyboardInterrupt:
            print("\n----- Agent stopped by user! -----")
        except Exception as e:
            print(f"Unexpected error: {e}")
            traceback.print_exc()
        finally:
            print("NanoClaude session ended")


nanoclaude = NanoClaude()
nanoclaude.repl()