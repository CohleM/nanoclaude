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
get_weather_tool = {
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

_DETAILED_BASH_DESCRIPTION = """Execute a bash command in the terminal within a persistent shell session.

### Command Execution
* One command at a time: You can only execute one bash command at a time. If you need to run multiple commands sequentially, use `&&` or `;` to chain them together.
* Persistent session: Commands execute in a persistent shell session where environment variables, virtual environments, and working directory persist between commands.
* Timeout: Commands have a soft timeout of 10 seconds, once that's reached, you have the option to continue or interrupt the command (see section below for details)

### Running and Interacting with Processes
* Long running commands: For commands that may run indefinitely, run them in the background and redirect output to a file, e.g. `python3 app.py > server.log 2>&1 &`. For commands that need to run for a specific duration, like "sleep", you can set the "timeout" argument to specify a hard timeout in seconds.
* Interact with running process: If a bash command returns exit code `-1`, this means the process is not yet finished. By setting `is_input` to `true`, you can:
  - Send empty `command` to retrieve additional logs
  - Send text (set `command` to the text) to STDIN of the running process
  - Send control commands like `C-c` (Ctrl+C), `C-d` (Ctrl+D), or `C-z` (Ctrl+Z) to interrupt the process

### Best Practices
* Directory verification: Before creating new directories or files, first verify the parent directory exists and is the correct location.
* Directory management: Try to maintain working directory by using absolute paths and avoiding excessive use of `cd`.

### Output Handling
* Output truncation: If the output exceeds a maximum length, it will be truncated before being returned.
"""


def bash_tool() -> dict:
    
    return {
        'type': 'function',
        'function': {
            'name': 'execute_bash',
            'description': _DETAILED_BASH_DESCRIPTION,
            'parameters': {
                'type': 'object',
                'properties': {
                    'command': {
                        'type': 'string',
                        'description': 
                            'The bash command to execute. Can be empty string to view additional logs when previous exit code is `-1`. Can be `C-c` (Ctrl+C) to interrupt the currently running process. Note: You can only execute one bash command at a time. If you need to run multiple commands sequentially, you can use `&&` or `;` to chain them together.'
                        ,
                    },
                    
                },
                'required': ['command'],
            },
        },
    }


def get_weather_fn(location):
    return str({"weather": "Sunny", "temp": "28C", "location": location})

class NanoClaude:

    def __init__(self):
        # ── agent state ─────────────────────────────────────────────────
        self.history = [{"role": "system", "content": "You're NanoClaude Code, a god tier coding agent"}]
        
        # ADD TOOLs
        self.tools = [bash_tool()]
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
                print('ASSISTANT\n', assistant_msg.content )
                print('='*50)

            # ADD TOOLS
            if hasattr(assistant_msg, "tool_calls") and assistant_msg.tool_calls:
                self.history.append(assistant_msg.model_dump())
                for tool in assistant_msg.tool_calls:


                    # if tool.function.name == 'get_weather':
                    #     location = json.loads(tool.function.arguments)['location']
                    #     output = get_weather_fn(location)
                        
                    #     self.history.append({
                    #     "role": "tool",
                    #     "content": output,
                    #     "tool_call_id": tool.id
                    #     })

                    observation = self.perform_action(tool)
                    self.history.append(observation)


                        

            # natural stop
            if response.choices[0].finish_reason == "stop":
                print('='*50)
                print("Execution Finished")
                print('='*50)
                break
    
    def perform_action(self, tool):
        tool_name = tool.function.name

        if tool_name == "execute_bash":
            arguments = json.loads(tool.function.arguments)
            command = arguments.get("command", "")

            # now we need a bash session to execute this command
            # and return that command's output back to our LLM
            pass
            


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