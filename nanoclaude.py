"""NanoClaude — a code-act agent with a Claude Code-style terminal interface.

Decorators handle spinner and tool logging; they no-op gracefully when
``headless=True`` (no printer attached).  No inheritance — one class, two modes.
"""

import json
import os
import traceback
from pydantic import BaseModel
from litellm import completion

import re
import time
import libtmux


class Config(BaseModel):
    model: str = "deepseek/deepseek-v4-pro"

class Content(BaseModel):
    type: str
    text: str

class Message(BaseModel):
    role: str
    content: list[Content]

# Convert observation to json
def convert_obs_to_json(result, tool):
    return {
        "role": "tool",
        "content": [{"type": "text", "text": result}],
        "tool_call_id": tool.id,
        "name": tool.function.name,
    }

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

from typing import Literal, Tuple

HARD_TIMEOUT = 300

PS1_BLOCK_BEGIN = '\n###PS1JSON###\n'
PS1_BLOCK_END   = '\n###PS1END###'

PS1_BLOCK_REGEX = re.compile(
    f'^{PS1_BLOCK_BEGIN.strip()}(.*?){PS1_BLOCK_END.strip()}',
    re.DOTALL | re.MULTILINE,
)


class CmdOutputMetadata(BaseModel):
    exit_code: str = "-1"
    pid: int = -1
    username: str | None = None
    hostname: str | None = None
    working_dir: str | None = None
    py_interpreter_path: str | None = None

    @classmethod
    def build_ps1_prompt(cls) -> str:
        json_template = json.dumps(
            {
                'pid': '$!',
                'exit_code': '$?',
                'username': r'\u',
                'hostname': r'\h',
                'working_dir': r'$(pwd)',
                'py_interpreter_path': r'$(which python 2>/dev/null || echo "")',
            },
            indent=2,
        ).replace('"', r'\"')

        return PS1_BLOCK_BEGIN + json_template + PS1_BLOCK_END + '\n'

    @classmethod
    def parse_ps1_blocks(cls, pane_text: str) -> tuple[list[re.Match], list[dict]]:
        matches = []
        metadata = []

#         print('PANE\n',pane_text)
        for match in PS1_BLOCK_REGEX.finditer(pane_text):
            
            try:
                parsed = json.loads(match.group(1).strip())
                matches.append(match)
                metadata.append(parsed)
            except json.JSONDecodeError:
                print(
                    f'Could not parse PS1 block, skipping:\n{match.group(1)}\n'
                    + traceback.format_exc()
                )

        return matches, metadata


class CmdOutputObservation(BaseModel):
    content: str
    metadata: CmdOutputMetadata

    def to_agent_observation(self) -> str:
        parts = [f'<RESULT>{self.content}</RESULT>']

        if self.metadata.working_dir:
            parts.append(f'[Current working directory: {self.metadata.working_dir}]')
        if self.metadata.py_interpreter_path:
            parts.append(f'[Python interpreter: {self.metadata.py_interpreter_path}]')
        if self.metadata.exit_code:
            parts.append(f'[Exit code: {self.metadata.exit_code}]')

        return '\n'.join(parts)


class BashSession:
    PS1 = CmdOutputMetadata.build_ps1_prompt()
    HISTORY_LIMIT = 10_000

    def __init__(self, work_dir: str, username: str):
        self.work_dir = work_dir
        self.username = username
        self.server = libtmux.Server()

    def initialize(self):
        self.session = self.server.new_session(
            session_name='bash_session',
            start_directory=self.work_dir,
            kill_session=True,
            x=1000, y=1000,
        )
        self.session.set_option('history-limit', str(self.HISTORY_LIMIT))

        initial_window = self.session.active_window
        self.window = self.session.new_window(
            window_name='bash',
            window_shell='/bin/bash',
            start_directory=self.work_dir,
        )
        self.pane = self.window.active_pane
        initial_window.kill()

        self.pane.send_keys(
            f'export PROMPT_COMMAND=\'export PS1="{self.PS1}"\'; export PS2=""'
        )
        time.sleep(0.5)
        self._clear_pane()

    def _clear_pane(self) -> None:
        self.pane.send_keys('C-l', enter=False)
        time.sleep(0.1)
        self.pane.cmd('clear-history')

    def _get_pane_content(self) -> str:
        lines = self.pane.cmd('capture-pane', '-J', '-pS', '-').stdout
        return '\n'.join(line.rstrip() for line in lines)

    def _slice_output_between_markers(
        self,
        pane_content: str,
        ps1_matches: list[re.Match],
        before_last_match: bool = False,
    ) -> Tuple[str, list[str]]:
        if len(ps1_matches) == 0:
            return pane_content, [pane_content]

        if len(ps1_matches) == 1:
            if before_last_match:
                text = pane_content[: ps1_matches[0].start()]
            else:
                text = pane_content[ps1_matches[0].end() + 1 :]
            return text, [text]

        segments = []
        for i in range(len(ps1_matches) - 1):
            segment = pane_content[
                ps1_matches[i].end() + 1 : ps1_matches[i + 1].start()
            ]
            segments.append(segment)

        combined = '\n'.join(segments)
        return combined, segments

    def execute(self, command: str) -> CmdOutputObservation:
        init_content = self._get_pane_content()
        init_matches, _ = CmdOutputMetadata.parse_ps1_blocks(init_content)
        init_marker_count = len(init_matches)

        self.pane.send_keys(command)
        time.sleep(0.5)

        start_time = time.time()
        while time.time() - start_time < HARD_TIMEOUT:
            pane_text = self._get_pane_content()
            ps1_matches, ps1_jsons = CmdOutputMetadata.parse_ps1_blocks(pane_text)

            if len(ps1_matches) == init_marker_count + 1:
                _, segments = self._slice_output_between_markers(pane_text, ps1_matches)
                output = ''.join(segments)

                return CmdOutputObservation(
                    content=output,
                    metadata=CmdOutputMetadata(**ps1_jsons[-1]),
                )

            time.sleep(0.5)

        return CmdOutputObservation(
            content='Command timed out — it may still be running.',
            metadata=CmdOutputMetadata(),
        )

class NanoClaude:

    def __init__(self):
        # ── agent state ─────────────────────────────────────────────────
        self.history = [{"role": "system", "content": "You're NanoClaude Code, a god tier coding agent"}]
        
        # ADD TOOLs
        self.tools = [bash_tool()]

        #initialize the bash tool
        self.bash_session = BashSession(
            work_dir=os.getcwd(),
            username="to",
        )
        self.bash_session.initialize()


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
            # print(response)
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
            print('='*50)
            print('EXECUTING BASH TOOL')
            print('='*50)
            arguments = json.loads(tool.function.arguments)
            out = self.bash_session.execute(arguments.get("command", ""))
            result = out.to_agent_observation()
            return convert_obs_to_json(result=result, tool=tool)
            


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