"""NanoClaude — a code-act agent with a Claude Code-style terminal interface.

Decorators handle spinner and tool logging; they no-op gracefully when
``headless=True`` (no printer attached).  No inheritance — one class, two modes.
"""

import json
import os
import traceback
import textwrap
from pydantic import BaseModel
from litellm import completion

import re
import time
import libtmux


import shutil
import sys
import tempfile
import time

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Tuple

# ═══════════════════════════════════════════════════════════════════════════
# Terminal colors & styling (Arch Linux aesthetic)
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = {
    'content': """You are OpenHands agent, a helpful AI assistant that can interact with a computer to solve tasks.

<ROLE>
Your primary role is to assist users by executing commands, modifying code, and solving technical problems effectively. You should be thorough, methodical, and prioritize quality over speed.
* If the user asks a question, like "why is X happening", don't try to fix the problem. Just give an answer to the question.
</ROLE>

<EFFICIENCY>
* Each action you take is somewhat expensive. Wherever possible, combine multiple actions into a single action, e.g. combine multiple bash commands into one, using sed and grep to edit/view multiple files at once.
* When exploring the codebase, use efficient tools like find, grep, and git commands with appropriate filters to minimize unnecessary operations.
</EFFICIENCY>

<FILE_SYSTEM_GUIDELINES>
* When a user provides a file path, do NOT assume it's relative to the current working directory. First explore the file system to locate the file before working on it.
* If asked to edit a file, edit the file directly, rather than creating a new file with a different filename.
* For global search-and-replace operations, consider using `sed` instead of opening file editors multiple times.
</FILE_SYSTEM_GUIDELINES>

<CODE_QUALITY>
* Write clean, efficient code with minimal comments. Avoid redundancy in comments: Do not repeat information that can be easily inferred from the code itself.
* When implementing solutions, focus on making the minimal changes needed to solve the problem.
* Before implementing any changes, first thoroughly understand the codebase through exploration.
* If you are adding a lot of code to a function or file, consider splitting the function or file into smaller pieces when appropriate.
</CODE_QUALITY>

<VERSION_CONTROL>
* When configuring git credentials, use "openhands" as the user.name and "openhands@all-hands.dev" as the user.email by default, unless explicitly instructed otherwise.
* Exercise caution with git operations. Do NOT make potentially dangerous changes (e.g., pushing to main, deleting repositories) unless explicitly asked to do so.
* When committing changes, use `git status` to see all modified files, and stage all files necessary for the commit. Use `git commit -a` whenever possible.
* Do NOT commit files that typically shouldn't go into version control (e.g., node_modules/, .env files, build directories, cache files, large binaries) unless explicitly instructed by the user.
* If unsure about committing certain files, check for the presence of .gitignore files or ask the user for clarification.
</VERSION_CONTROL>

<PULL_REQUESTS>
* When creating pull requests, create only ONE per session/issue unless explicitly instructed otherwise.
* When working with an existing PR, update it with new commits rather than creating additional PRs for the same issue.
* When updating a PR, preserve the original PR title and purpose, updating description only when necessary.
</PULL_REQUESTS>

<PROBLEM_SOLVING_WORKFLOW>
1. EXPLORATION: Thoroughly explore relevant files and understand the context before proposing solutions
2. ANALYSIS: Consider multiple approaches and select the most promising one
3. TESTING:
   * For bug fixes: Create tests to verify issues before implementing fixes
   * For new features: Consider test-driven development when appropriate
   * If the repository lacks testing infrastructure and implementing tests would require extensive setup, consult with the user before investing time in building testing infrastructure
   * If the environment is not set up to run tests, consult with the user first before investing time to install all dependencies
4. IMPLEMENTATION: Make focused, minimal changes to address the problem
5. VERIFICATION: If the environment is set up to run tests, test your implementation thoroughly, including edge cases. If the environment is not set up to run tests, consult with the user first before investing time to run tests.
</PROBLEM_SOLVING_WORKFLOW>

<SECURITY>
* Only use GITHUB_TOKEN and other credentials in ways the user has explicitly requested and would expect.
* Use APIs to work with GitHub or other platforms, unless the user asks otherwise or your task requires browsing.
</SECURITY>

<ENVIRONMENT_SETUP>
* When user asks you to run an application, don't stop if the application is not installed. Instead, please install the application and run the command again.
* If you encounter missing dependencies:
  1. First, look around in the repository for existing dependency files (requirements.txt, pyproject.toml, package.json, Gemfile, etc.)
  2. If dependency files exist, use them to install all dependencies at once (e.g., `pip install -r requirements.txt`, `npm install`, etc.)
  3. Only install individual packages directly if no dependency files are found or if only specific packages are needed
* Similarly, if you encounter missing dependencies for essential tools requested by the user, install them when possible.
</ENVIRONMENT_SETUP>

<TROUBLESHOOTING>
* If you've made repeated attempts to solve a problem but tests still fail or the user reports it's still broken:
  1. Step back and reflect on 5-7 different possible sources of the problem
  2. Assess the likelihood of each possible cause
  3. Methodically address the most likely causes, starting with the highest probability
  4. Document your reasoning process
* When you run into any major issue while executing a plan from the user, please don't try to directly work around it. Instead, propose a new plan and confirm with the user before proceeding.
</TROUBLESHOOTING>""",
    'role': 'system',
}

class Colors:
    """ANSI terminal colors — Arch Linux palette."""
    RESET     = '\033[0m'
    BOLD      = '\033[1m'
    DIM       = '\033[2m'
    ITALIC    = '\033[3m'
    UNDERLINE = '\033[4m'

    # Arch-inspired palette
    CYAN      = '\033[36m'
    BLUE      = '\033[34m'
    BRIGHT_BLUE = '\033[94m'
    WHITE     = '\033[37m'
    BRIGHT_WHITE = '\033[97m'
    GREEN     = '\033[32m'
    YELLOW    = '\033[33m'
    MAGENTA   = '\033[35m'
    RED       = '\033[31m'


def print_separator(char: str = '─', width: int = 70) -> None:
    """Print a styled horizontal separator line."""
    print(f"{Colors.DIM}{char * width}{Colors.RESET}")


def print_header(title: str) -> None:
    """Print a styled section header."""
    print()
    print(f"{Colors.BOLD}{Colors.CYAN}  ▸ {title}{Colors.RESET}")
    print_separator('─')


def print_tool_call(tool_name: str, arguments: dict) -> None:
    """Pretty-print a tool invocation with its parameters."""
    print()
    print(f"{Colors.BOLD}{Colors.BRIGHT_BLUE}  ⚙  TOOL CALL{Colors.RESET}")
    print(f"{Colors.DIM}  ├─ name:{Colors.RESET} {Colors.BOLD}{Colors.YELLOW}{tool_name}{Colors.RESET}")
    for key, value in arguments.items():
        val_str = str(value)
        if len(val_str) > 80:
            val_str = val_str[:77] + '...'
        # Replace newlines in value for clean display
        val_str = val_str.replace('\n', '\\n')
        print(f"{Colors.DIM}  ├─ {key}:{Colors.RESET} {Colors.GREEN}{val_str}{Colors.RESET}")
    print(f"{Colors.DIM}  └─ executing...{Colors.RESET}")


def print_assistant_message(content: str) -> None:
    """Pretty-print the assistant's response."""
    print()
    print(f"{Colors.BOLD}{Colors.MAGENTA}  🤖 ASSISTANT{Colors.RESET}")
    print_separator('─')
    # Indent the content
    for line in content.split('\n'):
        print(f"  {Colors.WHITE}{line}{Colors.RESET}")
    print_separator('─')


def print_tool_result(success: bool, summary: str = '', observation: str = '') -> None:
    """Pretty-print tool execution result, including the observation output."""
    icon = f"{Colors.GREEN}✓{Colors.RESET}" if success else f"{Colors.RED}✗{Colors.RESET}"
    print(f"  {icon} {Colors.DIM}Tool finished{Colors.RESET} {summary}")

    # ── Print the observation (truncated first section) ─────────────────
    if observation:
        # Grab the first ~500 chars / 8 lines for a preview
        lines = observation.split('\n')
        preview_lines = lines[:8]
        preview = '\n'.join(preview_lines)
        if len(preview) > 500:
            preview = preview[:500] + '...'
        elif len(lines) > 8:
            preview += '\n  ...'

        print(f"{Colors.DIM}  ┌ observation ────────────────────{Colors.RESET}")
        for line in preview.split('\n'):
            print(f"  {Colors.DIM}│{Colors.RESET} {Colors.WHITE}{line}{Colors.RESET}")
        print(f"{Colors.DIM}  └──────────────────────────────────{Colors.RESET}")


def print_finish() -> None:
    """Print the session-finish banner."""
    print()
    print_separator('═', 70)
    print(f"{Colors.BOLD}{Colors.GREEN}  ✓ Execution Finished{Colors.RESET}")
    print_separator('═', 70)
    print()


def print_prompt() -> None:
    """Print the REPL prompt line."""
    print()
    print(f"{Colors.BOLD}{Colors.BRIGHT_WHITE}  ❯ Enter your prompt{Colors.RESET} "
          f"{Colors.DIM}(or 'quit' to exit){Colors.RESET}")
    print(f"{Colors.DIM}  ─────────────────────────────────────{Colors.RESET}")


# ═══════════════════════════════════════════════════════════════════════════
# Splash Screen — Arch Linux style
# ═══════════════════════════════════════════════════════════════════════════

NANO_CLAUDE_LOGO = """
{cyan}  ███╗   ██╗ █████╗ ███╗   ██╗ ██████╗ {reset}
{cyan}  ████╗  ██║██╔══██╗████╗  ██║██╔═══██╗{reset}
{cyan}  ██╔██╗ ██║███████║██╔██╗ ██║██║   ██║{reset}
{cyan}  ██║╚██╗██║██╔══██║██║╚██╗██║██║   ██║{reset}
{cyan}  ██║ ╚████║██║  ██║██║ ╚████║╚██████╔╝{reset}
{cyan}  ╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝{reset}
{blue}  ██████╗██╗      █████╗ ██╗   ██╗██████╗ ███████╗{reset}
{blue}  ██╔════╝██║     ██╔══██╗██║   ██║██╔══██╗██╔════╝{reset}
{blue}  ██║     ██║     ███████║██║   ██║██║  ██║█████╗  {reset}
{blue}  ██║     ██║     ██╔══██║██║   ██║██║  ██║██╔══╝  {reset}
{blue}  ╚██████╗███████╗██║  ██║╚██████╔╝██████╔╝███████╗{reset}
{blue}   ╚═════╝╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚═════╝ ╚══════╝{reset}

{dim}         Claude Code · from scratch
        A god-tier coding agent{reset}
"""


def _visible_len(s: str) -> int:
    """Return the visible length of a string, stripping ANSI escape codes."""
    import re
    return len(re.sub(r'\033\[[0-9;]*m', '', s))


def show_splash() -> None:
    """Render the Arch Linux-style splash screen."""
    # Clear screen first
    os.system('clear' if os.name != 'nt' else 'cls')

    width = 74

    # Top decorative border (cyan gradient-style)
    print(f"{Colors.CYAN}{Colors.BOLD}╔{'═' * (width - 2)}╗{Colors.RESET}")

    # Logo centered inside the box
    logo = NANO_CLAUDE_LOGO.format(
        cyan=Colors.CYAN + Colors.BOLD,
        blue=Colors.BLUE + Colors.BOLD,
        reset=Colors.RESET,
        dim=Colors.DIM,
    )
    for line in logo.strip().split('\n'):
        stripped = line.strip()
        if not stripped:
            print(f"{Colors.CYAN}║{Colors.RESET}{' ' * (width - 2)}{Colors.CYAN}║{Colors.RESET}")
            continue
        vis_len = _visible_len(stripped)
        pad_total = width - 2 - vis_len
        left_pad = max(0, pad_total // 2)
        right_pad = max(0, pad_total - left_pad)
        print(f"{Colors.CYAN}║{Colors.RESET}{' ' * left_pad}{stripped}{' ' * right_pad}{Colors.CYAN}║{Colors.RESET}")

    # Separator inside box
    print(f"{Colors.CYAN}║{Colors.RESET}{' ' * (width - 2)}{Colors.CYAN}║{Colors.RESET}")

    # System info line
    import platform
    py_ver = platform.python_version()
    cwd = os.getcwd()
    home = os.path.expanduser('~')
    if cwd.startswith(home):
        cwd = '~' + cwd[len(home):]

    info_raw = f"  🐍 python {py_ver}  ·  📁 {cwd}  ·  🕐 {time.strftime('%H:%M:%S')}"
    info = f"{Colors.DIM}{info_raw}{Colors.RESET}"
    vis_info = len(info_raw)
    pad_info = width - 2 - vis_info
    print(f"{Colors.CYAN}║{Colors.RESET}{info}{' ' * max(0, pad_info)}{Colors.CYAN}║{Colors.RESET}")

    # Bottom with tagline
    print(f"{Colors.CYAN}║{Colors.RESET}{' ' * (width - 2)}{Colors.CYAN}║{Colors.RESET}")
    tagline_raw = "  ⚡ NanoClaude Code — Let's build something great."
    tagline = f"{Colors.BOLD}{Colors.BRIGHT_WHITE}{tagline_raw}{Colors.RESET}"
    pad_tag = width - 2 - len(tagline_raw)
    print(f"{Colors.CYAN}║{Colors.RESET}{tagline}{' ' * max(0, pad_tag)}{Colors.CYAN}║{Colors.RESET}")

    # Bottom border
    print(f"{Colors.CYAN}{Colors.BOLD}╚{'═' * (width - 2)}╝{Colors.RESET}")
    print()


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



_DETAILED_STR_REPLACE_EDITOR_DESCRIPTION = """Custom editing tool for viewing, creating and editing files in plain-text format
* State is persistent across command calls and discussions with the user
* If `path` is a text file, `view` displays the result of applying `cat -n`. If `path` is a directory, `view` lists non-hidden files and directories up to 2 levels deep
* The following binary file extensions can be viewed in Markdown format: [".xlsx", ".pptx", ".wav", ".mp3", ".m4a", ".flac", ".pdf", ".docx"]. IT DOES NOT HANDLE IMAGES.
* The `create` command cannot be used if the specified `path` already exists as a file
* If a `command` generates a long output, it will be truncated and marked with `<response clipped>`
* The `undo_edit` command will revert the last edit made to the file at `path`
* This tool can be used for creating and editing files in plain-text format.


Before using this tool:
1. Use the view tool to understand the file's contents and context
2. Verify the directory path is correct (only applicable when creating new files):
   - Use the view tool to verify the parent directory exists and is the correct location

When making edits:
   - Ensure the edit results in idiomatic, correct code
   - Do not leave the code in a broken state
   - Always use absolute file paths (starting with /)

CRITICAL REQUIREMENTS FOR USING THIS TOOL:

1. EXACT MATCHING: The `old_str` parameter must match EXACTLY one or more consecutive lines from the file, including all whitespace and indentation. The tool will fail if `old_str` matches multiple locations or doesn't match exactly with the file content.

2. UNIQUENESS: The `old_str` must uniquely identify a single instance in the file:
   - Include sufficient context before and after the change point (3-5 lines recommended)
   - If not unique, the replacement will not be performed

3. REPLACEMENT: The `new_str` parameter should contain the edited lines that replace the `old_str`. Both strings must be different.

Remember: when making multiple file edits in a row to the same file, you should prefer to send all edits in a single message with multiple calls to this tool, rather than multiple messages with a single call each.
"""




def file_editor_tool() -> dict:
    
    return {
        'type': 'function',
        'function': {
            'name': 'file_editor',
            'description': _DETAILED_STR_REPLACE_EDITOR_DESCRIPTION,
            'parameters': {
                'type': 'object',
                'properties': {
                    'command': {
                        'description': 'The commands to run. Allowed options are: `view`, `create`, `str_replace`, `insert`, `undo_edit`.',
                        'enum': [
                            'view',
                            'create',
                            'str_replace',
                            'insert',
                            'undo_edit',
                        ],
                        'type': 'string',
                    },
                    'path': {
                        'description': 'Absolute path to file or directory, e.g. `/workspace/file.py` or `/workspace`.',
                        'type': 'string',
                    },
                    'file_text': {
                        'description': 'Required parameter of `create` command, with the content of the file to be created.',
                        'type': 'string',
                    },
                    'old_str': {
                        'description': 'Required parameter of `str_replace` command containing the string in `path` to replace.',
                        'type': 'string',
                    },
                    'new_str': {
                        'description': 'Optional parameter of `str_replace` command containing the new string (if not given, no string will be added). Required parameter of `insert` command containing the string to insert.',
                        'type': 'string',
                    },
                    'insert_line': {
                        'description': 'Required parameter of `insert` command. The `new_str` will be inserted AFTER the line `insert_line` of `path`.',
                        'type': 'integer',
                    },
                    'view_range': {
                        'description': 'Optional parameter of `view` command when `path` points to a file. If none is given, the full file is shown. If provided, the file will be shown in the indicated line number range, e.g. [11, 12] will show lines 11 and 12. Indexing at 1 to start. Setting `[start_line, -1]` shows all lines from `start_line` to the end of the file.',
                        'items': {'type': 'integer'},
                        'type': 'array',
                    },
                },
                'required': ['command', 'path'],
            },
        },
    }


def get_weather_fn(location):
    return str({"weather": "Sunny", "temp": "28C", "location": location})

from typing import Literal, Tuple

HARD_TIMEOUT = 60 

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


# File Editor

# ═══════════════════════════════════════════════════════════════════════════
# runtime/exceptions.py
# ═══════════════════════════════════════════════════════════════════════════

class ToolError(Exception):
    """Raised when a tool encounters an error."""

    def __init__(self, message):
        self.message = message
        super().__init__(message)

    def __str__(self):
        return self.message


# ═══════════════════════════════════════════════════════════════════════════
# runtime/edit.py
# ═══════════════════════════════════════════════════════════════════════════

Command = Literal[
    'view',
    'create',
    'str_replace',
    'insert',
    'undo_edit',
]

SNIPPET_CONTEXT_WINDOW = 4


@dataclass
class FileEditObservation:
    path: Path
    success_message: str | None = None
    file_content: str | None = None
    new_file_content: str | None = None
    output: str | None = None


class Editor:
    def __init__(self):
        pass

    def __call__(
        self,
        command: Command,
        path: str,
        file_text: str | None = None,
        view_range: list[int] | None = None,
        old_str: str | None = None,
        new_str: str | None = None,
        insert_line: int | None = None,
        enable_linting: bool = False,
        **kwargs,
    ):
        _path = Path(path)
        if command == 'view':
            if view_range:
                start_line, end_line = view_range
                out = self.read_file(_path, start_line, end_line)
                success_msg = f'Successfully read the file from line {start_line} to {end_line} file content:\n{out}'
            else:
                out = self.read_file(_path)
                success_msg = f'Successfully read the file content:\n{out}'
            return FileEditObservation(
                path=_path,
                success_message=success_msg,
            )
        if command == 'create':
            self.write_file(path=_path, file_text=file_text)
            return FileEditObservation(
                path=_path,
                success_message=f'Successfully Create a file at {_path} with file content:\n{file_text}',
            )
        elif command == 'insert':
            return self.insert(_path, insert_line, new_str)
        elif command == 'str_replace':
            return self.str_replace(_path, new_str=new_str, old_str=old_str, enable_linting=False)

    def view(self, path, view_range) -> FileEditObservation:
        start_line, end_line = view_range
        out = self.read_file(path, start_line, end_line)
        return FileEditObservation(path=path, output=out)

    def read_file(self, path: Path, start_line: int | None = None, end_line: int | None = None):
        try:
            if start_line is not None and end_line is not None:
                with open(path, 'r', encoding='utf-8') as f:
                    text = []
                    for i, line in enumerate(f, 1):
                        if i >= start_line:
                            text.append(line)
                        if i > end_line:
                            break
                return ''.join(text)
            else:
                with open(path, 'r', encoding='utf-8') as f:
                    return ''.join(f)
        except Exception as e:
            raise ToolError(f'Error {e} ')

    def insert(self, path: Path, insert_line: int, new_str: str):
        try:
            with tempfile.NamedTemporaryFile(
                mode='w', encoding='utf-8', delete=False
            ) as temp_file:
                with open(path, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f, 1):
                        if i < insert_line:
                            temp_file.write(line)
                        if i > insert_line:
                            break

                new_str_ls = new_str.split('\n')
                for line in new_str_ls:
                    temp_file.write(line + '\n')

                with open(path, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f, 1):
                        if i >= insert_line:
                            temp_file.write(line)

                shutil.move(temp_file.name, path)

            return FileEditObservation(
                success_message=f'Successfully Inserted this code {new_str}',
                path=path,
            )
        except Exception as e:
            raise ToolError(f'Error {e}')

    def str_replace(
        self,
        path: Path,
        old_str: str,
        new_str: str | None,
        enable_linting: bool,
    ) -> FileEditObservation:
        new_str = new_str or ''

        file_content = self.read_file(path)

        pattern = re.escape(old_str)
        occurrences = [
            (
                file_content.count('\n', 0, match.start()) + 1,
                match.group(),
                match.start(),
            )
            for match in re.finditer(pattern, file_content)
        ]

        if not occurrences:
            raise ToolError(
                f'No replacement was performed, old_str `{old_str}` did not appear verbatim in {path}.'
            )
        if len(occurrences) > 1:
            line_numbers = sorted(set(line for line, _, _ in occurrences))
            raise ToolError(
                f'No replacement was performed. Multiple occurrences of old_str `{old_str}` in lines {line_numbers}. Please ensure it is unique.'
            )

        replacement_line, matched_text, idx = occurrences[0]

        new_file_content = (
            file_content[:idx] + new_str + file_content[idx + len(matched_text):]
        )

        self.write_file(path, new_file_content)

        start_line = max(0, replacement_line - SNIPPET_CONTEXT_WINDOW)
        end_line = replacement_line + SNIPPET_CONTEXT_WINDOW + new_str.count('\n')

        snippet = self.read_file(path, start_line=start_line + 1, end_line=end_line)

        success_message = f'The file {path} has been edited. '
        success_message += f'a snippet of {path}\nsnippet:\n{snippet}'

        return FileEditObservation(
            success_message=success_message,
            path=path,
            file_content=file_content,
            new_file_content=new_file_content,
        )

    def write_file(self, path: Path, file_text: str) -> None:
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(file_text)
        except Exception as e:
            raise ToolError(f'Ran into {e} while trying to write to {path}') from None


class NanoClaude:

    def __init__(self):
        # ── agent state ─────────────────────────────────────────────────
        self.history = [SYSTEM_PROMPT]
        
        # ADD TOOLs
        self.tools = [bash_tool(), file_editor_tool()]

        #initialize the bash tool
        self.bash_session = BashSession(
            work_dir=os.getcwd(),
            username="to",
        )
        self.bash_session.initialize()

        self.editor = Editor()


    def step(self):
        # Show subtle thinking indicator
        print(f"{Colors.DIM}  ⋯ thinking{Colors.RESET}", end='\r', flush=True)

        params = {
            **Config().model_dump(),
            "thinking": {"type": "enabled"},
            "api_key": os.environ['LLM_API_KEY']
        }
        params["messages"] = self.history
        params["tools"] = self.tools # Add tools
        result = completion(**params)

        # Clear the thinking line
        print(' ' * 40, end='\r')
        return result

    def execute(self, user_message):
        
        self.history.append(user_message.dict())

        while True:
            response = self.step()
            assistant_msg = response.choices[0].message

            if assistant_msg.content:
                print_assistant_message(assistant_msg.content)

            # ADD TOOLS
            if hasattr(assistant_msg, "tool_calls") and assistant_msg.tool_calls:
                self.history.append(assistant_msg.model_dump())
                for tool in assistant_msg.tool_calls:
                    observation = self.perform_action(tool)
                    self.history.append(observation)

            # natural stop
            if response.choices[0].finish_reason == "stop":
                print_finish()
                break
    
    def perform_action(self, tool):
        tool_name = tool.function.name
        arguments = json.loads(tool.function.arguments)

        # Pretty-print the tool call with all parameters
        print_tool_call(tool_name, arguments)

        if tool_name == "execute_bash":
            out = self.bash_session.execute(arguments.get("command", ""))
            result = out.to_agent_observation()
            print_tool_result(success=True, observation=result)
            return convert_obs_to_json(result=result, tool=tool)

        elif tool_name == "file_editor":
            args = {
                "command": arguments.get("command", ""),
                "path": arguments.get("path", ""),
                "file_text": arguments.get("file_text", ""),
                "old_str": arguments.get("old_str", ""),
                "new_str": arguments.get("new_str", ""),
                "insert_line": arguments.get("insert_line", ""),
                "view_range": arguments.get("view_range", ""),
            }
            try:
                result = self.editor(**args)
                result_text = result.success_message
                print_tool_result(success=True, observation=result_text)
            except ToolError as e:
                result_text = e.message
                print_tool_result(success=False, summary=f"{Colors.RED}{e.message}{Colors.RESET}", observation=result_text)
            return convert_obs_to_json(result=result_text, tool=tool)
            


    # ── interactive repl ────────────────────────────────────────────────
    def repl(self):
        show_splash()
        try:
            while True:
                print_prompt()
                instruction = input(f"{Colors.BRIGHT_WHITE}  {Colors.RESET}").strip()

                if instruction.lower() in ("quit", "exit", "q"):
                    print()
                    print(f"{Colors.DIM}  Shutting down...{Colors.RESET}")
                    break

                if not instruction:
                    continue

                content = Content(type="text", text=instruction)
                user_message = Message(content=[content], role="user")

                self.execute(user_message=user_message)

        except KeyboardInterrupt:
            print()
            print(f"{Colors.YELLOW}  ⚠ Agent stopped by user!{Colors.RESET}")
        except Exception as e:
            print(f"{Colors.RED}  ✗ Unexpected error: {e}{Colors.RESET}")
            traceback.print_exc()
        finally:
            print()
            print_separator('═', 70)
            print(f"{Colors.DIM}  ⚡ NanoClaude session ended — see you soon!{Colors.RESET}")
            print()


nanoclaude = NanoClaude()
nanoclaude.repl()