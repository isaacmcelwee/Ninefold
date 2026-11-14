import os
import sys
import subprocess
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import DirectoryReadTool, FileReadTool, ScrapeWebsiteTool
from crewai.tools import tool

# ==========================================
# 0. SECRETS & MEMORY INIT
# ==========================================
# Loads GITHUB_TOKEN from your .env file
load_dotenv()

# Explicitly configure CrewAI memory to use the local Ollama embedder
os.environ["EMBEDDINGS_OLLAMA_MODEL_NAME"] = "nomic-embed-text"
os.environ["EMBEDDINGS_OLLAMA_API_BASE"] = "http://localhost:11434"

# ==========================================
# 1. LLM & TOOL CONFIGURATION
# ==========================================

# Connect to your expanded local Qwen model using Ollama's OpenAI-compatible endpoint
local_llm = LLM(
    model="openai/qwen-game-dev", 
    base_url="http://localhost:11434/v1", 
    api_key="ollama-local",
    context_window_size=16384
)

# Built-in tools for reading the repository and scraping docs
repo_reader = DirectoryReadTool(directory='./')
file_reader = FileReadTool()
docs_scraper = ScrapeWebsiteTool()

# Custom Tool: Write/Overwrite Files
@tool("File Writer Tool")
def file_writer_tool(file_path: str, content: str) -> str:
    """Overwrites a file at the given file_path with the provided content. Use this to update existing code."""
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    return f"Success: Updated {file_path}"

# Custom Tool: Ruff Linter (Catches syntax errors instantly)
@tool("Ruff Linter Tool")
def ruff_linter_tool(directory: str = "src/") -> str:
    """Runs Ruff to instantly catch syntax errors and missing imports. Run this BEFORE pytest."""
    result = subprocess.run(['ruff', 'check', directory], capture_output=True, text=True, check=False)
    return f"Ruff Output:\n{result.stdout}\n{result.stderr}"

# Custom Tool: Execute Pytest (Catches logic errors)
@tool("Pytest Execution Tool")
def pytest_execution_tool(test_directory: str = "tests/") -> str:
    """Runs pytest on the specified directory and returns the terminal output and error logs."""
    try:
        result = subprocess.run(['pytest', test_directory], capture_output=True, text=True, check=False)
        return f"=== TEST RESULTS ===\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    except Exception as e:
        return f"Tool Error: Could not execute pytest. Details: {str(e)}"

# Custom Tool: Safe Docker Terminal (For DevOps)
@tool("Terminal Execution Tool")
def terminal_execution_tool(command: str) -> str:
    """Executes a terminal/shell command safely inside the sandboxed Docker container."""
    try:
        # Securely inject the token in Python so the LLM never sees it
        if "[SECURE_TOKEN]" in command:
            command = command.replace("[SECURE_TOKEN]", os.getenv("GITHUB_TOKEN"))
            
        # Routes the command into the isolated dev_sandbox container
        safe_command = f'docker exec dev_sandbox sh -c "{command}"'
        result = subprocess.run(safe_command, shell=True, capture_output=True, text=True, timeout=60)
        return f"Output:\n{result.stdout}\n{result.stderr}"
    except Exception as e:
        return f"Tool Error: {str(e)}"


# ==========================================
# 2. DEFINE THE AGENTS
# ==========================================

manager = Agent(
    role='Director of Game Engineering',
    goal='Oversee the software development lifecycle, delegate tasks to the appropriate experts, and ruthlessly verify the output.',
    backstory=(
        "You are a strict, highly technical engineering director. "
        "To assign work, you MUST natively invoke the 'delegate_work_to_coworker' tool. "
        "NEVER output raw JSON or use 'Final Answer' to delegate tasks. "
        "You never accept code that hasn't been linted, tested, and reviewed. "
        "If the Tester or Reviewer finds bugs, you immediately send the work back to the Coder. "
        "You ensure the final output strictly matches the initial architectural specification."
    ),
    llm=local_llm,
    verbose=True,
    allow_delegation=True # Required for the manager to assign tasks
)

pm = Agent(
    role='Lead Product Manager',
    goal='Extract comprehensive technical requirements from the approved chat log.',
    backstory='You take the raw chat history between the user and the frontend PM, and format it into a pristine technical specification.',
    llm=local_llm,
    verbose=True
)

designer = Agent(
    role='Lead Game System Architect',
    goal='Design highly optimized, low-latency game systems.',
    backstory='You design scalable state machines and entity systems before code is written.',
    llm=local_llm,
    tools=[repo_reader, file_reader, file_writer_tool],
    verbose=True
)

coder = Agent(
    role='Senior Gameplay Engineer',
    goal='Write clean, efficient code based on the architecture specs.',
    backstory='You translate design documents into production-ready code. You read documentation before using unfamiliar libraries.',
    llm=local_llm,
    tools=[repo_reader, file_reader, file_writer_tool, docs_scraper],
    verbose=True
)

tester = Agent(
    role='Lead SDET',
    goal='Write automated tests and execute them to ensure data flow integrity.',
    backstory='You build bulletproof test suites. You always run Ruff for syntax checks before running heavy Pytest logic checks.',
    llm=local_llm,
    tools=[repo_reader, file_reader, file_writer_tool, ruff_linter_tool, pytest_execution_tool],
    verbose=True
)

reviewer = Agent(
    role='Security & QA Lead',
    goal='Review code for memory leaks, bugs, and security vulnerabilities.',
    backstory='You are a ruthless code reviewer who finds flaws before they reach the main branch.',
    llm=local_llm,
    tools=[repo_reader, file_reader, file_writer_tool, ruff_linter_tool, pytest_execution_tool],
    verbose=True
)

devops = Agent(
    role='Lead DevOps & Release Engineer',
    goal='Manage the repository, install dependencies, and commit the final code.',
    backstory='You are a terminal-first Linux veteran. You ensure all code is properly tested in Docker and committed to Git.',
    llm=local_llm,
    tools=[repo_reader, file_reader, terminal_execution_tool],
    verbose=True
)


# ==========================================
# 3. DEFINE THE TASKS (Delegated by Manager)
# ==========================================

planning_task = Task(
    description='Awaiting Streamlit Input...',
    expected_output='A final project specification saved to the docs folder.',
    human_input=False 
)

design_task = Task(
    description='''
    Based on the finalized specification, design the technical architecture and state machine.
    Use the File Writer Tool to save the design document to 'docs/2_architecture_design.md'.
    ''',
    expected_output='A technical design document saved to the docs folder.'
)

code_task = Task(
    description='''
    1. Read the specification and architecture design.
    2. Optional: Use ScrapeWebsiteTool for updated API docs if needed.
    3. Implement the architecture in Python.
    4. Save the updated code to 'src/state_machine.py'.
    ''',
    expected_output='Complete, executable code saved to the repository.'
)

test_task = Task(
    description='''
    1. Write pytest tests for the codebase and save to 'tests/test_state_machine.py'.
    2. Use the Ruff Linter Tool first. Only run the Pytest Execution Tool once Ruff reports zero errors.
    3. If tests fail, rewrite the tests or report the traceback.
    ''',
    expected_output='A passing test suite or a detailed bug report.',
    async_execution=True
)

review_task = Task(
    description='''
    1. Read the source code and the test results.
    2. If tests failed, fix bugs in the source code.
    3. Use Ruff Linter Tool, then Pytest Execution Tool to verify fixes.
    4. Repeat until all tests pass.
    ''',
    expected_output='A bug-free codebase that passes all tests.'
)

deployment_task = Task(
    description='''
    Use the Terminal Execution Tool to execute this exact chain of commands:
    1. `git add .`
    2. `git commit -m "Automated AI implementation"`
    3. `git push https://[SECURE_TOKEN]@github.com/isaacmcelwee/Ninefold.git main`
    ''',
    expected_output='A successful Git commit and push output from the terminal tool.'
)


# ==========================================
# 4. ASSEMBLE CREW & EXPORT FUNCTION
# ==========================================

dev_team = Crew(
    agents=[pm, designer, coder, tester, reviewer, devops],
    tasks=[planning_task, design_task, code_task, test_task, review_task, deployment_task],
    process=Process.hierarchical,
    manager_agent=manager,
    memory=True
)

def run_pipeline(approved_spec_log):
    """
    Called by Streamlit app.py to trigger the CrewAI agents.
    Injects the chat history directly into the PM's task description.
    """
    planning_task.description = f'''
    The product owner has approved the following conversation log:
    "{approved_spec_log}"
    
    Analyze this log, extract the requirements, and output the final specification. 
    Save it using the File Writer Tool to 'docs/1_product_spec.md'.
    '''
    
    print("\nBooting dev team manager from Web UI...")
    return dev_team.kickoff()