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
load_dotenv()

os.environ["EMBEDDINGS_OLLAMA_MODEL_NAME"] = "nomic-embed-text"
os.environ["EMBEDDINGS_OLLAMA_API_BASE"] = "http://localhost:11434"

# ==========================================
# 1. LLM & TOOL CONFIGURATION
# ==========================================

local_llm = LLM(
    model="openai/qwen-game-dev", 
    base_url="http://localhost:11434/v1", 
    api_key="ollama-local",
    context_window_size=16384
)

repo_reader = DirectoryReadTool(directory='./')
file_reader = FileReadTool()
docs_scraper = ScrapeWebsiteTool()

@tool("File Writer Tool")
def file_writer_tool(file_path: str, content: str) -> str:
    """Overwrites a file at the given file_path with the provided content."""
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    return f"Success: Updated {file_path}"

@tool("Ruff Linter Tool")
def ruff_linter_tool(directory: str = "src/") -> str:
    """Runs Ruff to instantly catch syntax errors."""
    result = subprocess.run(['ruff', 'check', directory], capture_output=True, text=True, check=False)
    return f"Ruff Output:\n{result.stdout}\n{result.stderr}"

@tool("Pytest Execution Tool")
def pytest_execution_tool(test_directory: str = "tests/") -> str:
    """Runs pytest on the specified directory."""
    try:
        result = subprocess.run(['pytest', test_directory], capture_output=True, text=True, check=False)
        return f"=== TEST RESULTS ===\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    except Exception as e:
        return f"Tool Error: {str(e)}"

@tool("Terminal Execution Tool")
def terminal_execution_tool(command: str) -> str:
    """Executes a terminal/shell command safely inside Docker."""
    try:
        if "[SECURE_TOKEN]" in command:
            command = command.replace("[SECURE_TOKEN]", os.getenv("GITHUB_TOKEN"))
            
        safe_command = f'docker exec dev_sandbox sh -c "{command}"'
        result = subprocess.run(safe_command, shell=True, capture_output=True, text=True, timeout=60)
        return f"Output:\n{result.stdout}\n{result.stderr}"
    except Exception as e:
        return f"Tool Error: {str(e)}"


# ==========================================
# 2. DEFINE THE AGENTS (Delegation Disabled)
# ==========================================

pm = Agent(
    role='Lead Product Manager',
    goal='Extract comprehensive technical requirements from the approved chat log.',
    backstory='You take the raw chat history and format it into a pristine technical specification.',
    llm=local_llm,
    verbose=True,
    allow_delegation=False
)

designer = Agent(
    role='Lead Game System Architect',
    goal='Design highly optimized, low-latency game systems.',
    backstory='You design scalable state machines and entity systems before code is written.',
    llm=local_llm,
    tools=[repo_reader, file_reader, file_writer_tool],
    verbose=True,
    allow_delegation=False
)

coder = Agent(
    role='Senior Gameplay Engineer',
    goal='Write clean, efficient code based on the architecture specs.',
    backstory='You translate design documents into production-ready code.',
    llm=local_llm,
    tools=[repo_reader, file_reader, file_writer_tool, docs_scraper],
    verbose=True,
    allow_delegation=False
)

tester = Agent(
    role='Lead SDET',
    goal='Write automated tests and execute them to ensure data flow integrity.',
    backstory='You build bulletproof test suites. You always run Ruff for syntax checks before Pytest.',
    llm=local_llm,
    tools=[repo_reader, file_reader, file_writer_tool, ruff_linter_tool, pytest_execution_tool],
    verbose=True,
    allow_delegation=False
)

reviewer = Agent(
    role='Security & QA Lead',
    goal='Review code for bugs and rewrite failing logic.',
    backstory='You are a ruthless code reviewer. If tests fail, you use your tools to rewrite the source code and test again until it passes.',
    llm=local_llm,
    tools=[repo_reader, file_reader, file_writer_tool, ruff_linter_tool, pytest_execution_tool],
    verbose=True,
    allow_delegation=False
)

devops = Agent(
    role='Lead DevOps & Release Engineer',
    goal='Commit the final code to the repository.',
    backstory='You are a terminal-first Linux veteran. You ensure code is pushed safely to Git.',
    llm=local_llm,
    tools=[repo_reader, file_reader, terminal_execution_tool],
    verbose=True,
    allow_delegation=False
)


# ==========================================
# 3. DEFINE THE TASKS
# ==========================================

planning_task = Task(
    description='Awaiting Streamlit Input...',
    expected_output='A final project specification saved to the docs folder.',
    agent=pm
)

design_task = Task(
    description='''
    Based on the specification, design the technical architecture.
    Use the File Writer Tool to save it to 'docs/2_architecture_design.md'.
    ''',
    expected_output='A technical design document saved to the docs folder.',
    agent=designer
)

code_task = Task(
    description='''
    1. Read the specification and architecture design.
    2. Implement the architecture in Python.
    3. Save the code to 'src/state_machine.py'.
    ''',
    expected_output='Complete, executable code saved to the repository.',
    agent=coder
)

test_task = Task(
    description='''
    1. Write pytest tests for the codebase and save to 'tests/test_state_machine.py'.
    2. Use the Ruff Linter Tool first. Only run the Pytest Execution Tool once Ruff reports zero errors.
    ''',
    expected_output='A complete test suite saved to the repository.',
    agent=tester
)

review_task = Task(
    description='''
    1. Read the source code and the test results.
    2. If tests failed, use the File Writer Tool to fix bugs in the source code.
    3. Use Ruff Linter Tool, then Pytest Execution Tool to verify fixes.
    4. Repeat until all tests pass.
    ''',
    expected_output='A bug-free codebase that passes all tests.',
    agent=reviewer
)

deployment_task = Task(
    description='''
    Use the Terminal Execution Tool to execute this exact chain of commands:
    1. `git add .`
    2. `git commit -m "Automated AI implementation"`
    3. `git push https://[SECURE_TOKEN]@github.com/isaacmcelwee/Ninefold.git main`
    ''',
    expected_output='A successful Git commit and push output.',
    agent=devops
)


# ==========================================
# 4. ASSEMBLE CREW & EXPORT
# ==========================================

dev_team = Crew(
    agents=[pm, designer, coder, tester, reviewer, devops],
    tasks=[planning_task, design_task, code_task, test_task, review_task, deployment_task],
    process=Process.sequential,
    memory=True
)

def run_pipeline(approved_spec_log):
    planning_task.description = f'''
    The product owner has approved the following conversation log:
    "{approved_spec_log}"
    
    Analyze this log, extract the requirements, and output the final specification. 
    Save it using the File Writer Tool to 'docs/1_product_spec.md'.
    '''
    
    print("\nBooting dev team assembly line from Web UI...")
    return dev_team.kickoff()