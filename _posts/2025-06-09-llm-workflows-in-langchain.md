---
layout: single
title: "The Foundation: LLM Powered Pharmacometric Workflows"
date: 2025-06-09
categories: [demo]
tags: [pharmacometrics, AI, langchain, R, automation]
classes: wide
---

## From Task to Code: Building Reliable, Iterative LLM Workflows with LangChain

*A practical guide to transforming the concepts of AI-driven pharmacometrics into the executable code of today.*

In my first post, we explored the "iterative hell" of manual pharmacometric workflows and laid out a vision for how AI agents could transform our field. The response was clear: the vision is compelling, but how do we get from today's reality to that AI-driven future?

This post is the answer. It's the practical, hands-on guide to building the foundational engine for a pharmacometric AI agent. We will walk through the [accompanying Jupyter notebook](https://github.com/AriAnthony/PharmAI/blob/main/getting_started/pharmacometric_ai_demo.ipynb) step-by-step, turning the concepts of yesterday's vision into the executable code of today. By the end, you'll understand not just *how* to build a basic agent, but *why* each technical choice is a deliberate step toward a more powerful and reliable system.

### The Foundational Atom: Plan, Act, Observe, Reflect

Before we dive into the code, it's important to understand the fundamental pattern we're building. The workflow in our notebook represents the "atom" of any advanced AI agent:

1.  **Plan:** The LLM analyzes the task and context to create an R script—its plan of action.
2.  **Act:** The `subprocess` module executes that script in a controlled environment.
3.  **Observe:** We capture the results of the action—the `STDOUT` and `STDERR`.
4.  **Reflect:** We feed these observations back to the LLM to evaluate the outcome and decide on the next action, which could be to finish or to generate a new, corrected plan.

This cyclical, state-driven process is the core idea behind modern agentic frameworks like LangChain's [LangGraph](https://langchain-ai.github.io/langgraph/), which is designed specifically to handle these iterative loops of thought and action. Every complex system we can imagine—from automated PopPK model development to full-blown MIDD strategy design—is built by composing these simple, powerful atoms. Today, we're mastering the atom.

### Security First: Why Local LLMs Matter

As we begin, we'll re-emphasize the point from our first post: data security is non-negotiable. Using local LLMs via tools like [Ollama](https://ollama.com/) ensures that all proprietary data, from compound structures to clinical trial results, remains securely on-premises.
In this demo I am using Ollama to run local LLMs, ensuring that all data remains within your environment. However, foundation model providers like [Anthropic](https://www.anthropic.com/), [OpenAI](https://openai.com/), and [AWS Bedrock](https://aws.amazon.com/bedrock/) offer enterprise-grade security. Using these models will greatly enhance the capabilities of this agentic workflow, and thanks to LangChain's modular design, you can easily swap in these providers without changing the core logic of the agent (e.g., ChatBedrock instead of ChatOllama).

### Setting Up the Foundation

Our notebook begins with essential installations and a check to ensure Ollama is running with the required model. You can search various models at [Ollama's model hub](https://ollama.com/models). For this demo, we will use the [`qwen2.5-coder:7b`](https://ollama.com/library/qwen2.5-coder) model, which is well-suited for code generation tasks. Crucially, it has been trained for tool use (Ollama tags these with "tools") which allows it to generate structured outputs that we can validate and execute.

```python
# Install requirements if needed
# !pip install langchain langchain-community pydantic ollama pandas

import subprocess
from pathlib import Path
from pydantic import BaseModel, Field
import pandas as pd

# Check Ollama is running
try:
    result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
    print("✅ Ollama is running")
    print("Available models:", result.stdout.split('\n')[1] if result.stdout else "No models found")
except:
    print("❌ Ollama not found. Install with: curl -fsSL https://ollama.com/install.sh | sh")
    print("Then: ollama pull qwen2.5-coder:7b")

LLM_MODEL = "qwen2.5-coder:7b" # High performing model with tool use capabilities
```

### Step 1: Define Data Structures with Pydantic

The key to reliable AI workflows is structure. We use Pydantic to define what we expect from the LLM. This isn't just a Python best practice; it's the mechanism by which we build the **reliability and structured output** necessary for a system that can make **auditable decisions**, a key requirement we discussed for regulated environments. This capability, often referred to as 'tool use' or 'function calling,' is a critical feature in modern LLMs that allows them to interact with the world in a reliable, programmatic way. Refer to [OpenAI](https://platform.openai.com/docs/guides/function-calling) or [Anthropic](https://docs.anthropic.com/claude/docs/tool-use) for more details. [LangChain](https://python.langchain.com/v0.2/docs/how_to/structured_output/#the-with_structured_output-method) provides powerful abstractions to enforce this structured output.

```python
# Define our input schema 
class TaskInput(BaseModel):
    """Structure for pharmacometric analysis tasks"""
    task_name: str = Field(description="Short name for the analysis")
    task_details: str = Field(description="Natural language description of what to do")
    data_directory: str = Field(default="./data", description="Path to data files")

# Define our output schema
class AnalysisOutput(BaseModel):
    """What our LLM returns"""
    script: str = Field(description="Complete executable R script")
    thoughts: str = Field(description="LLM reasoning about the approach")
    status_complete: bool = Field(default=False, description="Whether the task is complete")
```

### Step 2: Build Up the Context for the LLM

An AI agent's intelligence comes from the knowledge it can access. In the first post, we talked about the power of **domain knowledge**. Providing data context and high-quality code examples is the most direct way to inject that knowledge into our agent, teaching it *how a pharmacometrician thinks*. This process of carefully crafting the input to the model is known as [prompt engineering](https://docs.anthropic.com/claude/docs/prompt-engineering), a crucial discipline for eliciting high-quality, relevant responses from LLMs.

```python
# Create a data preview for all CSVs in the data directory
data_context = ""
# Generalized data preview for all CSVs in DATA_DIR
DATA_DIR = "./data"
for csv_file in Path(DATA_DIR).glob("*.csv"):
    try:
        df_head = pd.read_csv(csv_file).head().to_string()
        data_context += f"\n--- pd.head() for file: {csv_file.name} ---\n{df_head}\n"
    except Exception as e:
        data_context += f"\n--- {csv_file.name} ---\nError reading file: {e}\n"
print("Data context loaded:\n", data_context)

# Provide an example of a high-quality nlmixr2 script
example_context = """
library(nlmixr2)

## The basic model consists of an ini block that has initial estimates
one.compartment <- function() {
  ini({
    tka <- log(1.57); label("Ka")
    tcl <- log(2.72); label("Cl")
    tv <- log(31.5); label("V")
    eta.ka ~ 0.6
    eta.cl ~ 0.3
    eta.v ~ 0.1
    add.sd <- 0.7
  })
  # and a model block with the error specification and model specification
  model({
    ka <- exp(tka + eta.ka)
    cl <- exp(tcl + eta.cl)
    v <- exp(tv + eta.v)
    d/dt(depot) <- -ka * depot
    d/dt(center) <- ka * depot - cl / v * center
    cp <- center / v
    cp ~ add(add.sd)
  })
}

## The fit is performed by the function nlmixr/nlmixr2 specifying the model, data and estimate
fit <- nlmixr(one.compartment, theo_sd, "saem",
              control=list(print=0), 
              table=list(cwres=TRUE, npde=TRUE))

# Print summary of parameter estimates and confidence intervals
print(fit)

# Basic Goodness of Fit Plots
plot(fit)
"""
```
What we've done here is build up a string `data_context` that contains a preview of all CSV files in the `./data` directory, which is where our pharmacometric data is stored. We also build an `example_context` string that contains a high-quality example of an R script using `nlmixr2`, which is a common tool in pharmacometrics for nonlinear mixed effects modeling. This example will serve as a reference for the LLM to understand the structure and style of the scripts we want it to generate.

### Step 3: Define the LLM Chain and First Pass

Now we assemble the prompt and use LangChain to call the LLM, requesting a structured response based on our `AnalysisOutput` model. This is the **Plan** step in our agent loop. 

```python
from langchain_ollama import ChatOllama
from langchain.schema import HumanMessage, SystemMessage, AIMessage

# Define a sample analysis task
sample_task = TaskInput(
    task_name="fit_poppk_model",
    task_details="""
    Load population pharmacokinetic data from the data directory.
    Fit a nonlinear mixed effects model using nlmixr2. 
    Print a summary table of parameter estimates and confidence intervals.
    Use ggplot2 for visualization.
    Write complete, executable R scripts using tidyverse principles
    """,
    data_directory="./data"
)

# Define the system prompt 
system_prompt = f"""You are an expert pharmacometrician generating R scripts. Data for your analysis is located in {sample_task.data_directory}.

Here is a summary of the data:
<data_context>
{data_context}
</data_context>

<task> 
{sample_task.task_name} 
</task> 

<task_details>
{sample_task.task_details}
</task_details>

<example_context>
{example_context}
</example_context>
"""

messages = [
    SystemMessage(content=system_prompt),
    HumanMessage(content="Please generate the R script for the above task.")
]

# Set up the LLM to return structured output
llm = ChatOllama(model=LLM_MODEL, temperature=0.1)
llm_struct = llm.with_structured_output(AnalysisOutput)

# Call the LLM
print("🧠 Generating analysis script...")
generated_analysis = llm_struct.invoke(messages)
```

### Step 4: Safe R Script Execution

With a plan in hand, the agent must now **Act** and **Observe**. We run the script using `subprocess`, carefully capturing all output. This controlled execution is vital for debugging and for the agent's learning process.

```python
# Write the generated script to a file
script_path = Path(sample_task.task_name.replace(" ", "_") + "_analysis.R")
script_path.write_text(generated_analysis.script)

# Execute the R script and capture the output
result = subprocess.run(
    ["Rscript", str(script_path)],
    cwd=".",
    capture_output=True,
    text=True,
    timeout=180
)

# Combine stdout and stderr for feedback
output = f"STDOUT:\\n{result.stdout}\\n\\nSTDERR:\\n{result.stderr}"
print(output)
```

### Step 5: The Feedback Loop — From Chain to Agent

This is the crucial moment where our workflow graduates from a simple script into a basic **AI agent**. We **Reflect** by feeding the results of the execution back to the LLM. That `STDERR` message isn't an error for us to fix; it's a learning opportunity for the agent. This iterative loop of action and observation is the defining characteristic of an AI agent, allowing it to perform complex tasks that require trial, error, and self-correction. This pattern is often formalized in frameworks like [ReAct (Reasoning and Acting)](https://lilianweng.github.io/posts/2023-06-23-agent/) and is the foundation of modern agent development.

```python
# Append the model's last attempt and the script's output to the conversation
new_messages = messages.copy()
new_messages.append(
    AIMessage(content=generated_analysis.model_dump_json())
)
new_messages.append(
    HumanMessage(content=output)
)

# Ask the LLM to evaluate the result and decide if it's done or needs to try again
print("🧠 Re-evaluating based on R script output...")
new_generated_analysis = llm_struct.invoke(new_messages)

# Print the final status
print("✅ Final status complete:", new_generated_analysis.status_complete)
```

### Step 6: Iterative Agentic Capabilities

While a single feedback pass is powerful, true agentic behavior comes from automating this loop. Instead of manually feeding back the output once, we can create a function that handles the entire "Plan, Act, Observe, Reflect" cycle until the task is complete. The notebook demonstrates this by abstracting the logic into a reusable `llm_feedback_loop` function.

This function, defined in the `llm_feedback.py` helper file, encapsulates our agent's core logic. It repeatedly calls the LLM, executes the resulting script, and feeds the output back into the message history, allowing the agent to refine its approach over several attempts.

Here is how it's called in the notebook:
```python
from llm_feedback import llm_feedback_loop

results = llm_feedback_loop(
    llm=llm_struct,
    messages=new_messages,
    task=sample_task,
    max_iterations=3
)
```

And here is the implementation of the helper functions, showing the full automated loop:
```python
# From llm_feedback.py

from pathlib import Path
from langchain.schema import AIMessage, HumanMessage
import subprocess

# Define a function to iterate through LLM call and message appends
def llm_feedback_loop(task, llm, messages, max_iterations=5):
    """
    Iteratively call the LLM with the provided messages and append the response to the messages.
    """
    for i in range(max_iterations):
        response = llm.invoke(messages)

        print("✅ Analysis generated!")
        print(f"Completed Status: {response.status_complete}")
        print(f"\nLLM thoughts: {response.thoughts}")
        print(f"\n📝 Generated R script ({len(response.script)} characters):")

        # Check for llm completion flag BEFORE running the R script
        if response.status_complete:
            print("LLM response: completed successfully.")
            break

        # Run the R script and capture output
        output = run_r_script(task, response)
        print(output)

        # Append the model output and the script results to the messages
        messages.append(
            AIMessage(content=response.model_dump_json())
        )
        messages.append(
            HumanMessage(content=output)
        )

        print(f"Iteration {i + 1}: {response}")
    
    return response

# Define function to write and run R script
def run_r_script(task, response):
    # Write R script to the current directory
    script_path = Path(task.task_name.replace(" ", "_") + "_analysis.R")
    script_path.write_text(response.script)
    
    # Execute R script in the current directory
    result = subprocess.run(
        ["Rscript", str(script_path)],
        cwd=".",
        capture_output=True,
        text=True,
        timeout=180
    )
    # Check if the script ran successfully
    if result.returncode == 0:
       output = f"STDOUT:\n{result.stdout}"
    else:
        output = f"STDERR:\n{result.stderr}"
   
    return output
```

### What We Just Built: From Vision to Reality

By following the notebook, we've built more than a script. We've laid the groundwork for a true AI partner by solving the core challenges outlined in our first post:

1.  **Solved for Reliability:** We used Pydantic for structured outputs, moving beyond fragile scripts to the **auditable, reproducible workflows** required in our industry.
2.  **Solved for Decision-Making:** We built an iterative feedback loop, creating a basic agent that can **interpret results and self-correct**, moving beyond the limitations of traditional automation that cannot make judgments.
3.  **Solved for Security:** We used Ollama to run a powerful local LLM, ensuring all proprietary data and analysis **remain securely on-premises**.
4.  **Solved for Expertise:** We showed how to provide data and code context, the first step towards encoding **deep domain knowledge** into our automated systems.

### What's Next?

This foundational atom—Plan, Act, Observe, Reflect—is the key. The notebook abstracts this into a reusable function, paving the way for more complex agents. In future posts, we'll build on this to explore:

*   **Advanced Task Orchestration**: Composing our agentic atoms to tackle multi-step analyses.
*   **RAG for Domain Knowledge**: Granting our agent access to a knowledge base of past analyses, allowing it to learn from experience.
*   **Production Deployment**: Building in the validation, monitoring, and compliance checks needed for real-world use.

The goal was to turn vision into reality. Now that we have the building blocks, the possibilities are truly exciting.

### Try It Yourself

I highly encourage you to [download the Jupyter notebook](https://github.com/AriAnthony/PharmAI/blob/main/getting_started/pharmacometric_ai_demo.ipynb) and run it yourself. Experiment by changing the `task_details`. The beauty of this framework is its adaptability—it can scale from basic NCA to complex population PK models.

If you want to learn more about agentic workflows, there are many excellent examples to learn from [here](https://github.com/langchain-ai/langgraph/tree/main/docs/docs/how-tos) and excellent video tutorials by [@RLanceMartin](https://x.com/RLanceMartin).


What pharmacometric tasks would you want to automate first? Share your thoughts in the comments or reach out on [LinkedIn](https://www.linkedin.com/in/aripb).

---
*This post was developed with assistance from Claude AI. All code examples are provided as educational material and should be thoroughly tested before production use. Views expressed are my own and do not represent my employer.*