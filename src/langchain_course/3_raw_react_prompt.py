import re
import inspect
from dotenv import load_dotenv
from langsmith import traceable
import ollama

load_dotenv()

MAX_ITERATIONS = 10
MODEL="qwen3:1.7b"

# --- Tools(LangChain @tool decorator) ---

@traceable(run_type="tool")
def get_product_price(product: str) -> float:
    """ Look up the price of product in catalog. """
    print(f">>>Executing get_product_price(product='{product}')")
    prices ={"laptop":1299.99, "headphones": 149.95, "keyboard": 89.50, "phone":999.99}
    return prices.get(product, 0)

@traceable(run_type="tool")
def apply_discount(price:float, tier: str) -> float:
    """ Apply discount tier to product price and return the final price
        Available tiers include: bronze, silver and gold. """
    print(f">>>Executing apply_discount(price='{price}', tier='{tier}')")
    price = float(price)
    discount_percentages = {"bronze":5, "silver":10, "gold":23}
    discount = discount_percentages.get(tier,0)
    return round(price*(1-discount/100), 2)


tools = {
    "get_product_price": get_product_price,
    "apply_discount": apply_discount
}

# CHANGE 3: Delete the JSON schemas. Tool now live inside the prompt as plain text.
# We derive descriptions from the functions themselves using inspect.

def get_tool_description(tools_dict):
    descriptions=[]

    for tool_name, tool_function in tools_dict.items():
        # Wrapped bypasses decorator wrappers(e.g. @traceable adds *, config=None)
        original_function = getattr(tool_function, "__wrapped__", tool_function)
        signature = inspect.signature(original_function)
        docstring = inspect.getdoc(tool_function) or ""
        descriptions.append(f"{tool_name}{signature}-{docstring}")
    return "\n".join(descriptions)


tool_descriptions = get_tool_description(tools)
tool_names = ", ".join(tools.keys())

react_prompt = f""""
STRICT RULES - You must follow these exactly: \n
1. NEVER guess or assume any product price. 
You MUST call get_product_price first to get the real price. \n
2. Only call apply_discount AFTER you have received a price 
 from get_product_price. Pass the exact price returned 
 by get_product_price - do NOT pass a made-up number.\n
3. NEVER calculate discounts yourself using math 
ALWAYS use apply_discount tool. \n
4. If the user doesn't specify a discount_tier,
ask them which tier to use - do NOT assume one.

Answer the following questions as best you can. You have access to the following tools:

{tool_descriptions}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {{question}}
Thought:"""


@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat_traced(model, messages, options):
    return ollama.chat(model=model, messages=messages, options=options)

# --- Agent Loops ---

@traceable(name="Ollama Agent Loop")
def run_agent(question:str):

    print(f"Question: {question}")
    print("="*70)

    #CHANGE 5: One prompt string replaces the system/user message split.
    prompt = react_prompt.format(question=question)
    scratchpad = ""

    

    for iteration in range(1, MAX_ITERATIONS+1):
        print(f"\n---- Iteration {iteration} -----")
        full_prompt = prompt + scratchpad

        #Difference 5: ollama.chat() directly instead of llm_with_tools.invoke()
        response = ollama_chat_traced(
            model=MODEL,
            messages=[{"role":"user", "content":full_prompt}],
            options = {"stop":["\nObservation"], "temperature":0})
        output = response.message.content
        print(f"LLM Output:\n{output}")

        print(f" Looking for Final Answer in LLM Output....")
        final_answer_match = re.search(r"Final Answer:\s*(.+)", output)
        if final_answer_match:
            final_answer = final_answer_match.group(1).strip()
            print(f"[Parsed] Final Answer: {final_answer}")
            print("\n"+"="*60)
            print(f"Final Answer: {final_answer}")
            return final_answer

        #CHANGE 6: Parse tool calls from raw text with regex - fragile if LLM doesnt follow format.
        print(f" [Parsing] Looking for action and action input in LLM output..")

        action_match = re.search(r"Action:\s*(.+)", output)
        action_input_match = re.search(r"Action Input:\s*(.+)", output)

        if not action_match or not action_input_match:
            print(
                "   [Parsing] ERROR: Could not parse Action/Action Input from LLM Output."
            )
            break

        tool_name = action_match.group(1).strip()
        tool_input_raw = action_input_match.group(1).strip()

        print(f"[Tool Selected] {tool_name} with args: {tool_input_raw}")


        # Split comma-seperated args; strip key= prefix if LLM outputs key=value format
        raw_args = [x.strip() for x in tool_input_raw.split(",")]
        args = [x.split("=",1)[-1].strip().strip("'\'") for x in raw_args]

        print(f"[Tool Executing] {tool_name}(args)....")

        if tool_name not in tools:
            observation = f"Error: Tool '{tool_name} not found. Available Tools: {list[str](tools.keys())}"
        else:
            observation = str(tools[tool_name](*args))

        print(f"    [Tool Result] {observation}")

        # Change 7: History is one growing string resent every iteration (replaces messages.append)
        scratchpad += f"{output}\nObservation: {observation}\nThought:"

    print("ERROR: Max iterations reached without a final answer")
    return None


if __name__== "__main__":
    print("Hello Langchain Agenet (.bind_tools)!")
    print()
    result=run_agent("What is the price of laptop after applying a gold discount?")