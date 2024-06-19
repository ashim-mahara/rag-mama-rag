import langroid as lr
import langroid.language_models as lm
from textwrap import dedent
from langroid.agent.special import DocChatAgent, DocChatAgentConfig
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


agent_config = lm.OpenAIGPTConfig(  # or OpenAIAssistant to use Assistant API
    # any model served via an OpenAI-compatible API
    chat_model=lm.OpenAIChatModel.GPT4_TURBO,  # or, e.g., "ollama/mistral"
    # chat_model="ollama/llama3:8b-instruct-fp16",
    # chat_model="ollama/command-r",
    chat_context_length=128_000,
    use_chat_for_completion=True,
)

# use LLM in an Agent
agent_cfg = lr.ChatAgentConfig(
    name="GPT4",
    llm=agent_config,
    # use_tools=True,
    # use_functions_api=True,
    # parsing=code_parser.CodeParsingConfig(),
    vecdb=None,
)
agent = lr.ChatAgent(agent_cfg)

# mdl = lm.OpenAIGPT(agent_cfg)


def format_prompts(prompts):
    """
    Formats each prompt dictionary into a structured f-string.

    Args:
    prompts (list of dict): A list of dictionaries, where each dictionary contains
                            'reference', 'query', and 'context' keys.

    Returns:
    list of str: A list containing formatted strings.
    """
    formatted_strings = []
    for prompt in prompts:
        reference = prompt["reference"]
        query = prompt["query"]
        context = prompt["context"]
        formatted_string = (
            f"Reference: {reference}\nQuery: {query}\nContext: {context}\n"
        )
        formatted_strings.append(formatted_string)

    return formatted_strings


ORG_REP_SYSTEM = dedent(
    """Hello, I am a cybersecurity student learning about AI models, we are trying to create prompts for a model specifically for cybersecurity professionals. Can you please look at the attached and generate prompts in a .json file for our RAG project?

    Context: {context}

    Reference: {context}

    Query: {}




    """
)

docs = pd.read_csv("./cmmc.csv", quotechar="|")

documents = [
    f"""Section: {doc["section"]} , Level: {doc["level"]}, Document: {doc["text"]}"""
    for idx, doc in docs.iterrows()
]

responses = []

prompts = format_prompts()

for prompt in documents:
    print("Prompt: ", prompt)
    response = agent.llm_response(prompt)
    responses.append(response)
    print(response)
