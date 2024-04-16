import langroid as lr
import langroid.language_models as lm

# set up LLM
llm_cfg = lm.OpenAIGPTConfig(  # or OpenAIAssistant to use Assistant API
    # any model served via an OpenAI-compatible API
    # chat_model="lm.OpenAIChatModel.GPT4_TURBO",  # or, e.g., "ollama/mistral"
    chat_model="ollama/mistral"
)
# use LLM directly
mdl = lm.OpenAIGPT(llm_cfg)
response = mdl.chat("What is the capital of Ontario?", max_tokens=10)

# use LLM in an Agent
agent_cfg = lr.ChatAgentConfig(llm=llm_cfg)
agent = lr.ChatAgent(agent_cfg)
agent.llm_response("What is the capital of China?")
response = agent.llm_response("And India?")  # maintains conversation state

SYSTEM = """You are an expert Cyber Security Professional Assistant."""

# wrap Agent in a Task to run interactive loop with user (or other agents)
task = lr.Task(agent, name="Bot", system_message=SYSTEM)
task.run("Hello")  # kick off with user saying "Hello"
