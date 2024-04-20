import langroid as lr
import langroid.language_models as lm
from dotenv import load_dotenv
from langroid.agent.tools import RecipientTool
from langroid.parsing import code_parser

from textwrap import dedent

# set up LLM
llm_1_cfg = lm.OpenAIGPTConfig(  # or OpenAIAssistant to use Assistant API
    # any model served via an OpenAI-compatible API
    # chat_model=lm.OpenAIChatModel.GPT3_5_TURBO,  # or, e.g., "ollama/mistral"
    # chat_model="ollama/llama3:8b-instruct-fp16",
    chat_model="ollama/command-r",
    chat_context_length=128_000,
    # use_chat_for_completion=True,
)

# set up LLM
llm_2_cfg = lm.OpenAIGPTConfig(  # or OpenAIAssistant to use Assistant API
    # any model served via an OpenAI-compatible API
    # chat_model=lm.OpenAIChatModel.GPT3_5_TURBO,  # or, e.g., "ollama/mistral"
    # chat_model="ollama/llama3:8b-instruct-fp16",
    chat_model="ollama/command-r",
    chat_context_length=128_000,
    # use_chat_for_completion=True,
)

ASSESSOR_NAME = "Ashim"
ORG_REP_NAME = "Devin"


# use LLM in an Agent
org_agent_cfg = lr.ChatAgentConfig(
    name=ORG_REP_NAME,
    llm=llm_1_cfg,
    # use_tools=True,
    # use_functions_api=True,
    # parsing=code_parser.CodeParsingConfig(),
    vecdb=None,
)
# org_agent.enable_message(RecipientTool.create(recipients=["ASS_BOT"]))

#
cmmc_agent_cfg = lr.ChatAgentConfig(
    name=ASSESSOR_NAME,
    llm=llm_2_cfg,
    # use_tools=True,
    # use_functions_api=True,
    # parsing=code_parser.CodeParsingConfig(),
    vecdb=None,
)

org_agent = lr.ChatAgent(org_agent_cfg)
cmmc_agent = lr.ChatAgent(cmmc_agent_cfg)


# org_agent.enable_message(RecipientTool.create(recipients=[ASSESSOR_NAME]))
# cmmc_agent.enable_message(RecipientTool.create(recipients=[ORG_REP_NAME]))


NO_ANSWER = lr.utils.constants.NO_ANSWER
DONE = lr.utils.constants.DONE
PASS = lr.utils.constants.PASS
PASS_TO = lr.utils.constants.PASS_TO
SEND_TO = lr.utils.constants.SEND_TO

ASSESSOR_SYSTEM = dedent(
    f"""
        You are a Cyber Security Maturity Model Certification (CMMC) Assessor called {ASSESSOR_NAME}. You are currently in conversation with {ORG_REP_NAME}.
        You will ask questions about the state of security practices in {ORG_REP_NAME}'s organization according to CMMC section provided.
        You will maintain the conversation flow and ask relevant questions based on the answers of previous questions.

        When you have finished with your assessment, reply with DONE.

        When asking the Multiplier, remember to only present your
        request in form of a question, do not add un-necessary phrases.
"""
)

ORG_REP_SYSTEM = dedent(
    f""" You are Devin. Your organization is being assessed based on CMMC.
        Instructions:
            1. Answer the questions from the assessor.
            2. Be concise and respectful.
"""
)


org_rep_task = lr.Task(
    org_agent,
    name="Devin",
    system_message=ORG_REP_SYSTEM,
    single_round=True,
    llm_delegate=False,
    interactive=False,
    done_if_response=[lr.Entity.LLM],
)

cmmc_assesor_task = lr.Task(
    cmmc_agent,
    name="Kevin",
    system_message=ASSESSOR_SYSTEM,
    single_round=False,
    interactive=False,
    llm_delegate=True,
    done_if_no_response=[lr.Entity.LLM],
)

cmmc_assesor_task.add_sub_task([org_rep_task])
cmmc_assesor_task.run()
