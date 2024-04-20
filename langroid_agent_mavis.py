import langroid as lr
import langroid.language_models as lm
from dotenv import load_dotenv
from langroid.agent.tools import RecipientTool
from langroid.parsing import code_parser
from langroid.agent.special import DocChatAgent, DocChatAgentConfig
from langroid.parsing.parser import ParsingConfig, PdfParsingConfig, Splitter


from textwrap import dedent

import glob

# set up LLM
llm_1_cfg = lm.OpenAIGPTConfig(  # or OpenAIAssistant to use Assistant API
    # any model served via an OpenAI-compatible API
    chat_model=lm.OpenAIChatModel.GPT4,  # or, e.g., "ollama/mistral"
    # chat_model="ollama/llama3:8b-instruct-fp16",
    #    chat_model="ollama/command-r",
    chat_context_length=8192,
    # use_chat_for_completion=True,
)

config = DocChatAgentConfig(
    llm=lr.language_models.OpenAIGPTConfig(
        chat_model=lr.language_models.OpenAIChatModel.GPT4, max_output_tokens=1024
    ),
    use_tools=True,
    vecdb=lr.vector_store.QdrantDBConfig(
        collection_name="cmmc_assessor",
        replace_collection=True,
    ),
    parsing=lr.parsing.parser.ParsingConfig(
        separators=["\n\n"],
        splitter=lr.parsing.parser.Splitter.SIMPLE,
        n_similar_docs=2,
        pdf=PdfParsingConfig(
            # alternatives: "unstructured", "pdfplumber", "fitz"
            library="pdfplumber",
        ),
    ),
)
mavis_agent = DocChatAgent(config)


#
cmmc_agent_cfg = lr.ChatAgentConfig(
    name="CMMC_ASSESSOR",
    llm=llm_1_cfg,
    use_tools=True,
    # use_functions_api=True,
    # parsing=code_parser.CodeParsingConfig(),
    vecdb=None,
)

cmmc_agent = lr.ChatAgent(cmmc_agent_cfg)


mavis_agent.enable_message(RecipientTool)
cmmc_agent.enable_message(RecipientTool.create(["DocAgent"]))


NO_ANSWER = lr.utils.constants.NO_ANSWER
DONE = lr.utils.constants.DONE
PASS = lr.utils.constants.PASS
PASS_TO = lr.utils.constants.PASS_TO
SEND_TO = lr.utils.constants.SEND_TO

ASSESSOR_SYSTEM = dedent(
    f"""l
        You are a Cyber Security Maturity Model Certification (CMMC) Assessor. You are talking with a system admin for Mavis Shop.
        You will ask questions about the state of security practices in Mavis Shop according to CMMC.
        You will maintain the conversation flow and ask relevant questions based on the answers of previous questions.

        You can use the Recipient tool to send messages to the DocAgent.

        Provide a final output where you write the final verdict and the reasons why you went with the verdict.
        When you have finished with your assessment, reply with DONE.
"""
)

ORG_REP_SYSTEM = dedent(
    f"""You are a system administrator for Mavis Shop.
        You are tenacious, creative and resourceful when given a question to
        find an answer for. You will receive questions from the CMMC Assessor, which you will
        try to answer ONLY based on content from certain documents (not from your
        general knowledge). However you do NOT have access to the documents.
        You will be assisted by DocAgent, who DOES have access to the documents.

        Here are the rules:
        (a) when the question is complex or has multiple parts, break it into small
         parts and/or steps and send them to DocAgent
        (b) if DocAgent says {NO_ANSWER} or gives no answer, try asking in other ways.
        (c) Once you collect all parts of the answer, say "DONE"
            and show me the consolidated final answer.
        (d) DocAgent has no memory of previous dialog, so you must ensure your
            questions are stand-alone questions that don't refer to entities mentioned
            earlier in the dialog.
        (e) if DocAgent is unable to answer after your best efforts, you can say
            {NO_ANSWER} and move on to the next question.
        (f) answers should be based ONLY on the documents, NOT on your prior knowledge.
        (g) be direct and concise, do not waste words being polite.
        (h) if you need more info from the user, before asking DocAgent, you should
        address questions to the "User" (not to DocAgent) to get further
        clarifications or information.
        (i) Always ask questions ONE BY ONE (to either User or DocAgent), NEVER
            send Multiple questions in one message.
        (j) Use bullet-point format when presenting multiple pieces of info.
        (k) When DocAgent responds without citing a SOURCE and EXTRACT(S), you should
            send your question again to DocChat, reminding it to cite the source and
            extract(s).
"""
)

docs = glob.glob("./mavis_shop/*.pdf")
mavis_agent.ingest_doc_paths(docs)

doc_task = lr.Task(
    mavis_agent,
    interactive=False,
    name="DocAgent",
    done_if_no_response=[lr.Entity.LLM],  # done if null response from LLM
    done_if_response=[lr.Entity.LLM],  # done if non-null response from LLM
    system_message=ORG_REP_SYSTEM,
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

cmmc_assesor_task.add_sub_task([doc_task])
cmmc_assesor_task.run()
