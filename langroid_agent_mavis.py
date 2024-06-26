from dotenv import load_dotenv

load_dotenv()

import pickle
import langroid as lr
import langroid.language_models as lm
from langroid.agent.tools import RecipientTool
from langroid.parsing import code_parser
from langroid.agent.special import DocChatAgent, DocChatAgentConfig
from langroid.parsing.parser import ParsingConfig, PdfParsingConfig, Splitter
from textwrap import dedent
import glob
import pandas as pd
import json

from langroid.mytypes import Document, DocMetaData

lr.utils.logging.setup_colored_logging()


local_llm = "ollama/command-r:35b-v0.1-q2_K"
# set up LLM
llm_1_cfg = lm.OpenAIGPTConfig(  # or OpenAIAssistant to use Assistant API
    # any model served via an OpenAI-compatible API
    chat_model=lm.OpenAIChatModel.GPT4,  # or, e.g., "ollama/mistral"
    # chat_model="ollama/llama3:8b-instruct-fp16",
    # chat_model=local_llm,
    # chat_context_length=128_000,
    chat_context_length=16_000,
    # use_chat_for_completion=True,
)

llm_2_cfg = lm.OpenAIGPTConfig(  # or OpenAIAssistant to use Assistant API
    # any model served via an OpenAI-compatible API
    chat_model=lm.OpenAIChatModel.GPT4,  # or, e.g., "ollama/mistral"
    # chat_model="ollama/llama3:8b-instruct-fp16",
    # chat_model=local_llm,
    # chat_context_length=128_000,
    chat_context_length=16_000,
    # use_chat_for_completion=True,
)


vector_db = lr.vector_store.QdrantDBConfig(
    collection_name="mavis_shop",
    replace_collection=True,
)

mavis_doc_agent_config = DocChatAgentConfig(
    llm=lr.language_models.OpenAIGPTConfig(
        chat_model=lm.OpenAIChatModel.GPT4_TURBO,
        # chat_model=local_llm,
        # max_output_tokens=8192,
        max_output_tokens=4096,
        # chat_context_length=128_000,
        chat_context_length=16_000,
    ),
    use_tools=True,
    use_functions_api=True,
    name="DocAgent",
    vecdb=vector_db,
    parsing=lr.parsing.parser.ParsingConfig(
        separators=["\n\n"],
        splitter=lr.parsing.parser.Splitter.PARA_SENTENCE,
        n_similar_docs=10,
        pdf=PdfParsingConfig(
            # alternatives: "unstructured", "pdfplumber", "fitz"
            library="pdfplumber",
        ),
    ),
)

doc_agent = DocChatAgent(mavis_doc_agent_config)


vector_db_cmmc = lr.vector_store.QdrantDBConfig(
    collection_name="cmmc_assessment",
    replace_collection=True,
)

cmmc_doc_agent_config = DocChatAgentConfig(
    llm=lr.language_models.OpenAIGPTConfig(
        chat_model=lm.OpenAIChatModel.GPT4_TURBO,
        # chat_model=local_llm,
        # max_output_tokens=8192,
        max_output_tokens=4096,
        # chat_context_length=128_000,
        chat_context_length=16_000,
    ),
    use_tools=True,
    use_functions_api=True,
    name="AssessmentDocs",
    vecdb=vector_db_cmmc,
    parsing=lr.parsing.parser.ParsingConfig(
        separators=["\n\n"],
        splitter=lr.parsing.parser.Splitter.PARA_SENTENCE,
        n_similar_docs=10,
        pdf=PdfParsingConfig(
            # alternatives: "unstructured", "pdfplumber", "fitz"
            library="pdfplumber",
        ),
    ),
)

cmmc_doc_agent = DocChatAgent(cmmc_doc_agent_config)


mavis_agent_cfg = lr.ChatAgentConfig(
    name="Devin",
    llm=llm_1_cfg,
    use_tools=True,
    use_functions_api=True,
)

mavis_agent = lr.ChatAgent(mavis_agent_cfg)

#
cmmc_agent_cfg = lr.ChatAgentConfig(
    name="CMMC_ASSESSOR",
    llm=llm_2_cfg,
    use_tools=True,
    use_functions_api=True,
    # parsing=code_parser.CodeParsingConfig(),
    vecdb=None,
)

cmmc_agent = lr.ChatAgent(cmmc_agent_cfg)


NO_ANSWER = lr.utils.constants.NO_ANSWER
DONE = lr.utils.constants.DONE
PASS = lr.utils.constants.PASS
PASS_TO = lr.utils.constants.PASS_TO
SEND_TO = lr.utils.constants.SEND_TO

cmmc_df = pd.read_csv("./cmmc.csv", quotechar="|")
nist_sections = pd.read_csv("./nist_sections.csv", quotechar="|")

ASSESSOR_SYSTEM = dedent(
    f"""You are a Cyber Security Maturity Model Certification (CMMC) Assessor.

        You are going to assess the organizations based on the CMMC Document. However you do NOT have access to the documents.
        You will be assisted by AssessmentDocs, who DOES have access to the documents.
        The tool contains both CMMC Sections and NIST Handbook Sections. They do not contain information about Mavis Shop. Only Devin knows about Mavis Shop.

        The CMMC topics are are: {cmmc_df["topic"]}

        The NIST section headers are: {nist_sections["section_heading"]}

        Both the CMMC sections for the corresponding topic and NIST sections for the CMMC sections are available with AssessmentDocs. Create questions for each of the topics in CMMC sections.

        Once you have created some questions for Devin, you can ask Devin questions based on the content of the documents at Mavis Shop.
        Keep in mind that Devin does not know anything about CMMC and NIST. Devin only knows about their own organizations data so you need to ask the questions without CMMC and NIST references when sending a query to Devin.

        You can use the Recipient tool to send messages to the Devin. Only ask one question at a time.

        The recipient tool is called like:

        TOOL: recipient_message
        ```json
        {{
            "request": "recipient_message",
            "intended_recipient": "AssessmentDocs",
            "content": "What are the CMMC requirements for 'Access Control' and 'System and Information Integrity'?"```
        }}
        ```

        If asked to add recipient properly, you can do it like:

        TOOL: recipient_message
        ```json
        {{
            "request": "add_recipient",
            "intended_recipient": "AssessmentDocs"
        }}
        ```

        You will maintain the conversation flow and ask relevant questions based on the answers of previous questions.

        Provide a final output where you write the final verdict and the reasons why you went with the verdict.
        When you have finished with your assessment, reply with DONE.

        Here are the rules:
        (a) when the question is complex or has multiple parts, break it into small
         parts and/or steps and send them to AssessmentDocs
        (b) if AssessmentDocs says {NO_ANSWER} or gives no answer, try asking in other ways.
        (c) Once you collect all parts of the answer, say "DONE"
            and show me the consolidated final answer.
        (d) AssessmentDocs has no memory of previous dialog, so you must ensure your
            questions are stand-alone questions that don't refer to entities mentioned
            earlier in the dialog.
        (e) if AssessmentDocs is unable to answer after your best efforts, you can say
            {NO_ANSWER} and move on to the next question.
        (f) answers should be based ONLY on the documents, NOT on your prior knowledge.
        (g) be direct and concise, do not waste words being polite.
        (h) if you need more info from the user, before asking AssessmentDocs, you should
        address questions to the "User" (not to AssessmentDocs) to get further
        clarifications or information.
        (i) Always ask questions ONE BY ONE (to either User, Devin or AssessmentDocs), NEVER
            send Multiple questions in one message.
        (j) Use bullet-point format when presenting multiple pieces of info.
        (k) When AssessmentDocs responds without citing a SOURCE and EXTRACT(S), you should
            send your question again to DocChat, reminding it to cite the source and
            extract(s).

"""
)

ORG_REP_SYSTEM = dedent(
    f"""Your name is Devin. You are a system administrator for Mavis Shop.
        You are tenacious, creative and resourceful when given a question to
        find an answer for. You will receive questions from the CMMC Assessor, which you will
        answer ONLY based on content from certain documents (not from your
        general knowledge). However you do NOT have access to the documents.
        You will be assisted by DocAgent, who DOES have access to the documents.

        Example Tool Usage:
            The recipient tool can be called with:
            TOOL: recipient_message
            ```json
            {{
                "request": "recipient_message",
                "intended_recipient": "DocAgent",
                "content": "What is Mavis Shop's policy on Access Control?"
            ```
            }}
            ```

            If asked to add recipient properly by the Recipient Tool, you can reply with:

            TOOL: recipient_message
            ```json
            {{
                "request": "add_recipient",
                "intended_recipient": "DocAgent"
            }}
            ```

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
        (l) If DocAgent answers questions with {NO_ANSWER} multiple times then you can assume that Mavis Shop does not perform the the security procedure asked.
"""
)

docs = glob.glob("./mavis_shop/*.pdf")
doc_agent.ingest_doc_paths(docs)

cmmc_doc_agent.ingest_dataframe(
    cmmc_df, content="text", metadata=["section", "topic", "level"]
)

cmmc_doc_agent.ingest_dataframe(
    nist_sections, content="text", metadata=["section_heading"]
)


doc_task = lr.Task(
    doc_agent,
    name="DocAgent",
    interactive=False,
    single_round=True,
    llm_delegate=True,
    done_if_no_response=[lr.Entity.LLM],  # done if null response from LLM
    done_if_response=[lr.Entity.LLM],  # done if non-null response from LLM
    # system_message=ORG_REP_SYSTEM,
)


cmmc_nist_doc_task = lr.Task(
    cmmc_doc_agent,
    name="AssessmentDocs",
    interactive=False,
    single_round=True,
    llm_delegate=True,
    done_if_no_response=[lr.Entity.LLM],  # done if null response from LLM
    done_if_response=[lr.Entity.LLM],  # done if non-null response from LLM
    # system_message=ORG_REP_SYSTEM,
)

mavis_task = lr.Task(
    mavis_agent,
    interactive=False,
    name="Devin",
    llm_delegate=True,
    single_round=False,
    done_if_no_response=[lr.Entity.LLM],  # done if null response from LLM
    system_message=ORG_REP_SYSTEM,
)

cmmc_assesor_task = lr.Task(
    cmmc_agent,
    name="CMMC_ASSESSOR",
    system_message=ASSESSOR_SYSTEM,
    single_round=False,
    interactive=False,
    llm_delegate=True,
    done_if_no_response=[lr.Entity.LLM],
)

# doc_agent.enable_message(RecipientTool)
mavis_agent.enable_message(RecipientTool.create(["DocAgent"]))
cmmc_agent.enable_message(RecipientTool.create(["AssessmentDocs", "Devin"]))

history = cmmc_agent.message_history
print("Assessor History", history)

mavis_task.add_sub_task([doc_task])
cmmc_assesor_task.add_sub_task([cmmc_nist_doc_task, mavis_task])
cmmc_assesor_task.run()

history = cmmc_agent.message_history

pickled_str = pickle.dumps(history)
