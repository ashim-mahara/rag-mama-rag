from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
import torch


class Assistant:
    """Gemma 2b based assistant that replies given the retrieved documents"""

    def __init__(self):

        self.nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        # self.tokenizer = AutoTokenizer.from_pretrained("Nexusflow/Starling-LM-7B-beta", token=access_token)
        # self.model = AutoModelForCausalLM.from_pretrained("Nexusflow/Starling-LM-7B-beta", token=access_token, quantization_config=self.nf4_config)

        self.tokenizer = AutoTokenizer.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.2", token=access_token
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.2", token=access_token
        )

    def create_prompt(self, query, retrieved_info):
        # instruction to areply to query given the retrived information
        # prompt = f"""You are helpful Cyber Security Assistant Chatbot. You need either to explain the concept or answer the question about Cyber Security.
        # Be detailed, use simple words and examples in your explanations. If required, utilize the relevant information.
        # Instruction: {query}
        # Relevant information: {retrieved_info}
        # Output:
        # """

        # prompt = f"""You are helpful Cyber Security Assistant Chatbot. You need either to explain the concept or answer the question about Cyber Security.
        # # Be detailed, use simple words and examples in your explanations. If required, utilize the relevant information. GPT4 Correct User: {query} Additional Information: {retrieved_info}<|end_of_turn|>GPT4 Correct Assistant:"""

        ## Mistral Prompt
        prompt = f""" ### [INST]
        Instruction: You are helpful Cyber Security Assistant Chatbot. You need either to explain the concept or answer the question about Cyber Security.
        Be detailed, use simple words and examples in your explanations but do not miss important technical details. If required, utilize the relevant information. Here is context to help: {retrieved_info}

        ### QUESTION: {query}
        [/INST]
        """

        return prompt

    def reply(self, query, retrieved_info):
		"""Generate text for the prompt

		Args:
			query (String): The question to be asked to the Assistant.
			retrieved_info (_type_): _description_

		Returns:
			_type_: _description_
		"""

        prompt = self.create_prompt(query, retrieved_info)
        input_ids = self.tokenizer(query, return_tensors="pt").input_ids.to(device)

        # Generate text with a focus on factual responses

        generated_text = self.model.generate(
            input_ids,
            max_length=2048,  # let answers be not that long
            temperature=0.9,  # Adjust temperature according to the task, for code generation it can be 0.9
            do_sample=True,
        )

        # Decode and return the answer
        answer = self.tokenizer.decode(generated_text[0], skip_special_tokens=True)

        return answer
