from vllm import LLM, SamplingParams
import time
import csv
import torch

domains = []

with open("majestic_million_new.csv", newline="", encoding="utf-8") as csv_file:
    reader = csv.DictReader(csv_file, delimiter=",")
    for row in reader:
        domains.append(row["Domain"])

INITIAL_PROMPT = "Determine if the website $$domain$$ is focused on cybersecurity, including whether it functions as a blog or a news outlet within the cybersecurity niche. Your response should select only one from the following options: 'Yes', 'No', or 'Not Sure'. Alongside your selection, provide a confidence level to your assessment, ranging from 0 to 100, where 0 is the least confident and 100 is the most confident. Format your response as a JSON object in the structure: {result: 'Your_Answer', score: Your_Confidence_Level}. Exclude any explanations for your choice."


prompts = [INITIAL_PROMPT.replace("$$domain$$", domain) for domain in domains]

sampling_params = SamplingParams(temperature=0.1, top_p=0.95, max_tokens=50)

llm = LLM(model="TheBloke/Mistral-7B-Instruct-v0.2-AWQ")

start_time = time.time()

outputs = llm.generate(prompts, sampling_params)

saved_outputs = []


def format_output(domain, output):

    formatted = {
        "domain": domain,
        "text": output.outputs[0].text,
        "finished": output.finished,
    }
    return formatted


# Print the outputs.
# for output in outputs:
for i in range(len(outputs)):
    output = outputs[i]
    domain = domains[i]
    prompt = output.prompt
    generated_text = output.outputs[0].text
    # print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    formatted_output = format_output(domain, output)
    print(formatted_output)
    saved_outputs.append(formatted_output)


with open(
    "domain_predictions_1_too_confidence_score.csv", "w", newline="", encoding="utf-8"
) as f:
    TITLE = "domain,text,finished".split(",")
    csv_writer = csv.DictWriter(
        f, TITLE, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL
    )

    csv_writer.writeheader()
    csv_writer.writerows(saved_outputs)


print("--- %s seconds ---" % (time.time() - start_time))
