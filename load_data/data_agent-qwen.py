# import parser
import sys
import os
import re
import json
import time
import logging

# Appends the parent directory to the system path to allow importing local modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from load_data.preprocess import * 

# fill in your openai api key

from tqdm import tqdm
import argparse
import random
#




# Set up excessive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler("pipeline_execution.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# setup local qwen
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configuration for Local Qwen
MODEL_NAME = "Qwen/Qwen3.5-27B" # Change to your specific Qwen version if needed
CACHE_DIR = os.path.expanduser("~/models/")

logger.info(f"Loading {MODEL_NAME} from cache...")

# 1. Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    cache_dir=CACHE_DIR,
    trust_remote_code=True
)

# 2. Load Model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    cache_dir=CACHE_DIR,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True
)
logger.info("✓ Model and tokenizer loaded successfully.\n")



# get_response = api calling
# format_example = format the question and options into a prompt template for the LLM
# extract_knowledge_based = extract the factual queries from the LLM's reasoning plan using regex
# clean_and_parse_json_string_with_codeblock = cleans up the JSON string returning a dict
# planning = generates a prompt template asking the LLM to create a step-by-step reasoning plan
# NER_agent = prompts the LLM to provide concrete factual knowledge for specific queries
# get_label = process the text 


def format_example(question, options, cot_content=""):
    """
    Format the question, multiple-choice options, and CoT content into a structured prompt string.
    
    Args:
        question (str): The main question text.
        options (list): A list of string options for the multiple-choice question.
        cot_content (str): Optional existing chain-of-thought content.
        
    Returns:
        str: A formatted string ready to be fed into the language model.
    """
    if cot_content == "":
        cot_content = ""
    # Remove "A: " prefix if the CoT content already includes it to avoid duplication
    if cot_content.startswith("A: "):
        cot_content = cot_content[3:]
        
    example = "Question: {}\nOptions: ".format(question)
    choice_map = "ABCDEFGHIJ" # Maps index to letter choices (A, B, C...)
    
    # Append each option with its corresponding letter
    for i, opt in enumerate(options):
        example += "{}. {}\n".format(choice_map[i], opt)
        
    # Append the answer section, optionally with the provided CoT reasoning
    if cot_content == "":
        example += "Answer: "
    else:
        example += "Answer: " + cot_content + "\n\n"
    return example


def extract_labeled_content_as_list(input_string):
    """
    Parses the generated reasoning plan string to extract labeled steps.
    
    Args:
        input_string (str): The raw output from the LLM containing step names, requirements, and content.
        
    Returns:
        list: A list of formatted strings, e.g., ["[reason]: <content>", "[rag]: <content>"].
    """
    logger.info("[TAG EXTRACTOR] Starting extraction of [reason] and [rag] tags.")
    if "</think>" in input_string:
        input_string = input_string.split("</think>")[-1]
    
    # Split the string by the specific delimiter defined in the prompt
    steps = re.split(r'\*Step name\*:', input_string)

    labeled_content = []
    for step in steps:
        if step.strip():
            # # Extract the requirement type (e.g., 'reason' or 'rag')
            # requirement_match = re.search(r'\*\*Requirement\*\*: \[(.*)\]', step)
            # if requirement_match:
            #     requirement = f"[{requirement_match.group(1)}]"

            # # Extract the actual content/reasoning of the step
            # content_match = re.search(r'\*\*Content\*\*: (.*)', step)
            # if content_match:
            #     content = content_match.group(1).strip()

            #     # Combine the requirement tag and content into a single string
            #     combined_step = f"{requirement}: {content}"
            #     labeled_content.append(combined_step)
            #     logger.info(f"[TAG EXTRACTOR] Assembled step: {combined_step[:100]}...")

            # adjusted to lane breaks 
            requirement_match = re.search(r'\*\*Requirement\*\*:\s*\[(.*?)\]', step)
            
            # Use re.DOTALL so (.*) captures everything, including newlines, until the end of the step
            content_match = re.search(r'\*\*Content\*\*:\s*(.*)', step, re.DOTALL)
            
            # Ensure BOTH were successfully found before trying to combine them
            if requirement_match and content_match:
                requirement = f"[{requirement_match.group(1).strip()}]"
                content = content_match.group(1).strip()
                
                combined_step = f"{requirement}: {content}"
                labeled_content.append(combined_step)
                logger.info(f"[TAG EXTRACTOR] Assembled step: {combined_step[:100]}...")
            else:
                logger.warning("[TAG EXTRACTOR] Failed to parse requirement or content in step.")



    logger.info(f"[TAG EXTRACTOR] Finished extracting {len(labeled_content)} steps.")
    return labeled_content


def get_response(prompt, model_name=None):
    """
    Sends a prompt to the local Qwen model and retrieves the response.
    (model_name argument is kept for compatibility with older function calls but ignored)
    """
    messages = [{"role": "user", "content": prompt}]
    
    logger.info(f"\n{'='*50}\n[LLM REQUEST START] Model: {MODEL_NAME}")
    logger.info(f"[LLM INPUT PROMPT]:\n{prompt}\n{'-'*50}")
    
    start_time = time.time()

    # Apply the model's specific chat format
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize inputs and move to the device the model is on
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # Generate output
    # NOTE: max_new_tokens is set to 2048 because planning and JSON outputs can be long.
    generated_ids = model.generate(
        **model_inputs,
        # max_new_tokens=4096, 
        max_new_tokens = 8192,  
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

    # Isolate the generated text from the prompt
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    # Decode the response
    content = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    end_time = time.time()
    elapsed_time = end_time - start_time
    
    logger.info(f"[LLM OUTPUT]:\n{content}\n{'-'*50}")
    logger.info(f"[LLM PERFORMANCE] Time taken: {elapsed_time:.2f} seconds")
    logger.info(f"[{MODEL_NAME} REQUEST END]\n{'='*50}\n")
    
    return content

def extract_knowledge_based(text):
    """
    Extracts the 'Knowledge based' sections from the LLM's planning output using regex.
    Args:
        text (str): The text containing the reasoning plan.
        
    Returns:
        list: A list of string segments representing the knowledge queries required for the steps.
    """
    if "</think>" in text:
        text = text.split("</think>")[-1]
    # if "[FINAL ANSWER]:" in text:
    #     text = text.split("[FINAL ANSWER]:")[-1]
    # logger.info(f"extract_knowledge_based(): [QUERY EXTRACTOR] Extracting knowledge-based queries from text:\n{text[:200]}...")
    # Uses a non-greedy dotall match to capture text between 'Knowledge based:' and 'Content'
    pattern = r"\*\*Knowledge based\*\*:(.*?)\*\*Content\*\*"
    matches = re.findall(pattern, text, re.DOTALL)  # dotail means accept newlines in the match as well 

    # Clean up whitespace and return
    return [match.strip() for match in matches]


def clean_and_parse_json_string_with_codeblock(json_str):
    """
    Cleans up a potentially malformed JSON string returned by the LLM and parses it into a dictionary.
    
    Args:
        json_str (str): The raw string containing the JSON.
        
    Returns:
        dict: The parsed JSON data.
    """
    logger.info("[JSON PARSER] Raw JSON input received.")

    # Remove the model's internal thinking process
    if "</think>" in json_str:
        json_str = json_str.split("</think>")[-1]

    # # --- NEW: Extract only what comes after the delimiter ---
    #     if "[FINAL ANSWER]:" in json_str:
    #         json_str = json_str.split("[FINAL ANSWER]:")[-1]
        
    #     logger.info(f"Extracted JSON portion:\n{json_str[:200]}...")  # Log the first 200 characters of the extracted JSON string

    # Remove markdown codeblock formatting
    json_str = json_str.replace('```json', '').replace('```', '')

    # Remove empty string values and trailing commas before braces/brackets
    json_str = re.sub(r'"[^"]*"\s*:\s*""\s*,?', '', json_str)
    json_str = re.sub(r',\s*}', '}', json_str)  
    json_str = re.sub(r',\s*]', ']', json_str)  

    # Strip whitespace and trailing commas
    json_str = json_str.strip().strip(',')

    # Check for structural integrity (balanced brackets/braces)
    if json_str.count('{') != json_str.count('}') or json_str.count('[') != json_str.count(']'):
        logger.error("[JSON PARSER ERROR] Unbalanced braces or brackets detected.")
        raise ValueError("The JSON string has unbalanced braces or brackets.")

    
    logger.info(f"Cleaned JSON String:\n{json_str[:200]}...") 

    # Parse the cleaned string into a Python dictionary
    parsed_dict = json.loads(json_str)
    logger.info(f"[JSON PARSER] Successfully parsed dictionary with {len(parsed_dict)} elements.")

    return parsed_dict



def planning(question, answer):
    """
    Generates a prompt template asking the LLM to create a step-by-step reasoning plan.
    
    Args:
        question (str): The question being asked.
        answer (str): The correct answer to guide the reasoning.
        
    Returns:
        str: The constructed prompt.
    """
    
    # -----------------prompt template example start ----------------
    #                                                               |
    template = f"""
        Here is the question:
        <Question>
        {question}
        <\\Question>

        Here is the correct answer:
        <Correct Answer>
        {answer}
        <\\Correct Answer>

        Factual knowledge is information that aligns with objective reality and can be verified through evidence or observation, such as scientific facts or historical events.

        provide a reasoning planning for above question to get correct answer, each step in your reasoning plan must be adhere strictly to the following format:

        *Step name*: 
        # put the name of step here.
        **Requirement**: 
        # If this step needs reasoning, return "[reason]" as label, if this step needs factual knowledge return "[rag]" as label.
        **Knowledge based**:  
        # Only if this step needs factual knowledge, put a query in question sentences about this factual knowledge for retrieval.
        **Content**: 
        # If this step is about reasoning, please provide your reasoning thinking, if this step needs factual knowledge please provide factual knowledge.

    """
    return template
    #                                                              |
    # -----------------prompt template example end -----------------


def NER_agent(questions):
    """
    Prompts the LLM to provide concrete factual knowledge for specific queries.
    
    Args:
        questions (list/str): A list or string of queries needing factual answers.
        model_name (str): The model to use.
        
    Returns:
        str: A JSON-formatted string containing the queries as keys and factual knowledge as values.
    """
    # ------------------prompt template example start -----------------
    #                                                                 |
    template = f"""
        Factual knowledge is information that aligns with objective reality and can be verified through evidence or observation, such as scientific facts or historical events.

        Please provide factual knowledge for below question set:
        <Questions>
            {questions}
        <\\Questions>

        You should return a dic in json format, for each element in dic, the key is each question in <Questions>, the value is the Factual knowledge of each question in <Questions>.
        Your answer format should strictly be in following steps:
        ```json
        {{
            "question 1": "The factual knowledge of question 1",
        ....
        }}
    """
        #                                                               |
        # ------------------prompt template example end -----------------

    # text = get_response(model_name, template)
    text = get_response(template)
    return text



def get_label(input_string, answer):
    """
    Orchestrates the creation of the reasoning plan and fills in factual knowledge where needed.

    Args:
        input_string (str): The main question.
        answer (str): The answer.
        
    Returns:
        list: A list of labeled steps containing either reasoning or factual knowledge (RAG).
    """
    # 1. Ask the LLM to generate the overall reasoning plan
    logger.info(f"\n>>> [get_label START] Processing Question: {input_string[:100]}...")
    prompt_template = planning(input_string, answer)    # generates a prompt template based on the question and answer to guide the LLM in creating a reasoning plan.
    llm_response = get_response(prompt_template)   # Sends a prompt to the OpenAI API and retrieves the response.

    # 2. Extract the factual queries, ask the NER agent to answer them, and parse the JSON result
    logger.info(">>> STEP 2: Extracting knowledge-based queries from the plan...")
    query_questions = extract_knowledge_based(llm_response)
    logger.info(f">>> Extracted Queries: {query_questions}")
    
    logger.info(">>> STEP 3: Calling NER Agent to fetch factual knowledge...")
    formatted_query_response = NER_agent(query_questions)
    
    logger.info(">>> STEP 4: Parsing NER Agent JSON response...")
    parsed_dict = clean_and_parse_json_string_with_codeblock(formatted_query_response)

    # 3. Replace the query sections in the original plan with the actual factual knowledge retrieved
    # currently in the query the key is the question sentence, and the value is the factual knowledge, 
    # we replace the question sentence with the factual knowledge, and add a label [rag] to indicate this is retrieved factual knowledge.
    logger.info(">>> STEP 5: Replacing queries in the plan with retrieved facts and injecting [rag] tags...")
    for item in parsed_dict.keys():
        logger.info(f"    Replacing query: '{item}' with factual knowledge.")
        llm_response = llm_response.replace(item, item + " [rag]" + parsed_dict[item]) 

    # 4. Extract and format the finalized steps into a list
    logger.info(">>> STEP 6: Formatting final labeled content...")
    final_list = extract_labeled_content_as_list(llm_response)
    
    logger.info(f">>> [get_label END] Successfully generated {len(final_list)} labeled steps.\n")
    return final_list


#-------------------generate dataset steps for StrategyQA----------------------

def generate_StrategyQA_agent(type):
    """
    Generates dataset steps for the StrategyQA dataset.

    Args:
        type (str): The split type (e.g., 'train', 'test').
    """
    data = load_dataset("ChilleD/StrategyQA")[type]
    json_file = "dataset_folder/StrategyQA_{}.json".format(type)
    
    # list existing data, if any exist, else start with empty list 
    processed_questions = set()
    if os.path.exists(json_file):
        with open(json_file, "r") as f:
            existing_data = json.load(f)
            processed_questions = {item["question"] for item in existing_data}
        print(f"✓ Loaded {len(existing_data)} previously processed examples")
        dict = existing_data  # Continue building on existing list
    else:
        dict = []

    
    for i, example in enumerate(tqdm(data, desc=f"Processing {type}")):
        question = example["question"]
        # Skip questions that have already been processed to avoid duplication
        if question in processed_questions:
            continue
            
        # Convert boolean answers to string
        answer = 'True' if example["answer"] else 'False'

        max_retries = 5
        retry_count = 0
        cot_steps = None

        # Retry mechanism in case the LLM fails to output the proper format or the API fails
        while retry_count < max_retries:
            try:
                cot_steps = get_label(question, answer)
                if(len(cot_steps) > 0):
                    print('success, aoligei')
                    break
                else:
                    retry_count += 1
                    # Note: 'e' is undefined here in the original code, it will throw a NameError if this block is reached
                    print(f"Error occurred while processing question: {question}. Attempt {retry_count} of {max_retries}. Error: {e}")
            except Exception as e:
                retry_count += 1
                print(f"Error occurred while processing question: {question}. Attempt {retry_count} of {max_retries}. Error: {e}")
        
        # if the maximum number of retries is reached, skip the question
        if retry_count == max_retries:
            print(f"Skipping question due to repeated errors: {question}")
            continue

        # Save successful result to dictionary
        new_entry = {
            "question": question,
            "answer": answer,
            "cot_steps": cot_steps,  
            "split": type
        }
        dict.append(new_entry)
        processed_questions.add(question) 
        
        # Incrementally dump to JSON file to avoid data loss on crash
        with open(json_file, "w") as f:
            json.dump(dict, f, indent=4)


#  -------------------generate dataset steps for MMLU-Pro----------------------

def generate_MMLU_pro_agent(split):
    """
    Generates dataset steps for the MMLU-Pro dataset.
    """
    type = split
    if type == "train":
        type = "validation"
    else:
        type = "test"
    print(type)
    data = load_dataset("TIGER-Lab/MMLU-Pro")[type]
    json_file = "dataset_folder/MMLU_Pro_{}.json".format(type)
    
    
    # LOAD EXISTING DATA IF FILE EXISTS
    processed_questions = set()
    if os.path.exists(json_file):
        with open(json_file, "r") as f:
            existing_data = json.load(f)
            # Use the formatted question string as the unique key
            processed_questions = {item["question"] for item in existing_data}
        print(f"✓ Loaded {len(existing_data)} previously processed examples")
        dict = existing_data  # Continue building on existing list
    else:
        dict = []



    for i, example in enumerate(tqdm(data, desc=f"Processing MMLU-Pro {type}")):
        # Format the question with its specific options
        question = format_example(example["question"], example["options"])
        
        if question in processed_questions:
            continue

        answer = example["answer"]

        max_retries = 5
        retry_count = 0
        cot_steps = None

        while retry_count < max_retries:
            try:
                cot_steps = get_label(question, answer)
                if (len(cot_steps) > 0):
                    break
                else:
                    retry_count += 1
                    print(f"Error occurred while processing question: {question}. Attempt {retry_count} of {max_retries}. ")
            except Exception as e:
                retry_count += 1
                print(f"Error occurred while processing question: {question}. Attempt {retry_count} of {max_retries}. Error: {e}")

        if retry_count == max_retries:
            print(f"Skipping question due to repeated errors: {question}")
            continue

        new_entry = {
            "question": question,
            "answer": answer,
            "cot_steps": cot_steps,
            "split": type
        }
        dict.append(new_entry)
        processed_questions.add(question)

        with open(json_file, "w") as f:
            json.dump(dict, f, indent=4)


# -------------------generate dataset steps for Commonsense QA----------------------

def generate_CommensenQA_agent(split):
    """
    Generates dataset steps for the Commonsense QA dataset.
    """
    type = split
    if type == "train":
        type = "train"

        data = load_dataset("tau/commonsense_qa")[type]
        json_file = "dataset_folder/commonsense_qa_{}.json".format(type)
        
        # load existing data to avoid duplication 
        processed_questions = set()
        if os.path.exists(json_file):
            with open(json_file, "r") as f:
                existing_data = json.load(f)
                processed_questions = {item["question"] for item in existing_data}
            print(f"✓ Loaded {len(existing_data)} previously processed examples")
            dict = existing_data
        else:
            dict = []

        for i, example in enumerate(tqdm(data, desc=f"Processing CommonsenseQA {type}")):
            q = example['question']
            choices = example['choices']['text']
            labels = example['choices']['label']
            
            # Reconstruct the question string with choices
            question = f"Question: {q} Options: "
            for label, choice in zip(labels, choices):
                question += f"{label}.{choice} "

            if question in processed_questions:
                continue
                

            answer = example['answerKey']

            max_retries = 5
            retry_count = 0
            cot_steps = None

            while retry_count < max_retries:
                try:
                    cot_steps = get_label(question, answer)
                    if (len(cot_steps) > 0):
                        break
                    else:
                        retry_count += 1
                        print(f"Error occurred while processing question: {question}. Attempt {retry_count} of {max_retries}.")
                except Exception as e:
                    retry_count += 1
                    print(f"Error occurred while processing question: {question}. Attempt {retry_count} of {max_retries}. Error: {e}")

            if retry_count == max_retries:
                print(f"Skipping question due to repeated errors: {question}")
                continue

            new_entry = {
                "question": question,
                "answer": answer,
                "cot_steps": cot_steps,
                "split": type
            }
            dict.append(new_entry)
            processed_questions.add(question)  

            with open(json_file, "w") as f:
                json.dump(dict, f, indent=4)

    else:  
        # Handle the validation/test split (no CoT generation, just formatting)
        type = "validation"
        data = load_dataset("tau/commonsense_qa")[type]
        print(data)
        json_file = "dataset_folder/commonsense_qa_test_clean_CC.json"
        
        # Optional: Add resume for test split too (if formatting is slow)
        processed_questions = set()
        if os.path.exists(json_file):
            with open(json_file, "r") as f:
                existing_data = json.load(f)
                processed_questions = {item["question"] for item in existing_data}
            print(f"✓ Loaded {len(existing_data)} previously formatted test examples")
            dict = existing_data
        else:
            dict = []

        for i, example in enumerate(tqdm(data, desc=f"Processing CommonsenseQA {type}")):
            q = example['question']
            choices = example['choices']['text']
            labels = example['choices']['label']
            
            question = f"Question: {q} Options: "
            for label, choice in zip(labels, choices):
                question += f"{label}.{choice} "

            if question in processed_questions:
                continue
                
            answer = example['answerKey']

            new_entry = {
                "question": question,
                "answer": answer,
                "cot_steps": [], # Leaves CoT empty for testing split
                "split": 'test'
            }
            dict.append(new_entry)
            
        with open(json_file, "w") as f:
            json.dump(dict, f, indent=4)
        # exit()



def clean_json(json_file, json_file1):
    """
    Cleans the generated JSON data by standardizing tags.
    Any tag that isn't [reason] or [rag] is automatically converted to [rag].

    Args:
        json_file (str): Path to the input JSON file.
        json_file1 (str): Path to the output (cleaned) JSON file.
    """
    with open(json_file, "r") as f:
        data = json.load(f)

    list = []
    pattern = re.compile(r"\[(.*?)\]")

    for item in data:
        question = item['question']
        answer = item['answer']
        cot_steps = item['cot_steps']
        type = item['split']

        # Filter out malformed statements missing a proper colon separator or content
        filtered_statements = [statement for statement in cot_steps if ': ' in statement and statement.split(': ')[1].strip()]
        if not filtered_statements:
            continue
        
        processed_data = []

        for entry in filtered_statements:
            match = pattern.search(entry)
            if match:
                tag = match.group(1)
                # Standardize unknown tags to 'rag'
                if tag not in ['reason','rag']:
                    entry = entry.replace(f"[{tag}]", "[rag]")  
            processed_data.append(entry)
        
        new_entry = {
            "question": question,
            "answer": answer,
            "cot_steps": processed_data,  
            "split": type
        }

        list.append(new_entry)

    with open(json_file1, "w") as f:
        json.dump(list, f, indent=4)


def generate_truthfulqa_agent(split):
    """
    Generates dataset steps for the TruthfulQA dataset.
    Handles multiple choice formatting and randomizing options.
    """
    split_type = split
    choice_letters = [chr(i) for i in range(ord('A'), ord('Z')+1)]

    # ============= TRAIN SPLIT (80% of validation, with CoT + Resume) =============
    if split_type == "train":
        ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice")
        json_file = "dataset_folder/truthful_qa_{}.json".format(split_type)
        data = ds['validation']
        train_data = data.select(range(int(len(data) * 0.8)))

        # ➕ RESUME LOGIC
        processed_questions = set()
        if os.path.exists(json_file):
            with open(json_file, "r") as f:
                existing_data = json.load(f)
                processed_questions = {item["question"] for item in existing_data}
            print(f"✓ Loaded {len(existing_data)} previously processed examples")
            data_list = existing_data
        else:
            data_list = []

        for i, example in enumerate(tqdm(train_data, desc=f"Processing TruthfulQA {split_type}")):
            question_text = example['question']
            choices = example['mc1_targets']['choices']
            labels = example['mc1_targets']['labels']
            
            # DETERMINISTIC SHUFFLE: Crucial for resume compatibility
            # Using index 'i' ensures the exact same shuffle every time this question is processed
            rng = random.Random(i)
            choices_and_labels = list(zip(choices, labels))
            rng.shuffle(choices_and_labels) 
            shuffled_choices, shuffled_labels = zip(*choices_and_labels)
            
            formatted_choices = [f"{choice_letters[j]}. {choice}" for j, choice in enumerate(shuffled_choices)]
            question = f"{question_text} {' '.join(formatted_choices)}"
            
            # ➕ SKIP IF ALREADY PROCESSED
            if question in processed_questions:
                continue
                
            # Find the correct answer (label == 1) based on the shuffled order
            answer_index = shuffled_labels.index(1)
            answer = choice_letters[answer_index]
            
            max_retries = 5
            retry_count = 0
            cot_steps = None
            
            while retry_count < max_retries:
                try:
                    cot_steps = get_label(question, answer)
                    if len(cot_steps) > 0:
                        print('success, aoligei')
                        break
                    else:
                        retry_count += 1
                        # ✅ FIX: Removed undefined 'e' variable
                        print(f"Error: Empty CoT steps. Attempt {retry_count} of {max_retries}")
                except Exception as e:
                    retry_count += 1
                    print(f"Error processing question. Attempt {retry_count} of {max_retries}. Error: {e}")

            if retry_count == max_retries:
                print(f"Skipping question due to repeated errors: {question_text[:50]}...")
                continue

            new_entry = {
                "question": question,
                "answer": answer,
                "cot_steps": cot_steps,
                "split": split_type
            }
            data_list.append(new_entry)
            processed_questions.add(question)  # Add to set to prevent future duplicates
            
            with open(json_file, "w") as f:
                json.dump(data_list, f, indent=4)

    # ============= TEST SPLIT (20% of validation, formatting only) =============
    else:  
        ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice")
        json_file = "dataset_folder/truthful_qa_{}.json".format(split_type)
        data = ds['validation']
        test_data = data.select(range(int(len(data) * 0.8), len(data)))
        
        processed_questions = set()
        if os.path.exists(json_file):
            with open(json_file, "r") as f:
                existing_data = json.load(f)
                processed_questions = {item["question"] for item in existing_data}
            print(f"✓ Loaded {len(existing_data)} previously formatted test examples")
            data_list = existing_data
        else:
            data_list = []
    
        for i, example in enumerate(tqdm(test_data, desc=f"Formatting TruthfulQA test")):
            question_text = example['question']
            choices = example['mc1_targets']['choices']
            labels = example['mc1_targets']['labels']
            
            # 🔑 Same deterministic shuffle for test split
            rng = random.Random(i)
            choices_and_labels = list(zip(choices, labels))
            rng.shuffle(choices_and_labels) 
            shuffled_choices, shuffled_labels = zip(*choices_and_labels)
            
            formatted_choices = [f"{choice_letters[j]}. {choice}" for j, choice in enumerate(shuffled_choices)]
            question = f"{question_text} {' '.join(formatted_choices)}"
            
            if question in processed_questions:
                continue
                
            answer_index = shuffled_labels.index(1)     
            answer = choice_letters[answer_index]
            
            new_entry = {
                "question": question,
                "answer": answer,
                "cot_steps": [],  # Leaves CoT empty for testing
                "split": 'test'
            }
            data_list.append(new_entry)
            processed_questions.add(question) # Add to set to prevent future duplicates
            
        with open(json_file, "w") as f:
            json.dump(data_list, f, indent=4)


def main(args):
    """
    Main function to route the execution based on CLI arguments and clean the resulting data.
    """
    # Create the output directory if it doesn't exist
    os.makedirs("dataset_folder", exist_ok=True)

    # Route execution based on the dataset argument
    if args.dataset == "StrategyQA":
        generate_StrategyQA_agent(args.mode)
    elif args.dataset == "MMLU_Pro":
        generate_MMLU_pro_agent(args.mode)
    elif args.dataset == "commonsense_qa":
        generate_CommensenQA_agent(args.mode)
    elif args.dataset == "truthful_qa":
        generate_truthfulqa_agent(args.mode)
        print(args.mode)
        
    # Standardize the tags in the generated file to ensure clean data for later use
    file = os.path.join("dataset_folder", "{}_{}.json".format(args.dataset, args.mode))
    clean_file = os.path.join("dataset_folder", "{}_{}.json".format(args.dataset, args.mode + "_clean_CC"))
    clean_json(file, clean_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Qwen Data Generation")
    parser.add_argument('--dataset', type=str, required=True, help="Dataset to process (e.g., StrategyQA)")
    parser.add_argument('--mode', type=str, required=True, help="Split mode (e.g., train, test)")
    
    args = parser.parse_args()
    
    # This actually kicks off the generation process
    main(args)