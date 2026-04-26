import sys
import os
import io
import json
import re
import torch
import argparse
import random
from tqdm import tqdm
from datasets import load_dataset  # Added missing import
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig

# # Fix encoding issues for Slurm logs
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
# sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# # Pathing for custom modules
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from load_data.preprocess import *

# Pathing for custom modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from load_data.preprocess import *
except ImportError as e:
    print(f"Warning: Could not import from load_data.preprocess: {e}")

# --- CONFIGURATION ---
MODEL_NAME = "Qwen/Qwen3.5-27B"
CACHE_DIR = os.path.expanduser("~/models/")
_pipe = None 


# with quantization with bitsandbytes --> but we can run 16bit anyways, and avoid update problems 
# def load_local_model():
#     global _pipe
#     if _pipe is not None:
#         return _pipe

#     print(f"Loading {MODEL_NAME} from {CACHE_DIR}...")
#     tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR, trust_remote_code=True)
    
#     bnb_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_use_double_quant=True,
#         bnb_4bit_quant_type="nf4",
#         bnb_4bit_compute_dtype=torch.bfloat16
#     )

#     model = AutoModelForCausalLM.from_pretrained(
#         MODEL_NAME,
#         quantization_config=bnb_config,
#         device_map="auto",
#         cache_dir=CACHE_DIR,
#         trust_remote_code=True
#     )

#     _pipe = pipeline(
#         "text-generation",
#         model=model,
#         tokenizer=tokenizer,
#     )
#     return _pipe

def load_local_model():
    global _pipe
    if _pipe is not None:
        return _pipe

    print(f"Loading {MODEL_NAME} from {CACHE_DIR} in pure bfloat16...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.bfloat16, # Native 16-bit precision
        cache_dir=CACHE_DIR,
        trust_remote_code=True
    )

    _pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )
    return _pipe

def get_response(model_name, prompt): # Updated signature to handle 2 arguments
    pipe = load_local_model()
    # Format messages for Qwen/HuggingFace pipeline
    messages = [{"role": "user", "content": prompt}]
    
    outputs = pipe(
        messages,
        max_new_tokens=1536,
        do_sample=False, 
        return_full_text=False,
    )
    # Extract the content from the generated response
    return outputs[0]["generated_text"]

# --- FORMATTING & PARSING ---

def format_example(question, options, cot_content=""):
    if cot_content.startswith("A: "):
        cot_content = cot_content[3:]
    example = "Question: {}\nOptions: ".format(question)
    choice_map = "ABCDEFGHIJ"
    for i, opt in enumerate(options):
        example += "{}. {}\n".format(choice_map[i], opt)
    example += "Answer: " + (cot_content + "\n\n" if cot_content else "")
    return example

def extract_labeled_content_as_list(input_string):
    steps = re.split(r'\*Step name\*:', input_string)
    labeled_content = []
    for step in steps:
        if step.strip():
            requirement_match = re.search(r'\*\*Requirement\*\*: \[(.*)\]', step)
            content_match = re.search(r'\*\*Content\*\*: (.*)', step)
            if requirement_match and content_match:
                requirement = f"[{requirement_match.group(1)}]"
                content = content_match.group(1).strip()
                labeled_content.append(f"{requirement}: {content}")
    return labeled_content

def extract_knowledge_based(text):
    pattern = r"\*\*Knowledge based\*\*:(.*?)\*\*Content\*\*"
    matches = re.findall(pattern, text, re.DOTALL)
    return [match.strip() for match in matches]

def clean_and_parse_json_string_with_codeblock(json_str):
    json_str = json_str.replace('```json', '').replace('```', '')
    json_str = re.sub(r'"[^"]*"\s*:\s*""\s*,?', '', json_str)
    json_str = re.sub(r',\s*}', '}', json_str)  
    json_str = re.sub(r',\s*]', ']', json_str)  
    json_str = json_str.strip().strip(',')
    if json_str.count('{') != json_str.count('}') or json_str.count('[') != json_str.count(']'):
        raise ValueError("The JSON string has unbalanced braces or brackets.")
    return json.loads(json_str)

def planning(question, answer):
    return f"""
        Here is the question:
        <Question>
        {question}
        </Question>

        Here is the correct answer:
        <Correct Answer>
        {answer}
        </Correct Answer>

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


# def NER_agent(questions, model_name="gpt-4o"):
#     template = f"""
#     Factual knowledge is information that aligns with objective reality and can be verified through evidence or observation, such as scientific facts or historical events.

#     Please provide factual knowledge for below question set:
# <Questions>
# {questions}
# <\Questions>

# You should return a dic in json format, for each element in dic, the key is each question in <Questions>, the value is the Factual knowledge of each question in <Questions>.
# Your answer format should strictly be in following steps:
# ```json
# {{
#       "question 1": "The factual knowledge of question 1",
# ....
# }}
# ```

#       """
#     text = get_response(model_name, template)

#     return text

def NER_agent(questions, model_name="local"):
    template = f"""
        Factual knowledge is information that aligns with objective reality and can be verified through evidence or observation, such as scientific facts or historical events.

            Please provide factual knowledge for below question set:
        <Questions>
        {questions}
        <\Questions>

        You should return a dic in json format, for each element in dic, the key is each question in <Questions>, the value is the Factual knowledge of each question in <Questions>.
        Your answer format should strictly be in following steps:
        ```json
        {{
            "question 1": "The factual knowledge of question 1",
        ....
        }}
        ```
    """
    return get_response(model_name, template)



# def get_label(input_string,answer):
#     planing = get_response("gpt-4o", planning(input_string,answer))
#     list = clean_and_parse_json_string_with_codeblock(NER_agent(extract_knowledge_based(planing)))
#     for item in list.keys():
#         planing = planing.replace(item, item + " [rag]" + list[item])
#     return extract_labeled_content_as_list(planing)


def get_label(input_string, answer):
    planing = get_response("local", planning(input_string, answer))
    kb_sections = extract_knowledge_based(planing)
    if kb_sections:
        ner_output = NER_agent(kb_sections)
        try:
            kb_list = clean_and_parse_json_string_with_codeblock(ner_output)
            for item in kb_list.keys():
                planing = planing.replace(item, f"{item} [rag]{kb_list[item]}")
        except:
            pass
    return extract_labeled_content_as_list(planing)


def generate_StrategyQA_agent(type):
    data = load_dataset("ChilleD/StrategyQA")[type]
    results_dict = []
    json_file = f"dataset_folder/StrategyQA_{type}.json"
    
    for example in tqdm(data):
        question = example["question"]
        answer = 'True' if example["answer"] else 'False'
        cot_steps = []
        for retry_count in range(5):
            try:
                cot_steps = get_label(question, answer)
                if len(cot_steps) > 0:
                    break
            except Exception as e:
                print(f"Error: {e}")
        
        results_dict.append({
            "question": question,
            "answer": answer,
            "cot_steps": cot_steps,
            "split": type
        })
        with open(json_file, "w") as f:
            json.dump(results_dict, f, indent=4)


def generate_MMLU_pro_agent(split):
    type = split
    if type =="train":
        type = "validation"
    else:
        type = "test"
    print(type)
    data = load_dataset("TIGER-Lab/MMLU-Pro")[type]
    dict = []
    json_file = "dataset_folder/MMLU_Pro_{}.json".format(type)

    for example in tqdm(data):
        question = format_example(example["question"],example["options"])
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
                    print(
                        f"Error occurred while processing question: {question}. Attempt {retry_count} of {max_retries}. Error: {e}")
            except Exception as e:
                retry_count += 1
                print(
                    f"Error occurred while processing question: {question}. Attempt {retry_count} of {max_retries}. Error: {e}")

        # if the maximum number of retries is reached, skip the question
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

        with open(json_file, "w") as f:
           json.dump(dict, f, indent=4)


def generate_CommensenQA_agent(split):
    type = split
    if type =="train":
        type = "train"

        data = load_dataset("tau/commonsense_qa")[type]
        dict = []
        json_file = "dataset_folder/commonsense_qa_{}.json".format(type)
        

        for example in tqdm(data):
            q = example['question']
            choices = example['choices']['text']
            labels = example['choices']['label']
            question = f"Question: {q} Options: "
            for label, choice in zip(labels, choices):
                question += f"{label}.{choice} "
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
                        print(
                            f"Error occurred while processing question: {question}. Attempt {retry_count} of {max_retries}. Error: {e}")
                except Exception as e:
                    retry_count += 1
                    print(
                        f"Error occurred while processing question: {question}. Attempt {retry_count} of {max_retries}. Error: {e}")

            # if the maximum number of retries is reached, skip the question
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

            with open(json_file, "w") as f:
                json.dump(dict, f, indent=4)

    else:  
        type = "validation"
        data = load_dataset("tau/commonsense_qa")[type]
        print(data)
        json_file = "dataset_folder/commonsense_qa_test_clean_CC.json"
        dict = []
        for example in tqdm(data):
            q = example['question']
            choices = example['choices']['text']
            labels = example['choices']['label']
            question = f"Question: {q} Options: "
            for label, choice in zip(labels, choices):
                question += f"{label}.{choice} "
            answer = example['answerKey']

            new_entry = {
                "question": question,
                "answer": answer,
                "cot_steps": [],
                "split": 'test'
            }
            dict.append(new_entry)
        with open(json_file, "w") as f:
            json.dump(dict, f, indent=4)
        exit()

        

   


def clean_json(json_file, json_file1):
    with open(json_file, "r") as f:
        data = json.load(f)

    list = []
    pattern = re.compile(r"\[(.*?)\]")

    for item in data:

        question = item['question']
        answer = item['answer']
        cot_steps = item['cot_steps']
        type = item['split']

        filtered_statements = [statement for statement in cot_steps if ': ' in statement and statement.split(': ')[1].strip()]
        if not filtered_statements:
            continue
        

        processed_data = []

        for entry in filtered_statements:
            match = pattern.search(entry)
            if match:
                tag = match.group(1)
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
    type = split
    if type =="train":
        type = "train"

        ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice")
        json_file = "dataset_folder/truthful_qa_{}.json".format(type)
        data = ds['validation']
        choice_letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
        train_data = data.select(range(int(len(data) * 0.8)))
        dict = []
        for example in train_data:
            question_text = example['question']
            choices = example['mc1_targets']['choices']
            labels = example['mc1_targets']['labels']
            choices_and_labels = list(zip(choices, labels))
            random.shuffle(choices_and_labels) 
            shuffled_choices, shuffled_labels = zip(*choices_and_labels)
            formatted_choices = [f"{choice_letters[i]}. {choice}" for i, choice in enumerate(shuffled_choices)]
            question = f"{question_text} {' '.join(formatted_choices)}"
            answer_index = shuffled_labels.index(1)
            answer = choice_letters[answer_index]
            

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
                        print(
                            f"Error occurred while processing question: {question}. Attempt {retry_count} of {max_retries}. Error: {e}")
                except Exception as e:
                    retry_count += 1
                    print(
                        f"Error occurred while processing question: {question}. Attempt {retry_count} of {max_retries}. Error: {e}")

            # if the maximum number of retries is reached, skip the question
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

            with open(json_file, "w") as f:
                json.dump(dict, f, indent=4)

    else:  
        ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice")
        json_file = "dataset_folder/truthful_qa_{}_clean_CC.json".format(type)
        data = ds['validation']
        test_data = data.select(range(int(len(data) * 0.8), len(data)))
        dict = []
        choice_letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
       
        for example in test_data:
            question_text = example['question']
            choices = example['mc1_targets']['choices']
            labels = example['mc1_targets']['labels']
            choices_and_labels = list(zip(choices, labels))
            random.shuffle(choices_and_labels) 
            shuffled_choices, shuffled_labels = zip(*choices_and_labels)
            formatted_choices = [f"{choice_letters[i]}. {choice}" for i, choice in enumerate(shuffled_choices)]
            question = f"{question_text} {' '.join(formatted_choices)}"
            answer_index = shuffled_labels.index(1)
            answer = choice_letters[answer_index]
            
            new_entry = {
                "question": question,
                "answer": answer,
                "cot_steps": [],
                "split": 'test'
            }
            dict.append(new_entry)
            with open(json_file, "w") as f:
                json.dump(dict, f, indent=4)
        exit()


def main(args):
    os.makedirs("dataset_folder", exist_ok=True)
    if args.dataset ==  "StrategyQA":
        generate_StrategyQA_agent(args.mode)
    elif args.dataset == "MMLU_Pro":
        generate_MMLU_pro_agent(args.mode)
    elif args.dataset == "commonsense_qa":
        generate_CommensenQA_agent(args.mode)
    elif args.dataset == "truthful_qa":
        generate_truthfulqa_agent(args.mode)
        print(args.mode)
        
    file = os.path.join("dataset_folder", "{}_{}.json".format(args.dataset, args.mode))
    clean_file = os.path.join("dataset_folder", "{}_{}.json".format(args.dataset, args.mode + "_clean_CC"))
    clean_json(file, clean_file)


   
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--dataset', type=str, default='truthful_qa', choices = ['commonsense_qa',"StrategyQA","truthful_qa"])
#     parser.add_argument('--mode',type=str,default="train",choices = ['train','test'])
#     args = parser.parse_args()
#     main(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='truthful_qa', choices=['commonsense_qa',"StrategyQA","truthful_qa"])
    parser.add_argument('--mode', type=str, default="train", choices=['train','test'])
    args = parser.parse_args()
    main(args)    
    
