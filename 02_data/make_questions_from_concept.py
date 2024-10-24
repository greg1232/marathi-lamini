import lamini

import jsonlines

import logging

logger = logging.getLogger(__name__)


def main():
    setup_logging()

    concept_description = load_concept_description()

    concepts = extract_concepts(concept_description)

    questions = make_questions(concepts)

    questions_and_answers = answer_questions(questions)

    save_questions(questions_and_answers)


def setup_logging():
    logging.basicConfig(level=logging.INFO)


def load_concept_description():
    path = "/app/marathi-llm/data/raw-data/concepts/ahey.txt"

    with open(path, "r") as file:
        concept_description = file.read()

    return concept_description


def extract_concepts(concept_description):
    llm = lamini.Lamini(model_name="meta-llama/Meta-Llama-3.1-8B-Instruct")

    prompt = make_extract_concepts_prompt(concept_description)

    logger.info(f"Prompt: \n{prompt}")

    output = llm.generate(
        prompt, output_type={"explanation": "str", "concepts": "list(str, 7)"}
    )

    logger.info(f"Output: {output}")

    concepts = output["concepts"]

    return [
        {"concept": concept, "description": concept_description} for concept in concepts
    ]


def make_extract_concepts_prompt(concept_description):
    prompt = "You are an expert marathi teacher.\n"
    prompt += "Consider the following notes about Marathi concepts.\n"
    prompt += "==============================\n"
    prompt += concept_description + "\n"
    prompt += "==============================\n"
    prompt += "Extract the concepts from the notes.\n"
    prompt += "First explain how you extracted the concepts in one sentence.\n"
    prompt += "Finally, list the concepts extracted.\n"
    prompt += "Stop when you have extracted all concepts.\n"
    prompt += "Each concept should be described in about 5 words.\n"
    prompt += "Format your response as a JSON object with the following schema:\n"
    prompt += "{\n"
    prompt += '  "explanation": "explanation of how you extracted the concepts",\n'
    prompt += '  "concepts": ["concept1", "concept2", ...]\n'
    prompt += "}\n"
    return prompt


def make_questions(concepts):
    llm = lamini.Lamini(model_name="meta-llama/Meta-Llama-3.1-8B-Instruct")

    prompts = make_make_questions_prompts(concepts)

    questions_and_answers = llm.generate(
        prompts, output_type={"explanation": "str", "questions": "list(str, 5)"}
    )

    questions = []

    for concept, qa in zip(concepts, questions_and_answers):
        logger.info(
            f"Explanation for concept: {concept['concept']}\n{qa['explanation']}"
        )
        for question in qa["questions"]:
            logger.info(f"Question for concept: {concept['concept']}\n{question}")
            questions.append(
                {
                    "concept": concept["concept"],
                    "description": concept["description"],
                    "question": question,
                    "question_explanation": qa["explanation"],
                }
            )

    return questions


def make_make_questions_prompts(concepts):
    prompts = []

    for concept in concepts:
        prompt = make_make_questions_prompt(concept)
        logger.info(f"Prompt for concept: {concept['concept']}\n{prompt}")
        prompts.append(prompt)

    return prompts


def make_make_questions_prompt(concept):
    prompt = "<|start_header_id|>user<|end_header_id|>"
    prompt += "You are an expert marathi teacher.\n"
    prompt += "Consider the following notes explanaing several concepts.\n"
    prompt += "==============================\n"
    prompt += concept["description"] + "\n"
    prompt += "==============================\n"
    prompt += "Focus on the following concept in the notes.\n"
    prompt += "==============================\n"
    prompt += concept["concept"] + "\n"
    prompt += "==============================\n"
    prompt += (
        "First explain your strategy for writing the questions in three sentences.\n"
    )
    prompt += "Then, write 5 different questions about the concept.\n"
    prompt += "Each question should be about one sentence long.\n"
    prompt += "Ask questions that test understanding of the concept.\n"
    prompt += "Ask the questions in marathi.\n"
    prompt += "Format your response as a JSON object with the following schema:\n"
    prompt += '{ "explanation": "explanation of how you wrote the questions", "question": list(str) }'
    prompt += "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

    return prompt


def answer_questions(questions):
    llm = lamini.Lamini(model_name="meta-llama/Meta-Llama-3.1-8B-Instruct")

    prompts = make_answer_questions_prompts(questions)

    answers = llm.generate(
        prompts, output_type={"explanation": "str", "answer": "str"}
    )

    questions_and_answers = []

    for question, qa in zip(questions, answers):
        logger.info(f"Explanation for the answer: {question['question']}\n{qa['explanation']}")
        logger.info(f"Answer for question: {question['question']}\n{qa['answer']}")
        questions_and_answers.append(
            {
                "concept": question["concept"],
                "description": question["description"],
                "question": question["question"],
                "question_explanation": question["question_explanation"],
                "answer": qa["answer"],
                "answer_explanation": qa["explanation"],
            }
        )

    return questions_and_answers

def make_answer_questions_prompts(questions):
    prompts = []

    for question in questions:
        prompt = make_answer_questions_prompt(question)
        logger.info(f"Prompt for question: {question['question']}\n{prompt}")
        prompts.append(prompt)

    return prompts

def make_answer_questions_prompt(question):
    prompt = "<|start_header_id|>user<|end_header_id|>"
    prompt += "You are an expert marathi teacher.\n"
    prompt += "Consider the following notes explanaing several concepts.\n"
    prompt += "==============================\n"
    prompt += question["description"] + "\n"
    prompt += "==============================\n"
    prompt += "Focus on the following concept in the notes.\n"
    prompt += "==============================\n"
    prompt += question["concept"] + "\n"
    prompt += "==============================\n"
    prompt += "Consider the following question about the concept.\n"
    prompt += "==============================\n"
    prompt += question["question"] + "\n"
    prompt += "==============================\n"
    prompt += "First explain your strategy for answering the question in three sentences.\n"
    prompt += "Then, write the answer to the question.\n"
    prompt += "The answer should be about one sentence long.\n"
    prompt += "Answer the question in marathi.\n"
    prompt += "Format your response as a JSON object with the following schema:\n"
    prompt += '{ "explanation": "str", "answer": "str" }'
    prompt += "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

    return prompt


def save_questions(questions_and_answers):
    path = "/app/marathi-llm/data/ahey.jsonlines"

    with jsonlines.open(path, "w") as file:
        for question_and_answer in questions_and_answers:
            file.write(question_and_answer)


main()
