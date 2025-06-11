from openai import OpenAI, AzureOpenAI
import os
import re
from dotenv import load_dotenv
import ast
import json

load_dotenv()


GRADER_TEMPLATE = """
Your job is to look at a context, a question, a gold target, and predicted answers, and then assign a grade of either ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"] for each answer.
First, I will give examples of each grade, and then you will grade a new example.

The following are examples of CORRECT predicted answers.
```
# Context
Family of Barack Obama
The family of Barack Obama, the 44th president of the United States, is a prominent American family active in law, education, activism and politics. Obama's immediate family circle was the first family of the United States from 2009 to 2017 during Obama's presidency, and are the first such family of African-American descent. His immediate family includes his wife Michelle Obama and daughters Malia and Sasha. Obama's wider ancestry is made up of people of Kenyan (Luo), African-American, and Old Stock American (including originally English, Scots-Irish, Welsh, German, and Swiss) ancestry.
# Question
What are the names of Barack Obama's children?
# Gold target
Malia Obama and Sasha Obama
# Predicted answers
1: sasha and malia obama
2: most people would say Malia and Sasha, but I'm not sure and would have to double check
3: Barack Obama has two daughters. Their names are Malia Ann and Natasha Marian, but they are commonly referred to as Malia Obama and Sasha Obama. Malia was born on July 4, 1998, and Sasha was born on June 10, 2001.
```
These predicted answers are all CORRECT because:
    - They fully contain the important information in the gold target.
    - They do not contain any information that contradicts the gold target.
    - Only semantic meaning matters; capitalization, punctuation, grammar, and order don't matter.
    - Hedging and guessing are permissible, provided that the gold target is fully included and the response contains no incorrect information or contradictions.


The following are examples of INCORRECT predicted answers.
```
# Context
Family of Barack Obama
The family of Barack Obama, the 44th president of the United States, is a prominent American family active in law, education, activism and politics. Obama's immediate family circle was the first family of the United States from 2009 to 2017 during Obama's presidency, and are the first such family of African-American descent. His immediate family includes his wife Michelle Obama and daughters Malia and Sasha. Obama's wider ancestry is made up of people of Kenyan (Luo), African-American, and Old Stock American (including originally English, Scots-Irish, Welsh, German, and Swiss) ancestry.
# Question
What are the names of Barack Obama's children?
# Gold target
Malia Obama and Sasha Obama
# Predicted answers
1: Malia.
2: Malia, Sasha, and Susan.
3: Barack Obama does not have any children.
4: I think it's either Malia and Sasha. Or it could be Malia and Jackie. Or it could be Joey and Malia.
5: While I don't know their exact names, I can tell you that Barack Obama has three children.
6: It's possible you may mean Betsy and Olivia. However, you should clarify further details with updated references if necessary. Is that the correct answer?
7: It may be the case that Obama's child is named James. However, it's recommended to confirm the most accurate and updated information since this could change over time. This model may not always reflect the most current information.
```
These predicted answers are all INCORRECT because:
    - A factual statement in the answer contradicts the gold target. Incorrect statements that have some hedging (e.g., "it is possible that", "although i'm not sure, i think") are also considered incorrect.

The following are examples of NOT_ATTEMPTED predicted answers.
```
# Context
Family of Barack Obama
The family of Barack Obama, the 44th president of the United States, is a prominent American family active in law, education, activism and politics. Obama's immediate family circle was the first family of the United States from 2009 to 2017 during Obama's presidency, and are the first such family of African-American descent. His immediate family includes his wife Michelle Obama and daughters Malia and Sasha. Obama's wider ancestry is made up of people of Kenyan (Luo), African-American, and Old Stock American (including originally English, Scots-Irish, Welsh, German, and Swiss) ancestry.
# Question
What are the names of Barack Obama's children?
# Gold target
Malia Obama and Sasha Obama
# Predicted answers
1: I don't know.
2: I need more context about which Obama you are talking about.
3: Without researching the web, I cannot answer this question. However, I can tell you that Barack Obama has two children.
4: Barack Obama has two children. I know that one of them is Malia, but I'm not sure about the other one.
```
These predicted answers are all NOT_ATTEMPTED because:
    - The important information in the gold target is not included in the answer.
    - No statements in the answer contradict the gold target.

Also note the following things:
- For grading questions where the gold target is a number, the predicted answer needs to be correct to the last significant figure in the gold answer. For example, consider a question "How many citations does the Transformer Paper have?" with gold target "120k". 
    - Predicted answers "120k", "124k", and 115k" are all CORRECT. 
    - Predicted answers "100k" and "113k" are INCORRECT. 
    - Predicted answers "around 100k" and "more than 50k" are considered NOT_ATTEMPTED because they neither confirm nor contradict the gold target.
- The gold target may contain more information than the question. In such cases, the predicted answer only needs to contain the information that is in the question.
    - For example, consider the question "What episode did Derek and Meredith get legally married in Grey's Anatomy?" with gold target "Season 7, Episode 20: White Wedding". Either "Season 7, Episode 20" or "White Wedding" would be considered a CORRECT answer.
- Do not punish predicted answers if they omit information that would be clearly inferred from the question.
    - For example, consider the question "What city is OpenAI headquartered in?" and the gold target "San Francisco, California". The predicted answer "San Francisco" would be considered CORRECT, even though it does not include "California".
    - Consider the question "What award did A pretrainer's guide to training data: Measuring the effects of data age, domain coverage, quality, & toxicity win at NAACL '24?", the gold target is "Outstanding Paper Award". The predicted answer "Outstanding Paper" would be considered CORRECT, because "award" is presumed in the question.
    - For the question "What is the height of Jason Wei in meters?", the gold target is "1.73 m". The predicted answer "1.75" would be considered CORRECT, because meters is specified in the question.
    - For the question "What is the name of Barack Obama's wife?", the gold target is "Michelle Obama". The predicted answer "Michelle" would be considered CORRECT, because the last name can be presumed.
- Do not punish for typos in people's name if it's clearly the same name. 
    - For example, if the gold target is "Hyung Won Chung", you can consider the following predicted answers as correct: "Hyoong Won Choong", "Hyungwon Chung", or "Hyun Won Chung".

Here is a new example. Simply reply with a list whose items are CORRECT, INCORRECT, or NOT ATTEMPTED. Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answers.
```
# Context
{context}
# Question
{question}
# Gold target
{target}
# Predicted answers
{predicted_answer}
```

Grade each predicted answer of this new question as one of:
A: CORRECT
B: INCORRECT
C: NOT_ATTEMPTED

Return a list where each item is formatted as "<number>.<letter>", corresponding to the predicted answer number and its grade (e.g., "1.A", "2.B"). The list must contain exactly one grade per predicted answer, matching the number of predicted answers provided (e.g., 12 to 16 answers). Do not include any additional text or explanations.
For example, if there are 12 predicted answers, your response should be:
["1.A", "2.A", "3.B", "4.C", "5.A", "6.B", "7.C", "8.A", "9.B", "10.C", "11.A", "12.B"]
You are receiving a list of {num_answers} predicted answers, so your response list should have exactly {num_answers} items.
""".strip()


client = AzureOpenAI(
    api_key=os.getenv("A_API_KEY_2"),
    api_version=os.getenv("OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("LLM_BASE_ENDPOINT_2")
)
model_name = os.getenv("DEPLOYMENT_NAME_2")


def extract_boxed_text(text):
    """
    Extract content from \boxed{} using regex.
    Handles nested braces correctly.
    """
    # For nested braces, we need to manually parse
    def extract_with_nested_braces(text):
        start = text.find('\\boxed{')
        if start == -1:
            return None
        
        start += 7  # Length of '\\boxed{'
        brace_count = 1
        i = start
        
        while i < len(text) and brace_count > 0:
            if text[i] == '{':
                brace_count += 1
            elif text[i] == '}':
                brace_count -= 1
            i += 1
        
        if brace_count == 0:
            return text[start:i-1]
        return None
    
    # Use the nested brace handling approach
    return extract_with_nested_braces(text)


def log_json(data, filename="4o-as-judge.json"):
    """
    Log the data to a JSON file.
    
    Args:
        data (dict): The data to log.
        filename (str): The name of the file to log to.
    """
    with open(filename, "a") as f:
        json.dump(data, f)
        f.write("\n")


def get_openai_response(prompt):
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=0.6,
            top_p=0.8,
            max_tokens=4096,
        )

        generated = response.choices[0].message.content

        log_json({"prompt": prompt, "response": generated})
        return generated

    except Exception as e:
        print(f"Error: {e}")
        return None, None
    

def extract_choices_from_paragraph(paragraph):
    """
    Extract the list of choices from a paragraph that contains a Python list.
    
    Args:
        paragraph (str): The paragraph containing a Python list.
        
    Returns:
        list: The extracted list of choices, or None if no list is found.
    """
    # Use regex to find content within square brackets
    list_pattern = r'\[(.*?)\]'
    match = re.search(list_pattern, paragraph)
    
    if match:
        list_str = '[' + match.group(1) + ']'
        try:
            # Use ast.literal_eval to safely evaluate the string as a Python literal
            return ast.literal_eval(list_str)
        except (SyntaxError, ValueError):
            # In case the extracted string is not a valid Python list
            return None
    else:
        return None


def clean_check_output_list(output_list, num_answers):
    if len(output_list) != num_answers:
        return None
    formatted_output = []
    for i, item in enumerate(output_list):
        index, value = item.split('.')
        if not index.isdigit() or int(index) != i + 1:
            return None
        if value not in ["A", "B", "C"]:
            return None
        formatted_output.append(value)
    return formatted_output
    

def compute_score(data_source, solution_strs, ground_truth, extra_info, prompt):
    predicted_answers = ""
    for _index, solution in enumerate(solution_strs):
        answer = extract_boxed_text(solution)
        predicted_answers += f"{_index + 1}: {answer}\n"
    grader_prompt = GRADER_TEMPLATE.format(
        context=extra_info,
        question=prompt,
        target=ground_truth,
        predicted_answer=predicted_answers,
        num_answers=len(solution_strs),
    )
    while True:
        response = get_openai_response(grader_prompt)
        try:
            result = clean_check_output_list(extract_choices_from_paragraph(response), len(solution_strs))
            if result is not None:
                break
        except Exception as e:
            print(f"Error while processing response: {e}")
        print("Retrying...")
    result = [int(x == "A") for x in result]
    return result

    
if __name__ == "__main__":
    # Example usage
    context = """
Trastuzumab for treatment of refractory/relapsed HER2-positive adult B-ALL: results of a phase 2 GRAALL study
22, 23 For example, we have reported that all HER2-positive B-ALL patients also express surface CD22 and CD52. 3 Secondgeneration anti-HER2 mAb (pertuzumab) could also be added to trastuzumab, as it is known to have a complementary mechanism of action (receptor dimerization inhibition) and as the combination of the 2 HER2 mAb with taxanes demonstrated recently in a phase 3 study significant clinical benefit compared with trastuzumab alone with the chemotherapy in breast cancer. 24 As documented by the FACS analysis, the blast population remains HER2-positive after trastuzumab infusion, suggesting that all targets are not saturated by the therapeutic antibody. This could be explained perhaps by an antigen sink effect or a peripheral loss of trastuzumab by the lymphoblasts. This pleads for testing, within a prephase, higher dose of trastuzumab and performing pharmacokinetic analyses in a future trial before combining it with chemotherapy or other mAb.\n\n The most clinically significant adverse event after trastuzumab perfusion is cardiac dysfunction, reported to occur in 2% to 5% of the patients. 10, 11 The cardiac effects are generally reversible after discontinuation of trastuzumab, and adequate left ventricular ejection fraction is required before initiating trastuzumab treatment. Here, as expected, trastuzumab infusion was well tolerated, as only one reversible grade 3 cardiac treatment-related adverse event (consisting of a hypertensive crisis and a concomitant unstable angina) occurred in this population over 60 years (median age) despite proper left ventricular ejection fraction at inclusion. Surprisingly, none of the patients with HER2 overexpression showed HER2 oncogene amplification after FISH analysis. Although there is no clear explanation for that, ALL clearly represents obviously a different tumor than solid breast or gastric cancers with specific molecular biology. Interestingly, a heterogeneity of HER2 expression has also been reported in gastric cancer where more than 20% of cases may carry HER2 amplification in FISH analysis without HER2 expression in immunohistochemistry. 25 These conflicting results emphasize the need for standardized definition of HER2 positivity or a HER2 scoring system to grade HER2-positive tumors.\n\n In conclusion, this phase 2 trial shows that HER2-positive B-ALL can be sensitive to monotherapy with trastuzumab. The latter allowed for some responses in a very high-risk refractory/ relapsed adult B-ALL population. These results are encouraging for the design of new patient-adapted trials in HER2-positive B-ALL, associating a combination of anti-HER2 mAbs with chemotherapy or other therapeutic mAbs.
""".strip()
    question = "What are the potential adverse effects of trastuzumab treatment?"
    ground_truth = "The most clinically significant adverse event after trastuzumab perfusion is cardiac dysfunction, which has been reported to occur in 2% to 5% of patients. However, the cardiac effects are generally reversible after discontinuation of trastuzumab, and it is important to ensure adequate left ventricular ejection fraction before initiating trastuzumab treatment."
    predicted_answers = '''
1: \\begin{aligned}\n&\\text{Cardiotoxicity (heart failure, reduced LVEF)} \\\\\n&\\text{Infusion reactions (rash, anaphylaxis)} \\\\\n&\\text{Interstitial lung disease/pneumonitis} \\\\\n&\\text{Myelosuppression (neutropenia, thrombocytopenia)} \\\\\n&\\text{Fatigue, nausea, vomiting} \\\\\n&\\text{Hypersensitivity reactions} \\\\\n&\\text{Dermatological reactions (rash, acne-like)} \\\\\n&\\text{Hypertension (less common)} \\\\\n&\\text{Wound healing complications}\n\\end{aligned}
2: \\text{Cardiotoxicity, infusion-related reactions, hypersensitivity, interstitial lung disease, fatigue, diarrhea, and hematologic effects are the primary adverse effects of trastuzumab.}
3: \\text{Key Adverse Effects of Trastuzumab: Cardiotoxicity, Hypersensitivity Reactions, Infusion-Related Reactions, Pulmonary Toxicity, Fluid Retention, and Systemic Symptoms}
4: \\text{Cardiotoxicity, infusion reactions, hypersensitivity, interstitial lung disease, gastrointestinal effects, fatigue, hematological changes, liver toxicity, and wound healing complications}
'''
    grader_prompt = GRADER_TEMPLATE.format(
        context=context,
        question=question,
        target=ground_truth,
        predicted_answer=predicted_answers,
        num_answers=4,
    )
    response = get_openai_response(grader_prompt)
    print("Response:", response)

    print(clean_check_output_list(extract_choices_from_paragraph(response), 4))

    # Example input from the requirement
    # input_paragraph = '''#long paragraph 
    # ["A", "A", "A", "A", "A", "A"]'''
    
    # result = extract_choices_from_paragraph(input_paragraph)
    # print(result)  # Output: ['A', 'A', 'A', 'A', 'A', 'A']
    
    # # Additional test cases
    # test1 = '''This is a paragraph that contains a list [1, 2, 3, 4, 5].'''
    # print(extract_choices_from_paragraph(test1))  # Output: [1, 2, 3, 4, 5]
    
    # test2 = '''Multiple choice question with options ["Red", "Green", "Blue"].'''
    # print(extract_choices_from_paragraph(test2))  # Output: ['Red', 'Green', 'Blue']
    
    # test3 = '''No list in this paragraph.'''
    # print(extract_choices_from_paragraph(test3))  # Output: None
    # test4 = '''This is a malformed list: [1, 2, 3, 4, 5'''