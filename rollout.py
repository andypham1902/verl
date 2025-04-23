import pandas as pd
from pandarallel import pandarallel
from openai import OpenAI, AzureOpenAI
import os
import re
from dotenv import load_dotenv
# Initialize pandarallel
pandarallel.initialize(progress_bar=True, nb_workers=8)

load_dotenv()

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:30000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
# model_name = "meta-llama/Llama-3.3-70B-Instruct"
model_name = "Qwen/QwQ-32B"

# openai_api_key = os.getenv("DEEPSEEK_API_KEY")
# openai_api_base = "https://api.deepseek.com/v1"

# client = OpenAI(
#     api_key=openai_api_key,
#     base_url=openai_api_base,
# )
# model_name = "deepseek-reasoner"

# client = AzureOpenAI(
#     api_key=os.getenv("A_API_KEY_2"),
#     api_version=os.getenv("OPENAI_API_VERSION"),
#     azure_endpoint=os.getenv("LLM_BASE_ENDPOINT_2")
# )
# model_name = os.getenv("DEPLOYMENT_NAME_2")

# Configure Gemini API
from google import genai
gemini_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY_II"))


def extract_boxed_text(text):
    pattern = r'oxed{(.*?)}'
    matches = re.findall(pattern, text)
    if not matches:
        return ""
    for match in matches[::-1]:
        if match != "":
            return match
    return ""


def get_openai_response(row):
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                # {"role": "system", "content": "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step. Return final answer within \\boxed{}, after taking modulo 1000."},
                {"role": "user", "content": row["problem"] + '\n' + "Let's think step by step. Put your thinking process within <think> </think> and your final answer choice within \\boxed{}"},
            ],
            temperature=0.6,
            top_p=0.8,
            max_tokens=8192,
            # extra_body={
            #     "repetition_penalty": 1.05,
            # },
        )
        generated = response.choices[0].message.content
        # print(generated)
        return generated
        # print(int(extract_boxed_text(generated.split("</think>")[1])))
        # if "</think>" in generated:
        #     return int(extract_boxed_text(generated.split("</think>")[1]))
        # else:
        #     return None

    except Exception as e:
        print(f"Error: {e}")
        return None


def get_gemini_response(row):
    try:
        user_prompt = row["problem"]
        instruction_prompt = "Let's think step by step. Put your thinking process within <think> </think> and your final answer choice within \\boxed{}"
        # Generate the response
        # contents = [
        #     {"role": "user", "parts": [{"text": user_prompt + '\n' + instruction_prompt}]}
        # ]
        response = gemini_client.models.generate_content(
            model="gemini-2.5-pro-preview-03-25",
            contents=user_prompt + '\n' + instruction_prompt,
        )
        generated = response.text
        # Process output similar to OpenAI format
        # if "</think>" in generated:
        #     return int(extract_boxed_text(generated.split("</think>")[1]))
        # else:
        #     # If no </think> tag, try to extract directly
        #     return int(extract_boxed_text(generated))
        return generated

    except Exception as e:
        print(f"Error with Gemini: {e}")
        return None


def main():
    # Read the CSV file
    df = pd.read_parquet("publicqa_hard.parquet")
    df = df[50000:]
    # Shuffle the dataframe
    # df = df.sample(frac=1).reset_index(drop=True)
    print(f"Total rows: {len(df)}")
    for j in range(0, len(df), 8):
        print(f"Processing from row {j}")
        _df = df[j:j+8]
        i = 1
        while i <= 1:
            _df[f"# {i}"] = _df.parallel_apply(lambda x: get_openai_response(x), axis=1)
            i += 1
        try:
            if os.path.exists("_qwq32b.parquet"):
                existing_df = pd.read_parquet("_qwq32b.parquet")
                combined_df = pd.concat([existing_df, _df], ignore_index=True)
                combined_df.to_parquet("_qwq32b.parquet", index=False)
            else:
                _df.to_parquet("_qwq32b.parquet", index=False)
        except Exception as e:
            print(f"Error while saving parquet: {e}")
            continue

    # DEBUG
    # df = df[:1]
    # df.apply(lambda x: get_openai_response(x), axis=1)



if __name__ == "__main__":
    main()
