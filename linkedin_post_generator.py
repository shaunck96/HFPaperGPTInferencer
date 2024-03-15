from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.chat_models import ChatOpenAI
import tiktoken
import json
import os
import ast

with open("openai_config.json") as f:
    key = json.load(f)

os.environ["OPENAI_API_KEY"] = key["openai_key"]

def num_tokens_from_string(string: str, 
                           encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def split_into_chunks(text, 
                      chunk_size=8000):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

response_schemas = [
    ResponseSchema(
        name="Algorithmic Innovations",
        description="Insights on novel algorithms, optimizations, regularization methods, loss functions, and architectural modifications."
    ),
    ResponseSchema(
        name="Empirical Results",
        description="Insights on performance metrics, benchmark comparisons, analysis of results, and visualization of model outputs."
    ),
    ResponseSchema(
        name="Theoretical Analyses",
        description="Insights on convergence proofs, generalization bounds, complexity analysis, and theoretical guarantees."
    ),
    ResponseSchema(
        name="Applications and Use Cases",
        description="Insights on real-world applications, case studies, domain-specific challenges, and opportunities."
    ),
    ResponseSchema(
        name="Datasets and Data Preprocessing",
        description="Insights on dataset descriptions, data preprocessing techniques, data augmentation, and handling of imbalanced or noisy data."
    ),
    ResponseSchema(
        name="Ethical and Societal Implications",
        description="Insights on fairness, bias, discrimination, privacy concerns, and broader societal impacts."
    ),
    ResponseSchema(
        name="Future Directions and Open Problems",
        description="Insights on current limitations, future research directions, unresolved issues, and suggestions for improvement."
    )
]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

chat_model = ChatOpenAI(model_name="gpt-4", temperature=0.5, openai_api_key=key["openai_key"])

prompt = ChatPromptTemplate(
    messages=[
        HumanMessagePromptTemplate.from_template("""
            As an expert in distilling complex scientific research into digestible content, you are tasked with crafting a LinkedIn post that makes the key points from a research paper engaging and accessible to a mixed audience including industry professionals, academics, and laypeople. The summary should adhere to the following guidelines:
            
            - **Headline**: Create a compelling headline that encapsulates the essence and novelty of the research.
            
            - **Introduction**: Provide a brief overview of the research paper and its significance.
            
            - **Algorithmic Innovations**:
               - Insights on novel algorithms, optimizations, regularization methods, loss functions, and architectural modifications.
            
            - **Empirical Results**:
               - Insights on performance metrics, benchmark comparisons, analysis of results, and visualization of model outputs.
               
            - **Theoretical Analyses**:
               - Insights on convergence proofs, generalization bounds, complexity analysis, and theoretical guarantees.
            
            - **Applications and Use Cases**:
               - Insights on real-world applications, case studies, domain-specific challenges, and opportunities.
               
            - **Datasets and Data Preprocessing**:
               - Insights on dataset descriptions, data preprocessing techniques, data augmentation, and handling of imbalanced or noisy data.
               
            - **Ethical and Societal Implications**:
               - Insights on fairness, bias, discrimination, privacy concerns, and broader societal impacts.
            
            - **Future Directions and Open Problems**:
               - Insights on current limitations, future research directions, unresolved issues, and suggestions for improvement.
               
            - **Engagement**: End with a thought-provoking question, a suggestion for further discussion, or a call to action to encourage dialogue.
            
            Provide this engaging summary in at least 300 words based on the following details and formatting instructions:
            
            Research Paper Summary: {hist}
            URL: {link}
            
            Use the format instructions below to ensure your post is consistent, clear, and captivating:
            
            {format_instructions}
            
            Remember, the ultimate goal is to make the research understandable and compelling to a wide range of readers. Your summary should invite curiosity and conversation, making complex ideas approachable and intriguing. 
            Only use the provided context to generate the summary. Strictly adhere to the output format.
        """)
    ],
    input_variables=["hist", "link"],
    partial_variables={"format_instructions": format_instructions}
)

val_output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
val_format_instructions = val_output_parser.get_format_instructions()

validation_prompt = ChatPromptTemplate(
    messages=[
        HumanMessagePromptTemplate.from_template("""
        Ensure the LinkedIn post follows the prescribed structure and meets quality standards. Apply these validation steps:

        1. Confirm the output is structured as a dictionary with keys corresponding to each section of the post: 'headline', 'introduction', 'algorithmic_innovations', 'empirical_results', 'theoretical_analyses', 'applications_and_use_cases', 'datasets_and_data_preprocessing', 'ethical_and_societal_implications', 'future_directions_and_open_problems', and 'engagement'. Each key should map to a non-empty string value.
        2. Ensure the 'headline' is compelling and encapsulates the research's main message.
        3. Verify the 'introduction' offers a succinct yet informative overview of the research's significance.
        4. Check that each section ('algorithmic_innovations', 'empirical_results', etc.) contains relevant and insightful information.
        5. Make sure the 'engagement' section includes a thought-provoking question or a call to action to prompt reader interaction.
        6. Review the output for syntax errors or trailing commas that could interfere with JSON parsing. Ensure proper use of quotes, brackets, and commas.
        7. Attempt to parse the output as JSON (or as a Python dictionary). Confirm there are no parsing errors, indicating that the structure conforms to JSON/dictionary syntax standards.
        8. If all conditions are met, the output is correctly formatted and ready for use.

        Apply the above checks to the following output, ensuring it adheres to the provided structure and can be effectively parsed as JSON or a Python dictionary:

        {answer}

        Use these formatting guidelines to ensure the post is consistent, clear, and engaging:

        {val_format_instructions}
""")
    ],
    input_variables=["answer"],
    partial_variables={"format_instructions": val_format_instructions}
)

date_str = "2024-03-15"
with open("C:\\Users\\307164\\Desktop\\Huggingface_Paper_Extractor\\inference\\{}\\inference.json".format(date_str), "r") as f:
    inference = json.load(f)

for index in range(len(list(inference.keys()))):
    link = list(inference.keys())[index]
    input_to_llm = inference[link]

    #if num_tokens_from_string(input_to_llm, "gpt-3.5-turbo-16k") < 16000:
    _input = prompt.format_prompt(hist=input_to_llm, 
                                    link=link)
    output = chat_model(_input.to_messages())
    # Attempt to parse the output; handle parsing errors.
    try:
        # Attempt to convert the string representation of a dictionary back into a dictionary.
        output_dict = ast.literal_eval(output.content.replace("```json\n", "").replace("\n```", ""))
    except ValueError as e:
        # If there's a parsing error, log it and skip this iteration.
        _input = validation_prompt.format_prompt(answer=output)
        output = chat_model(_input.to_messages())
        output_dict = ast.literal_eval(output.content.replace("```json\n", "").replace("\n```", ""))

    output_dict['Summary'] = input_to_llm
    print(link)
    print("\n\n")
    # Create directory if it does not exist
    directory_path = f"linkedin_posts/{date_str}"
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    # Define the filename based on the link
    filename = link.split("\\")[-1]  # Changed from backslash to forward slash for URL
    file_path = f"{directory_path}/post_{filename}.json"  # Correct the file path

    # Write the output in JSON format
    with open(file_path, "w") as f:
        json.dump(output_dict, f)  # Use json.dump() to write the dict directly into the file

    print(f"File saved: {file_path}")
