import os
import json

# Load the JSON file containing the extracted information
date_str = "2024-03-15"
with open(f"C:\\Users\\307164\\Desktop\\Huggingface_Paper_Extractor\\inference\\{date_str}\\inference.json", "r") as f:
    inference = json.load(f)

directory_path = f"C:\\Users\\307164\\Desktop\\Huggingface_Paper_Extractor\\summary_markdowns\\{date_str}"
if not os.path.exists(directory_path):
    os.makedirs(directory_path)

# Function to create a Markdown formatted string from the paper details
def create_markdown(entry):
    # Split the content to separate each paper if there are multiple papers per file
    papers = entry.split('\n\nTitle: ')[1:]  # Skip the first split part which is empty due to leading '\n'
    markdown_outputs = []

    for paper in papers:
        sections = paper.split('\n\n')  # Split each section
        title = sections[0].strip()  # First section is always the title
        content = '\n'.join([f"- {section.strip()}" for section in sections[1:]])  # Other sections
        markdown_template = (
            f"## {title}\n\n" +
            f"{content}\n\n" +
            "---\n\n"  # Separator between multiple papers
        )
        markdown_outputs.append(markdown_template)
    
    return '\n'.join(markdown_outputs)

# Loop through each item in the inference dictionary, format using Markdown, and save to a file
for link, content in inference.items():
    markdown_content = create_markdown(content)

    # Define the filename based on the link or some unique identifier
    filename = os.path.basename(link) + ".md"  # Adjust this for valid filenames
    file_path = os.path.join(directory_path, filename)

    # Write the formatted Markdown post to a file
    with open(file_path, "w", encoding="utf-8") as md_file:
        md_file.write(markdown_content)

    print(f"Markdown file saved: {file_path}")
