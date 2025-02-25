from dotenv import load_dotenv
import json
from mistralai import Mistral
import os

load_dotenv(".env")

with open("data/job_listings.json", 'r') as file:
    data = json.load(file)
job_ids = list(data.keys())

api_key = os.getenv("MISTRAL_API_KEY")
client = Mistral(api_key)
# Get job descriptions for each job ID, extract only main information
job_descriptions = {}

# Open a JSON file to write the responses
with open('job_descriptions.json', 'w') as json_file:
    json_file.write('[\n')  # Start of JSON array

    for i, job_id in enumerate(job_ids):
        prompt = [{
            "role": "system",
            "content": """You are a tool designed to extract and summarize key information from job descriptions. Your output will be used for embeddings, so focus on precision and conciseness. Use bullet points instead of complete sentences where appropriate. Make it short
                    Write the result in french, don't use special character like: \u00e9 or other.
                    For each job description, extract and organize the following information:
                    Job Title:
                    Extract the job title.
                    Job Summary:
                    Summarize the job description in 3-4 concise bullet points.
                    Required Skills:
                    List the key technical and soft skills required for the job.
                    Responsibilities:
                    Highlight the main tasks and responsibilities associated with the role.
                    Qualifications:
                    List the educational background, certifications, and experience required.
                    Company Value Proposition:
                    Highlight any benefits, perks, or unique offerings mentioned by the company.
                    Company Culture:
                    Describe the company culture, if mentioned (e.g., work environment, values, or mission).
                    If any of the above information is not available, skip that section. Prioritize clarity and brevity in your output.
                    """
        },
        {
            "role": "user",
            "content": "Here is a job description: " + data[job_id]
        }]

        resp = client.chat.complete(
            model="mistral-large-latest",
            messages=prompt
        )

        job_descriptions[job_id] = resp.choices[0].message.content

        # Write each response to the JSON file
        json.dump({job_id: resp.choices[0].message.content}, json_file)
        if i < len(job_ids) - 1:  # Add a comma after each item except the last one
            json_file.write(',\n')
            
        print(f"Job {i+1} of {len(job_ids)} processed")

    json_file.write('\n]')