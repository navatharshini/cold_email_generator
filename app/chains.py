import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv

load_dotenv()

class Chain:
    def __init__(self):
        self.llm = ChatGroq(temperature=0, groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.1-8b-instant")

    def extract_jobs(self, cleaned_text):
        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}
            ### INSTRUCTION:
            The scraped text is from the career's page of a website.
            Your job is to extract the job postings and return them in JSON format containing the following keys: `role`, `experience`, `skills` and `description`.
            Only return the valid JSON.
            ### VALID JSON (NO PREAMBLE):
            """
        )
        chain_extract = prompt_extract | self.llm
        res = chain_extract.invoke(input={"page_data": cleaned_text})
        try:
            json_parser = JsonOutputParser()
            res = json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Context too big. Unable to parse jobs.")
        return res if isinstance(res, list) else [res]

    def write_mail(self, job, links):
        prompt_email = PromptTemplate.from_template(
    """
    ### JOB DESCRIPTION:
    {job_description}
    
    ### INSTRUCTION:
    You are Navatharshini, a second-year student at the University of Moratuwa, studying Computer Science and Engineering (CSE). You are actively searching for an internship in the field of software development. 
    Your job is to write a cold email to the hiring manager regarding the job mentioned above, highlighting your academic achievements, skills, and why you are the right fit for the internship role. 
    Also, mention any relevant projects or skills from your academic work or personal portfolio that align with the company's needs.
    
    Additionally, include any relevant links to showcase your skills or projects:
    {link_list}
    
    Remember, you are Navatharshini, a motivated and eager-to-learn student seeking a valuable internship experience. Do not provide a preamble.
    
    ### EMAIL (NO PREAMBLE):
    """
);
        chain_email = prompt_email | self.llm
        res = chain_email.invoke({"job_description": str(job), "link_list": links})
        return res.content

if __name__ == "__main__":
    print(os.getenv("GROQ_API_KEY"))