from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
import os
import requests
from typing import Optional, Dict, Any, List

# Import calculator tools
from calculators.fd_calculator import calculate_fd
from calculators.rd_calculator import calculate_rd
from calculators.loan_emi_calculator import calculate_emi
from calculators.mutual_fund_sip_calculator import calculate_mutual_fund_sip
from calculators.mutual_fund_lumpsum_calculator import calculate_mutual_fund_lumpsum

# Tool Definitions
class PerplexitySearchTool(BaseTool):
    name: str = "Perplexity Web Search"
    description: str = "Performs a web search using the Perplexity API to find relevant financial information."

    def _run(self, query: str) -> str:
        api_key = os.getenv("PERPLEXITY_API_KEY")
        if not api_key:
            return "Error: Perplexity API key not found."
        
        try:
            response = requests.post(
                "https://api.perplexity.ai/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "pplx-7b-online",
                    "messages": [{"role": "user", "content": f"Find information on: {query}"}],
                },
            )
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except requests.exceptions.RequestException as e:
            return f"Error calling Perplexity API: {e}"
        except KeyError:
            return f"Error parsing Perplexity API response. Response: {response.text}"
        except Exception as e:
            return f"An unexpected error occurred with Perplexity Search: {e}"

class GroqExtractorTool(BaseTool):
    name: str = "Groq Data Extractor"
    description: str = "Uses Groq LLM to extract structured data from text."

    def _run(self, text_input: str) -> str:
        api_key = os.getenv("GROQ_API_KEY")
        model_name = os.getenv("GROQ_MODEL_NAME", "llama3-70b-8192")
        if not api_key:
            return "Error: Groq API key not found."

        extraction_prompt = """
        Your task is to extract key financial data from the provided text.
        Look for:
        1. Financial terms (e.g., FD, SIP, EMI)
        2. Monetary values
        3. Interest rates
        4. Time periods/durations
        5. Any other relevant financial parameters

        Format your response as a JSON object with the following structure:
        {
            "identified_calculation_type": "FD|RD|LOAN_EMI|MUTUAL_FUND_SIP|MUTUAL_FUND_LUMPSUM|NONE",
            "parameters": {
                "param_name": value,
                ...
            },
            "extracted_data": {
                "key_point_1": "value/description",
                ...
            }
        }
        """

        try:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "messages": [
                    {"role": "system", "content": extraction_prompt},
                    {"role": "user", "content": text_input}
                ],
                "model": model_name
            }
            
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Error calling Groq API for data extraction: {e}"

class GroqSummarizerTool(BaseTool):
    name: str = "Groq Summarizer"
    description: str = "Uses Groq LLM to generate a concise summary from processed data."

    def _run(self, text_input: str) -> str:
        api_key = os.getenv("GROQ_API_KEY")
        model_name = os.getenv("GROQ_MODEL_NAME", "llama3-70b-8192")
        if not api_key:
            return "Error: Groq API key not found."

        summarization_prompt = """
        You are a financial advisor assistant. Your task is to create a concise, clear, and informative summary
        from the provided financial data and analysis. Focus on directly answering the user's original query
        while highlighting the most relevant information and insights.
        
        Structure your response in a user-friendly way with:
        1. A direct answer to the query
        2. Key data points and calculations presented clearly
        3. Brief contextual information if relevant
        4. Any important caveats or notes the user should be aware of
        
        Keep your summary professional but conversational, and prioritize accuracy over length.
        """

        try:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "messages": [
                    {"role": "system", "content": summarization_prompt},
                    {"role": "user", "content": text_input}
                ],
                "model": model_name
            }
            
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Error calling Groq API for summarization: {e}"

class FDCalculatorTool(BaseTool):
    name: str = "Fixed Deposit Calculator"
    description: str = "Calculates maturity amount and interest earned for a Fixed Deposit (FD)."

    def _run(self, principal: str, interest_rate: str, tenure: str, tenure_unit: str = "years") -> str:
        try:
            principal_amount = float(principal)
            rate = float(interest_rate)
            period = float(tenure)
            
            if tenure_unit.lower() in ['month', 'months']:
                period = period / 12
            elif tenure_unit.lower() in ['day', 'days']:
                period = period / 365
                
            result = calculate_fd(principal_amount, rate, period)
            return str(result)
        except Exception as e:
            return f"Error in FD calculation: {e}"

class RDCalculatorTool(BaseTool):
    name: str = "Recurring Deposit Calculator"
    description: str = "Calculates maturity amount and interest earned for a Recurring Deposit (RD)."

    def _run(self, monthly_installment: str, interest_rate: str, tenure_months: str) -> str:
        try:
            monthly_amount = float(monthly_installment)
            rate = float(interest_rate)
            months = int(tenure_months)
            
            result = calculate_rd(monthly_amount, rate, months)
            return str(result)
        except Exception as e:
            return f"Error in RD calculation: {e}"

class LoanEMICalculatorTool(BaseTool):
    name: str = "Loan EMI Calculator"
    description: str = "Calculates Equated Monthly Installment (EMI) for a loan."

    def _run(self, principal: str, interest_rate: str, tenure: str, tenure_unit: str = "years") -> str:
        try:
            loan_amount = float(principal)
            rate = float(interest_rate)
            period = float(tenure)
            
            if tenure_unit.lower() in ['year', 'years']:
                period = period * 12  # Convert years to months
                
            result = calculate_emi(loan_amount, rate, period)
            return str(result)
        except Exception as e:
            return f"Error in loan EMI calculation: {e}"

class MutualFundSIPCalculatorTool(BaseTool):
    name: str = "Mutual Fund SIP Calculator"
    description: str = "Calculates returns for Systematic Investment Plan (SIP) in mutual funds."

    def _run(self, monthly_amount: str, expected_return_rate: str, duration_years: str) -> str:
        try:
            amount = float(monthly_amount)
            rate = float(expected_return_rate)
            years = float(duration_years)
            
            result = calculate_mutual_fund_sip(amount, rate, years)
            return str(result)
        except Exception as e:
            return f"Error in mutual fund SIP calculation: {e}"

class MutualFundLumpsumCalculatorTool(BaseTool):
    name: str = "Mutual Fund Lumpsum Calculator"
    description: str = "Calculates returns for lumpsum investment in mutual funds."

    def _run(self, investment_amount: str, expected_return_rate: str, duration_years: str) -> str:
        try:
            amount = float(investment_amount)
            rate = float(expected_return_rate)
            years = float(duration_years)
            
            result = calculate_mutual_fund_lumpsum(amount, rate, years)
            return str(result)
        except Exception as e:
            return f"Error in mutual fund lumpsum calculation: {e}"

class FinancialCalculatorTool(BaseTool):
    name: str = "Financial Calculator"
    description: str = "Performs financial calculations based on extracted data and calculation type."

    def _run(self, extracted_data_json: str) -> str:
        import json
        
        try:
            data = json.loads(extracted_data_json)
            
            calc_type = data.get("identified_calculation_type", "NONE")
            params = data.get("parameters", {})
            
            if calc_type == "FD":
                principal = params.get("principal", 0)
                interest_rate = params.get("interest_rate", 0)
                tenure = params.get("tenure", 0)
                tenure_unit = params.get("tenure_unit", "years")
                
                fd_tool = FDCalculatorTool()
                return fd_tool._run(str(principal), str(interest_rate), str(tenure), tenure_unit)
                
            elif calc_type == "RD":
                monthly_installment = params.get("monthly_installment", 0)
                interest_rate = params.get("interest_rate", 0)
                tenure_months = params.get("tenure_months", 0)
                
                rd_tool = RDCalculatorTool()
                return rd_tool._run(str(monthly_installment), str(interest_rate), str(tenure_months))
                
            elif calc_type == "LOAN_EMI":
                principal = params.get("principal", 0)
                interest_rate = params.get("interest_rate", 0)
                tenure = params.get("tenure", 0)
                tenure_unit = params.get("tenure_unit", "years")
                
                emi_tool = LoanEMICalculatorTool()
                return emi_tool._run(str(principal), str(interest_rate), str(tenure), tenure_unit)
                
            elif calc_type == "MUTUAL_FUND_SIP":
                monthly_amount = params.get("monthly_amount", 0)
                expected_return_rate = params.get("expected_return_rate", 0)
                duration_years = params.get("duration_years", 0)
                
                sip_tool = MutualFundSIPCalculatorTool()
                return sip_tool._run(str(monthly_amount), str(expected_return_rate), str(duration_years))
                
            elif calc_type == "MUTUAL_FUND_LUMPSUM":
                investment_amount = params.get("investment_amount", 0)
                expected_return_rate = params.get("expected_return_rate", 0)
                duration_years = params.get("duration_years", 0)
                
                lumpsum_tool = MutualFundLumpsumCalculatorTool()
                return lumpsum_tool._run(str(investment_amount), str(expected_return_rate), str(duration_years))
                
            else:
                return "No financial calculation was identified or required for this query."
                
        except json.JSONDecodeError:
            return "Error: Invalid JSON format in extracted data."
        except Exception as e:
            return f"Error performing financial calculation: {e}"

# Agent Definitions
def setup_agents(pplx_llm,groq_llm):
    # Information Retriever Agent
    information_retriever_agent = Agent(
        role="Expert Web Researcher",
        goal="Fetch comprehensive and current web information based on the user's query using the Perplexity API.",
        backstory=(
            "You are a master of the internet, capable of finding the most relevant and up-to-date "
            "information on any financial topic. You use the Perplexity API to ensure accuracy and timeliness."
        ),
        tools=[PerplexitySearchTool()],
        verbose=True,
        allow_delegation=False,
        llm=pplx_llm,
    )

    # Data Processor & Analyst Agent
    data_processor_agent = Agent(
        role="Insightful Data Analyst and Calculator",
        goal="Extract structured data from web content, identify if calculations are needed, and perform them.",
        backstory=(
            "You are a meticulous analyst with a knack for numbers. You can dissect textual information "
            "to find key financial data points and perform precise calculations using specialized tools."
        ),
        tools=[
            GroqExtractorTool(),
            FinancialCalculatorTool()
        ],
        verbose=True,
        allow_delegation=False,
        llm=groq_llm,
    )

    # Report Synthesizer Agent
    report_synthesizer_agent = Agent(
        role="Concise Report Writer",
        goal="Generate a clear, user-friendly summary from processed data and calculations.",
        backstory=(
            "You are a skilled communicator who can synthesize complex financial information into "
            "an easy-to-understand report that directly answers the user's query."
        ),
        tools=[GroqSummarizerTool()],
        verbose=True,
        allow_delegation=False,
        llm=groq_llm,
    )
    
    return information_retriever_agent, data_processor_agent, report_synthesizer_agent

class FinanceCrewRunner:
    def __init__(self, query, perplexity_key=None, groq_key=None,pplx_model="sonar", groq_model=None):
        self.query = query
        
        # Set API keys as environment variables if provided
        if perplexity_key:
            os.environ["PERPLEXITY_API_KEY"] = perplexity_key
        if groq_key:
            os.environ["GROQ_API_KEY"] = groq_key
        if groq_model:
            os.environ["GROQ_MODEL_NAME"] = groq_model
        
        self.pplx_llm = LLM(
            model=pplx_model,
            api_base="https://api.perplexity.ai/",
            api_key=perplexity_key
        )
        
        # Initialize the Groq LLM
        self.groq_llm = LLM(
            model=groq_model,
            api_base="https://api.groq.com/openai/v1",
            api_key=groq_key
        )
            
        # Set up agents
        self.retriever_agent, self.processor_agent, self.synthesizer_agent = setup_agents(self.pplx_llm, self.groq_llm)

    def run(self):
        # Define tasks for the workflow
        retrieval_task = Task(
            description=f"Research the user's query: '{self.query}'. Plan and execute a web search using the Perplexity API tool to gather relevant financial information.",
            expected_output="Raw text content from the web search, suitable for data extraction.",
            agent=self.retriever_agent,
        )

        processing_task = Task(
            description="Process the raw text from the web search. Extract key financial entities and parameters. Determine if calculations are necessary and perform them if needed.",
            expected_output="Structured data including extracted information and any calculation results.",
            agent=self.processor_agent,
            context=[retrieval_task]
        )

        summarization_task = Task(
            description="Generate a comprehensive yet concise summary that directly addresses the user's initial query based on the processed data and calculations.",
            expected_output="Final summary text with clear financial insights and answers.",
            agent=self.synthesizer_agent,
            context=[processing_task]
        )

        # Create and run the crew
        financial_crew = Crew(
            agents=[self.retriever_agent, self.processor_agent, self.synthesizer_agent],
            tasks=[retrieval_task, processing_task, summarization_task],
            process=Process.sequential,
            verbose=True
        )

        result = financial_crew.kickoff()
        return result

# Function to use in app.py
def process_financial_query(query, perplexity_key=None, groq_key=None, groq_model=None):
    """
    Process a financial query through the CrewAI system.
    
    Args:
        query (str): User's financial question or query
        perplexity_key (str, optional): Perplexity API key
        groq_key (str, optional): Groq API key
        groq_model (str, optional): Groq model name
        
    Returns:
        str: The final summarized answer to the user's query
    """
    crew_runner = FinanceCrewRunner(
        query=query,
        perplexity_key=perplexity_key,
        groq_key=groq_key,
        groq_model=groq_model
    )
    
    result = crew_runner.run()
    return result