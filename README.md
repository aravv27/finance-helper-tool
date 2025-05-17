# Finance Helper Tool

An intelligent assistant that researches financial topics, extracts relevant data, performs calculations, and provides summarized reports to users.

## 🌟 Features

- **Intelligent Query Processing**: Submit any financial query or scenario
- **Real-time Web Research**: Fetches up-to-date information using Perplexity API
- **Data Extraction**: Identifies key financial data points from research
- **Financial Calculations**: Supports multiple calculation types:
  - Fixed Deposit (FD)
  - Recurring Deposit (RD)
  - Loan EMI
  - Mutual Fund SIP
  - Mutual Fund Lumpsum
- **Concise Reporting**: Generates easy-to-understand financial summaries

## 🛠️ Technology Stack

- **Backend**: Python
- **Frontend**: Streamlit
- **Multi-Agent Framework**: CrewAI
- **Web Search**: Perplexity API (PPLX API)
- **LLM for Extraction & Summarization**: Groq API (Llama 3)
- **Calculation Engine**: Custom Python functions

## 📋 Prerequisites

- Python 3.8+
- Perplexity API key
- Groq API key

## 🔧 Installation

1. Clone this repository:
git clone https://github.com/aravv27/finance-helper-tool.git
cd finance-helper-tool

2. Create a virtual environment:
python -m venv venv

source venv/bin/activate # On Windows: venv\Scripts\activate
3. Install dependencies:
pip install -r requirements.txt

4. Create a `.env` file in the project root and add your API keys:
PERPLEXITY_API_KEY="YOUR_PERPLEXITY_API_KEY_HERE"
GROQ_API_KEY="YOUR_GROQ_API_KEY_HERE"
GROQ_MODEL_NAME="YOUR_GROQ_LLAMA3_MODEL_HERE"

## 🚀 Usage
Run the Streamlit application:
streamlit run app.py

Then open your browser and go to the URL displayed in the terminal (typically http://localhost:8501).

## 📂 Project Structure

finance-helper-tool/
│
├── app.py # Streamlit application
├── finance_crew.py # CrewAI agent definitions and workflow
├── .env # Environment variables (API keys)
├── requirements.txt # Python dependencies
├── README.md # Project documentation
│
└── calculators/ # Financial calculation modules
    ├── fd_calculator.py # Fixed Deposit calculator
    ├── rd_calculator.py # Recurring Deposit calculator
    ├── loan_emi_calculator.py # Loan EMI calculator
    ├── mutual_fund_sip_calculator.py
    └── mutual_fund_lumpsum_calculator.py

## 🧩 How It Works

The application follows this workflow:

1. **User Input**: User submits a financial query via the Streamlit interface
2. **Information Retrieval**: An AI agent searches the web using the Perplexity API
3. **Data Extraction**: Another agent extracts key data from the search results
4. **Calculations**: If needed, financial calculations are performed
5. **Report Generation**: A final agent synthesizes all information into a coherent report
6. **Output**: The report is displayed to the user

## 🔒 Security Note

This application requires API keys. Never commit your `.env` file to version control.
