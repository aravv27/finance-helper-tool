import streamlit as st
import os
from finance_crew import process_financial_query
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up page configuration
st.set_page_config(
    page_title="Finance Helper Tool",
    page_icon="💰",
    layout="wide"
)

# Page header
st.title("Finance Helper Tool")
st.markdown("Ask any finance-related question and get comprehensive answers with real-time data!")

# Sidebar for API key settings (optional, as they can be loaded from .env)
with st.sidebar:
    st.header("API Configuration")
    st.info("Use our API keys or provide your own (recommended for heavy usage)")
    
    # Toggle for using personal API keys
    use_personal_keys = st.checkbox("Use personal API keys", False)
    
    if use_personal_keys:
        perplexity_key = st.text_input(
            "Perplexity API Key",
            type="password",
            help="Get yours from https://pplx.ai"
        )
        
        groq_key = st.text_input(
            "Groq API Key", 
            type="password",
            help="Get yours from https://console.groq.com"
        )
        
        groq_model = st.text_input(
            "Groq Model Name",
            value="llama-3.3-70b-versatile",
            help="Default: llama-3.3-70b-versatile"
        )
    else:
        # Use environment variables when personal keys aren't enabled
        perplexity_key = os.getenv("PERPLEXITY_API_KEY", "")
        groq_key = os.getenv("GROQ_API_KEY", "")
        groq_model = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")

    show_debug = st.checkbox("Show debugging information", False, key="debug_checkbox")
    
    st.markdown("---")
    st.markdown("### Examples:")
    st.markdown("• What is the EMI for a home loan of ₹50 lakhs at 8.5% interest for 20 years?")
    st.markdown("• Compare FD and RD returns for ₹10,000 at 7% interest for 5 years")
    st.markdown("• How much would I accumulate with a SIP of ₹5,000 monthly at 12% return for 10 years?")

# Main query input
user_query = st.text_area(
    "Your Financial Query:",
    height=100,
    placeholder="Enter your financial question here..."
)

# Process button
submit_button = st.button("Submit", type="primary", use_container_width=True)

# Display results
if submit_button and user_query:
    with st.spinner("Processing your query... This may take a moment."):
        try:
            # Show intermediate steps (if in debug mode)
            if st.sidebar.checkbox("Show debugging information", False):
                st.subheader("Processing Steps:")
                col1, col2, col3 = st.columns(3)
                with col1:
                    with st.status("Web Research", state="running"):
                        st.write("Searching for relevant financial information...")
                with col2:
                    with st.status("Data Analysis", state="running"):
                        st.write("Extracting key data and performing calculations...")
                with col3:
                    with st.status("Report Generation", state="running"):
                        st.write("Creating your financial report...")
            
            # Process the query
            result = process_financial_query(
                query=user_query,
                perplexity_key=perplexity_key if perplexity_key else None,
                groq_key=groq_key if groq_key else None,
                groq_model=groq_model if groq_model else None
            )
            
            # Display the final result
            st.markdown("---")
            st.subheader("Your Financial Report")
            st.markdown(result)
            
            # If in debug mode, mark steps as complete
            if show_debug:  # Use the variable instead of repeating the checkbox
                col1.status("Web Research", state="complete")
                col2.status("Data Analysis", state="complete")
                col3.status("Report Generation", state="complete")
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.info("Please check your API keys and try again.")

# Footer
st.markdown("---")
st.caption("Finance Helper Tool - Powered by CrewAI, Perplexity, and Groq LLMs")