import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.dates as mdates
import re
import logging
import os # Import os for file path checking
import json # Import json for parsing LLM output (global import)
import calendar
import plotly.express as px
from difflib import get_close_matches





# --- Library Imports with Error Handling ---
try:
    import dateparser
except ImportError:
    st.error("❌ The 'dateparser' library is not installed. Please install it using `pip install dateparser` in your terminal.")
    st.stop()
except Exception as e:
    st.error(f"❌ An unexpected error occurred during dateparser library import: {e}. Please check your Python environment.")
    st.stop()

try:
    import plotly.express as px
except ImportError:
    st.error("❌ The 'plotly' library is not installed. Please install it using `pip install plotly` in your terminal.")
    st.stop()
except Exception as e:
    st.error(f"❌ An unexpected error occurred during plotly library import: {e}. Please check your Python environment.")
    st.stop()

try:
    from fuzzywuzzy import fuzz
    from fuzzywuzzy import process 
except ImportError:
    st.error("❌ The 'fuzzywuzzy' library is not installed. Please install it using `pip install fuzzywuzzy` in your terminal.")
    st.stop()
except Exception as e:
    st.error(f"❌ An unexpected error occurred during fuzzywuzzy library import: {e}. Pleas...")

try:
    from openai import OpenAI, APIConnectionError, RateLimitError, AuthenticationError
except ImportError:
    st.error("❌ The 'openai' library is not installed. Please install it using `pip install openai` in your terminal.")
    st.stop()
except Exception as e:
    st.error(f"❌ An unexpected error occurred during OpenAI library import: {e}. Please check your Python environment.")
    st.stop()

# --- Inline Configuration (if config.py is not used) ---
# Check if config.py exists, otherwise use inline class
try:
    import config
except ImportError:
    # Define an inline Config class if config.py is not found
    class Config:
        OPENAI_MODEL = "gpt-3.5-turbo" # Or "gpt-4" for better performance, but higher cost
        OPENAI_CONTEXT_MESSAGES = 5 # Number of previous messages to send to OpenAI for context
        FUZZY_MATCH_THRESHOLD = 75 # Confidence score for fuzzy matching entities (0-100)
        CONFIDENCE_THRESHOLD = 0.6 # Confidence score for SentenceTransformer intent matching (0-1)
        BASE_INTENTS = [
            "total spend by brand", "spend by company", "spend of a brand",
            "top sectors by spend", "top brands by spend",
            "compare online offline spend",
            "monthly trend for brand",
            "highest spend day",
            "compare two months",
            "predict next week spend", "forecast spend spend next week",
            "compare two brands",
            "list all brands", "list all sectors",
            "general question about marketing",
            "what is marketing spend analysis",
            "tell me about AI in marketing.",
            "calculate sum for column",
            "what is the average of column",
            "find the maximum value in column",
            "show me a histogram of column",
            "list unique values in column",
            "how many unique values in column",
            "complex data query", # New intent for complex, multi-filter queries
            # ✅ 5. INTENT Mapping Adjustments
            "top 3 sectors",
            "top 5 sectors",
            "highest offline spend by a brand",
            "daily trend for brand",
        ]
    config = Config()
# --- End Inline Configuration ---

# === Column resolver helper ===
def _norm(s: str) -> str:
    return str(s).strip().lower().replace("_", " ").replace("-", " ")

def resolve_col(df, logical_col: str):
    # case/space/underscore-insensitive lookup + aliases
    idx = {_norm(c): c for c in df.columns}
    want = _norm(logical_col)
    if want in idx:
        return idx[want]

    aliases = {
        "amount": {"spend", "cost", "value", "amount usd", "total spend"},
        "sector": {"industry"},
        "brand": {"advertiser"},
        "channel": {"placement", "media channel"},
        "source": {"platform", "publisher"},
        "country": {"market"},
        "category": {"sub category", "subcategory"},
        "date": {"transaction date", "day"},
        "month": {"month name", "month str", "monthname"},
        "year": {"yr", "fiscal year"},
    }

    # try aliases for the target
    for alt in [logical_col, *aliases.get(want, set())]:
        nalt = _norm(alt)
        if nalt in idx:
            return idx[nalt]
    return None


# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Kinesso Chatbot", page_icon=":bar_chart:", layout="centered")

# Inject custom CSS for font
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
    
    /* Apply Inter font to all text elements within Streamlit's main content area */
    html, body, [class*="st-"] { /* General selector for Streamlit elements */
        font-family: 'Inter', sans-serif !important;
    }
    
    /* More specific targeting for chat messages and markdown text */
    div[data-testid="stChatMessage"],
    div[data-testid="stChatMessage"] p,
    div[data-testid="stMarkdownContainer"],
    div[data-testid="stMarkdownContainer"] p,
    div[data-testid="stText"] {
        font-family: 'Inter', sans-serif !important;
    }

    /* Ensure inputs and selectboxes also use Inter */
    .stTextInput > div > div > input,
    .stSelectbox > div > div > div > div {
        font-family: 'Inter', sans-serif !important;
    }

    /* Adjust for potential issues with code blocks or pre-formatted text if they don't inherit */
    pre, code {
        font-family: 'Inter', monospace !important; /* Keep monospace for code, but ensure Inter is primary if possible */
    }

    </style>
    """,
    unsafe_allow_html=True
)

# Initialize OpenAI Client using Streamlit Secrets
try:
    # In a Canvas environment, the API key is provided automatically if left empty
    openai_api_key = "" 
    client = OpenAI(api_key=openai_api_key)
except KeyError:
    st.error("❌ OpenAI API key not found. Please add `OPENAI_API_KEY` to your `.streamlit/secrets.toml` file.")
    st.stop()
except Exception as e:
    st.error(f"❌ Error initializing OpenAI client: {e}. Please check your API key and connection.")
    st.stop()

# --- OpenAI General Response Function ---
def get_openai_response(query, conversation_history):
    """
    Gets a general response from OpenAI's GPT model for queries not handled by specific tools.
    Includes a portion of the conversation history for context.
    """
    logger.info(f"Calling OpenAI for general response with query: {query}")
    messages = [{"role": "system", "content": "You are a helpful assistant. Answer the user's questions concisely and politely. If the question is about marketing spend data, and you don't have a specific tool to answer it, suggest what kind of data you can analyze or what questions you can answer."}]
    
    # Add a limited portion of conversation history for context
    context_messages_limit = getattr(config, 'OPENAI_CONTEXT_MESSAGES', 5) # Default to 5 if not in config
    start_index = max(0, len(conversation_history) - context_messages_limit)
    for msg in conversation_history[start_index:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    
    messages.append({"role": "user", "content": query})

    try:
        response = client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=200
        )
        return response.choices[0].message.content
    except APIConnectionError as e:
        logger.error(f"OpenAI API Connection Error: {e}")
        return "I'm sorry, I couldn't connect to the AI at the moment. Please check your internet connection or try again later."
    except RateLimitError:
        logger.error("OpenAI API Rate Limit Exceeded.")
        return "I'm experiencing high demand right now. Please wait a moment and try again."
    except AuthenticationError:
        logger.error("OpenAI API Authentication Failed. Check API key.")
        return "I'm unable to authenticate with the AI service. Please ensure your OpenAI API key is correctly configured."
    except Exception as e:
        logger.error(f"General Error calling OpenAI API: {e}", exc_info=True)
        return f"I encountered an unexpected error: {e}. Please try again."

# --- Configuration for Data Columns ---
# CORE and OPTIONAL columns (all internal names are lowercase)
CORE_REQUIRED_COLUMNS = ["Amount", "Month"] 
OPTIONAL_ANALYTICAL_COLUMNS = ["Source", "Sector", "Category", "Product", "Media Agency", "Producer", "Brand", "Country", "Media", "Channel"]
ALL_INTERNAL_COLUMNS = CORE_REQUIRED_COLUMNS + OPTIONAL_ANALYTICAL_COLUMNS

# LLM-based Column Mapping Function
def get_llm_column_mapping(df_columns):
    """
    Uses OpenAI's GPT to suggest column mappings based on semantic understanding.
    """
    logger.info("Attempting to get LLM-based column mapping...")
    
    all_internal_columns = CORE_REQUIRED_COLUMNS + OPTIONAL_ANALYTICAL_COLUMNS

    prompt_messages = [
        {"role": "system", "content": f"""You are a highly intelligent and accurate data assistant. Your task is to map user-provided column names to a set of predefined internal column names based on their semantic meaning.
        
        Here are the predefined internal column names and their expected meaning:
        - "Amount": The monetary value, spend, cost, revenue, sales, budget, actuals, etc. This is a CORE REQUIRED column.
        - "Month": The date or period of the transaction or record. This is a CORE REQUIRED column.
        - "Brand": The specific brand, company, advertiser, or product line associated with the spend. This is an OPTIONAL ANALYTICAL column.
        - "Sector": The industry sector, category, business unit, or market segment. This is an OPTIONAL ANALYTICAL column.
        - "Source": The channel, platform, media, or origin of the spend (e.g., 'Online', 'Offline', 'Digital', 'Traditional'). This is an OPTIONAL ANALYTICAL column.
        - "Category": The product category or service type.
        - "Product": The specific product name or SKU.
        - "Media": The type of media used (e.g., 'Social Media', 'Print').
        - "Agency": The advertising or marketing agency involved.
        - "Producer": The content producer or creator.
        - "Country": The country where the spend occurred.
        - "Channel": The specific channel within a media type (e.g., 'Facebook', 'Instagram' for 'Social Media').

        Given a list of actual column names from a user's dataset, identify the best semantic match for each internal column.
        If an internal column cannot be confidently mapped to any of the provided actual columns, or if it's an OPTIONAL ANALYTICAL column that doesn't have a clear match, you can map it to 'None'.
        You MUST provide a mapping for all CORE REQUIRED columns. If you cannot find a confident match for a CORE REQUIRED column, still output 'None' for it, but the system will prompt the user to map it manually.
        Provide your response as a JSON object where keys are the internal column names and values are the mapped actual column names from the provided list.
        Ensure the mapped actual column names exactly match one of the provided `actual_columns`.
        The order of keys in the output JSON does not matter.

        Example 1:
        Internal Columns: {all_internal_columns}
        Actual Columns: ["Date of Sale", "Product ID", "Sales Value", "Company", "Industry Group"]
        Output:
        {{"Amount": "Sales Value", "Month": "Date of Sale", "Brand": "Company", "Sector": "Industry Group", "Source": None, "Category": None, "Product": None, "Media": None, "Agency": None, "Producer": None, "Country": None, "Channel": None}}

        Example 2:
        Internal Columns: {all_internal_columns}
        Actual Columns: ["TransactionDate", "Cost"]
        Output:
        {{"Amount": "Cost", "Month": "TransactionDate", "Brand": None, "Sector": None, "Source": None, "Category": None, "Product": None, "Media": None, "Agency": None, "Producer": None, "Country": None, "Channel": None}}
        """},
        {"role": "user", "content": f"Internal Columns: {all_internal_columns}\nActual Columns: {df_columns}\nOutput:"}
    ]

    try:
        response = client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=prompt_messages,
            temperature=0.0,
            max_tokens=250,
            response_format={"type": "json_object"}
        )
        
        llm_output = response.choices[0].message.content
        logger.info(f"LLM raw output for column mapping: {llm_output}")
        
        suggested_mapping = json.loads(llm_output)
        
        validated_mapping = {}
        for internal_col, user_col in suggested_mapping.items():
            if user_col and user_col in df_columns:
                validated_mapping[internal_col] = user_col
            else:
                validated_mapping[internal_col] = None
        
        return validated_mapping

    except json.JSONDecodeError:
        logger.error("LLM did not return valid JSON for column mapping.")
        return None
    except APIConnectionError as e:
        logger.error(f"OpenAI API Connection Error during column mapping: {e}")
        st.warning("Could not connect to AI for smart column suggestions. Please check your internet connection or OpenAI API status.")
        return None
    except RateLimitError:
        logger.error("OpenAI API rate limit reached for column mapping. Please wait and try again.")
        return None
    except AuthenticationError:
        logger.error("OpenAI API Authentication Failed during column mapping. Check API key.")
        st.error("Authentication failed for OpenAI API. Cannot provide smart column suggestions. Please check your API key.")
        return None
    except Exception as e:
        logger.error(f"General Error calling OpenAI API for column mapping: {e}", exc_info=True)
        st.warning(f"An unexpected error occurred while getting smart column suggestions: {e}")
        return None

from context_parser import infer_contextual_filters


# Load the SentenceTransformer model once and cache it
@st.cache_resource
def load_sentence_transformer_model():
    """
    Loads the SentenceTransformer model and caches it for efficient reuse.
    """
    logger.info("Loading SentenceTransformer model...")
    try:
        model_name = 'all-MiniLM-L6-v2'
        model = SentenceTransformer(model_name)
        logger.info(f"SentenceTransformer model '{model_name}' loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Error loading SentenceTransformer model: {e}", exc_info=True)
        st.error(f"❌ Failed to load NLP model. Please check your internet connection or try again later. Error: {e}")
        return None

model = load_sentence_transformer_model()

# Use session state for column mapping and data loading status
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False
if "raw_df" not in st.session_state:
    st.session_state.raw_df = None
if "column_mapping" not in st.session_state:
    st.session_state.column_mapping = {col: None for col in ALL_INTERNAL_COLUMNS}
if "data_mapped" not in st.session_state:
    st.session_state.data_mapped = False
if "llm_mapping_attempted" not in st.session_state:
    st.session_state.llm_mapping_attempted = False
if "processed_df" not in st.session_state:
    st.session_state.processed_df = None
if "skip_mapping_ui" not in st.session_state:
    st.session_state.skip_mapping_ui = False
if "sidebar_filters" not in st.session_state:
    st.session_state.sidebar_filters = {}
if "current_filtered_df" not in st.session_state: # New: Store the currently filtered DataFrame
    st.session_state.current_filtered_df = None


@st.cache_data
def load_and_map_data(uploaded_file_obj, column_mapping_dict): # Renamed parameters to avoid confusion
    """
    Loads data from the uploaded Excel file and renames columns based on the mapping.
    Performs type conversions and adds derived columns.
    """
    logger.info(f"Attempting to load and map data from {uploaded_file_obj.name}")
    try:
        df_raw = pd.read_excel(uploaded_file_obj)
        
        # Create a copy to avoid modifying the original raw_df in session state directly
        processed_df = df_raw.copy() 

        # Clean and convert 'Month' to datetime
        if 'Month' in processed_df.columns:
            processed_df['Month'] = pd.to_datetime(processed_df['Month'], errors='coerce')
            processed_df['Month_str'] = processed_df['Month'].dt.strftime('%B')  # e.g., 'April'

        # Rename columns based on the user's confirmed mapping
        renamed_columns = {}
        for internal_col, user_col in column_mapping_dict.items():
            if user_col and user_col in processed_df.columns:
                renamed_columns[user_col] = internal_col
        
        processed_df = processed_df.rename(columns=renamed_columns)

        # Ensure CORE_REQUIRED_COLUMNS exist after mapping
        missing_core_columns = [col for col in CORE_REQUIRED_COLUMNS if col not in processed_df.columns]
        if missing_core_columns:
            raise ValueError(f"Missing core required columns after mapping: {', '.join(missing_core_columns)}. "
                             f"Please ensure '{' and '.join(CORE_REQUIRED_COLUMNS)}' are correctly mapped.")

        # Clean and convert 'Amount' column
        if "Amount" in processed_df.columns:
            processed_df["Amount"] = processed_df["Amount"].astype(str).str.replace(r'[\\$, -]', '', regex=True)
            processed_df["Amount"] = pd.to_numeric(processed_df["Amount"], errors='coerce')
            processed_df["Amount"] = processed_df["Amount"].fillna(0)

        # Convert 'Month' column to datetime
        if "Month" in processed_df.columns:
            processed_df["Month"] = pd.to_datetime(processed_df["Month"], errors='coerce')
            
            original_rows = len(processed_df)
            processed_df.dropna(subset=["Month"], inplace=True)
            dropped_rows = original_rows - len(processed_df)
            if dropped_rows > 0:
                st.warning(f"⚠️ Dropped {dropped_rows} rows due to invalid or unparseable dates in the 'Month' column.")
            
            if processed_df.empty:
                raise ValueError("No valid data entries found after cleaning. Please check 'Month' and 'Amount' columns.")

            # Add derived date columns
            processed_df["Year"] = processed_df["Month"].dt.year
            processed_df["Day"] = processed_df["Month"].dt.date # Store as date object for daily grouping
            processed_df["Weekday"] = processed_df["Month"].dt.day_name()
        
        logger.info("Data loaded and cleaned successfully.")
        return processed_df
    except Exception as e:
        logger.error(f"Error loading or cleaning data: {e}", exc_info=True)
        st.error(f"❌ Error processing your Excel file: {e}. "
                 f"Please ensure it's a valid Excel file and all required columns are correctly mapped and formatted.")
        return None # Return None to indicate failure

# Add this at the top with other configurations
COMMON_QUERY_WORDS = {
    "spend", "by", "on", "of", "for", "in", "what", "how", "much", "total", 
    "average", "compare", "show", "list", "top", "bottom", "highest", "lowest",
    "vs", "and", "or", "a", "an", "the", "with", "from", "to", "through", "over", "under",
    "daily", "monthly", "trend", "next", "week", "forecast", "predict", "vs" 
}

# New helper function for safer fuzzy extraction
def safe_fuzzy_extract_one(query_phrase, entity_list_original_case, threshold=config.FUZZY_MATCH_THRESHOLD):
    """
    Performs fuzzy extraction, prioritizing exact matches and using a stricter scorer
    for short query phrases. Returns the original-cased matched entity.
    """
    logger.debug(f"safe_fuzzy_extract_one: Query Phrase: '{query_phrase}', Entity List Sample: {entity_list_original_case[:5]}, Threshold: {threshold}")
    st.sidebar.write(f"DEBUG - safe_fuzzy_extract_one: Query Phrase: '{query_phrase}', Entity List Sample: {entity_list_original_case[:5]}, Threshold: {threshold}")

    valid_entities_lower = [str(e).strip().lower() for e in entity_list_original_case if pd.notna(e) and str(e).strip()]
    if not valid_entities_lower:
        logger.debug("safe_fuzzy_extract_one: No valid entities in list.")
        st.sidebar.write("DEBUG - safe_fuzzy_extract_one: No valid entities.")
        return None, 0

    best_match_entity_lower = None
    best_score = -1

    current_scorer = fuzz.token_set_ratio
    if len(query_phrase.split()) == 1 and len(query_phrase) <= 6: # For single, short words like 'online', 'bmw'
        current_scorer = fuzz.ratio # Use simple ratio for exactness
    
    logger.debug(f"safe_fuzzy_extract_one: Using scorer: {current_scorer.__name__}")
    st.sidebar.write(f"DEBUG - safe_fuzzy_extract_one: Using scorer: {current_scorer.__name__}")


    for entity_lower in valid_entities_lower:
        score = current_scorer(query_phrase.lower(), entity_lower)
        
        # --- NEW STRICTER CHECK FOR SHORT QUERY PHRASES (especially prepositions) ---
        # If query_phrase is very short (e.g., "by", "on", "in")
        # And the matched entity is much longer, and the score is high, it's likely a false positive.
        # This prevents "by" matching "La Perle by Dragone" with a high score.
        # Also helps with "online" not matching "Online Banking" if it's not an exact match.
        if len(query_phrase) <= 3: # Very short single word (e.g., "by", "on", "in")
            # If the score is high (e.g., >= 90) but the matched entity is significantly longer (e.g., > 3x the length),
            # and the query_phrase is not a distinct whole word within the entity, it's a weak match.
            if score >= 90 and len(entity_lower) > len(query_phrase) * 3: # Multiplier can be tuned (e.g., 3-5)
                # Check if query_phrase appears as a whole word in entity_lower
                if not re.search(r'\b' + re.escape(query_phrase.lower()) + r'\b', entity_lower):
                    logger.debug(f"safe_fuzzy_extract_one: Skipping (short query, long entity, no word boundary match): '{query_phrase}' vs '{entity_lower}' (Score: {score})")
                    st.sidebar.write(f"DEBUG - safe_fuzzy_extract_one: Skipping (short query, long entity, no word boundary match): '{query_phrase}' vs '{entity_lower}' (Score: {score})")
                    continue # Skip this match, it's likely a false positive

        # This also prevents "bmw" from matching "bmw group" with 100 if we want "bmw" to be a standalone brand.
        # However, "BMW Group" is a valid Producer. The issue is the *column type* ambiguity.
        # The column priority in extract_filters_from_query should handle 'BMW' as Brand vs 'BMW Group' as Producer.

        if score > best_score:
            best_score = score
            best_match_entity_lower = entity_lower

    if best_match_entity_lower and best_score >= threshold:
        # Find the original-cased entity from the original list
        original_cased_entity = None
        for entity_orig in entity_list_original_case:
            if pd.notna(entity_orig) and str(entity_orig).strip().lower() == best_match_entity_lower:
                original_cased_entity = str(entity_orig).strip()
                break
        
        if original_cased_entity:
            logger.debug(f"safe_fuzzy_extract_one: Raw Match: '{original_cased_entity}', Score: {best_score}")
            st.sidebar.write(f"DEBUG - safe_fuzzy_extract_one: Raw Match: '{original_cased_entity}', Score: {best_score}")
            logger.debug(f"safe_fuzzy_extract_one: Returning Match: '{original_cased_entity}', Score: {best_score} (>= Threshold)")
            st.sidebar.write(f"DEBUG - safe_fuzzy_extract_one: Returning Match: '{original_cased_entity}', Score: {best_score} (>= Threshold)")
            return original_cased_entity, best_score
        else:
            logger.debug(f"safe_fuzzy_extract_one: Matched lowercased entity '{best_match_entity_lower}' but couldn't find original cased version. Returning None.")
            st.sidebar.write(f"DEBUG - safe_fuzzy_extract_one: Matched lowercased entity '{best_match_entity_lower}' but couldn't find original cased version. Returning None.")
            return None, 0
    
    logger.debug("safe_fuzzy_extract_one: No match found.")
    st.sidebar.write("DEBUG - safe_fuzzy_extract_one: No match found.")
    return None, 0

# --- Core Data Analysis Functions ---
# These functions directly perform analysis on the DataFrame

def find_entity_fuzzy(query, entity_list, cutoff=0.6):
    """Attempts to fuzzy match an entity (like a brand or channel) from the query."""
    matches = get_close_matches(query.lower(), [e.lower() for e in entity_list if isinstance(e, str)], n=1, cutoff=cutoff)
    if matches:
        for entity in entity_list:
            if entity.lower() == matches[0]:
                return entity, True
    return None, False

def get_total_spend_by_column(column_name, value):
    """
    Calculates total spend for a given value within a specified categorical column.
    E.g., total spend for 'Instagram' in 'Channel' column, or 'Retail' in 'Sector' column.
    """
    data_to_analyze = st.session_state.current_filtered_df if st.session_state.current_filtered_df is not None and not st.session_state.current_filtered_df.empty else st.session_state.processed_df

    if data_to_analyze is None or data_to_analyze.empty:
        return "No data available to perform this analysis. Please upload your data first."
    if column_name not in data_to_analyze.columns:
        return f"Column '{column_name}' not available in the dataset for this analysis."
    
    temp_df = data_to_analyze.copy()
    
    # Ensure the column is treated as string for comparison
    temp_df = temp_df[temp_df[column_name].astype(str).str.lower() == str(value).lower()]
    
    if temp_df.empty:
        return f"No data found for {column_name.title()}: {value.title()}."

    total_spend = temp_df["Amount"].sum()
    return f"The total spend for {column_name.title()}: **{value.title()}** was **${total_spend:,.2f}**."

# ✅ 1. Fix: Top 3 or 5 Sectors by Spend
def top_n_sectors_by_spend(df, top_n=3):
    if "Sector" not in df.columns or "Amount" not in df.columns:
        return "Missing 'Sector' or 'Amount' columns in data."

    sector_spend = df.groupby("Sector")["Amount"].sum().sort_values(ascending=False).head(top_n)
    result = f"Top {top_n} Sectors by Spend:\n"
    for i, (sector, amount) in enumerate(sector_spend.items(), start=1):
        result += f"{i}. {sector} — ${amount:,.2f}\n"
    return result

def get_top_sectors(n=3, ascending=False, use_filters=False, show_chart=True):
    """
    Returns a sentence AND (optionally) shows a bar chart of top/bottom N sectors by spend.
    """
    import pandas as pd, altair as alt, streamlit as st

    df = st.session_state.get("processed_df")
    if df is None or df.empty:
        return "No data available."

    sector_col = resolve_col(df, "Sector") or resolve_col(df, "Industry")
    amount_col = resolve_col(df, "Amount")
    if not sector_col or not amount_col:
        return "Required columns not found (need Sector/Amount)."

    if use_filters:
        filters = st.session_state.get("filters") or st.session_state.get("active_filters") or {}
        try:
            df = apply_dynamic_filters(df, filters or {})
        except Exception:
            pass
        if df is None or df.empty:
            return "No data available after filtering."

    df[amount_col] = pd.to_numeric(df[amount_col], errors="coerce").fillna(0)

    grouped = (
        df.groupby(sector_col, dropna=False)[amount_col]
          .sum()
          .reset_index()
          .rename(columns={sector_col: "Sector", amount_col: "Total Spend"})
    )

    if grouped.empty:
        return "No sectors found to rank."

    nz = grouped[grouped["Total Spend"] > 0]
    data = nz if not nz.empty else grouped
    top_n_data = (data.sort_values("Total Spend", ascending=ascending)
                       .head(int(n)))

    if top_n_data.empty:
        return "No sectors found after aggregation."

    # ---- chart (Altair) ----
    if show_chart:
        order = alt.SortField(field="Total Spend", order="ascending" if ascending else "descending")
        chart = (
            alt.Chart(top_n_data)
            .mark_bar()
            .encode(
                x=alt.X("Total Spend:Q", title="Total Spend"),
                y=alt.Y("Sector:N", sort=order, title="Sector"),
                tooltip=["Sector:N", alt.Tooltip("Total Spend:Q", format=",.2f")]
            )
            .properties(height=max(180, 30 * len(top_n_data)), width="container")
        )
        st.altair_chart(chart, use_container_width=True)

    direction = "lowest" if ascending else "top"
    lines = [f"Here are the {direction} {n} sectors by total spend:"]
    for _, row in top_n_data.iterrows():
        lines.append(f"- **{row['Sector']}**: ${row['Total Spend']:,.2f}")
    return "\n".join(lines)


def get_top_n_by_column(column_name, n=5, ascending=False, use_filters=False, show_chart=True):
    """
    Returns a sentence AND (optionally) shows a bar chart of top/bottom N items for a given column.
    """
    import pandas as pd, altair as alt, streamlit as st

    df = st.session_state.get("processed_df")
    if df is None or df.empty:
        return "No data available to perform this analysis. Please upload your data first."

    col_resolved = resolve_col(df, column_name) or resolve_col(df, column_name.title())
    amount_col = resolve_col(df, "Amount")
    if not col_resolved or not amount_col:
        return f"Column '{column_name}' not available in the dataset for this analysis."

    if use_filters:
        filters = st.session_state.get("filters") or st.session_state.get("active_filters") or {}
        try:
            df = apply_dynamic_filters(df, filters or {})
        except Exception:
            pass
        if df is None or df.empty:
            return f"No {column_name.lower()} available after filtering."

    df[amount_col] = pd.to_numeric(df[amount_col], errors="coerce").fillna(0)

    grouped = (
        df.groupby(col_resolved, dropna=False)[amount_col]
          .sum()
          .reset_index()
          .rename(columns={col_resolved: column_name, amount_col: "Total Spend"})
    )

    if grouped.empty:
        return f"No {column_name.lower()} found to rank."

    nz = grouped[grouped["Total Spend"] > 0]
    data = nz if not nz.empty else grouped
    top_n_data = (data.sort_values("Total Spend", ascending=ascending)
                       .head(int(n)))

    if top_n_data.empty:
        return f"No {column_name.lower()} found after aggregation."

    # ---- chart (Altair) ----
    if show_chart:
        order = alt.SortField(field="Total Spend", order="ascending" if ascending else "descending")
        chart = (
            alt.Chart(top_n_data)
            .mark_bar()
            .encode(
                x=alt.X("Total Spend:Q", title="Total Spend"),
                y=alt.Y(f"{column_name}:N", sort=order, title=column_name),
                tooltip=[f"{column_name}:N", alt.Tooltip("Total Spend:Q", format=",.2f")]
            )
            .properties(height=max(180, 30 * len(top_n_data)), width="container")
        )
        st.altair_chart(chart, use_container_width=True)

    direction = "lowest" if ascending else "top"
    lines = [f"Here are the {direction} {n} {column_name.lower()} by total spend:"]
    for _, row in top_n_data.iterrows():
        label = str(row[column_name])
        lines.append(f"- **{label}**: ${row['Total Spend']:,.2f}")
    return "\n".join(lines)



    # Create a bar chart
    fig = px.bar(top_n_data, x=column_name, y='Total Spend',
                 title=f'{"Lowest" if ascending else "Top"} {n} {column_name.title()} by Total Spend (Excluding Zero Spend)',
                 labels={'Total Spend': 'Total Spend ($)'},
                 color=column_name,
                 height=400)
    st.plotly_chart(fig)

    response_text += "\nWould you like to:"
    response_text += f"\n- See daily trend for a specific {column_name.lower()}?"
    response_text += "\n- Compare total spend by brand?"
    return response_text


def compare_online_offline_func(year=None):
    """Compares online and offline spend, optionally filtered by year, and generates a plot within the current filtered data."""
    # Use the currently filtered DataFrame if available, otherwise use the full processed_df
    data_to_analyze = st.session_state.current_filtered_df if st.session_state.current_filtered_df is not None and not st.session_state.current_filtered_df.empty else st.session_state.processed_df

    if data_to_analyze is None or data_to_analyze.empty:
        return "No data available to perform this analysis. Please upload your data first."
    if "Source" not in data_to_analyze.columns:
        return "Source column not available in the dataset for online/offline comparison."
    temp_df = data_to_analyze.copy()
    if year:
        temp_df = temp_df[temp_df["Year"] == year]

    if temp_df.empty:
        return f"No data available for Online vs Offline comparison for the year {year if year else 'all years'}. Please choose a year with data."
    else:
        result = temp_df.groupby("Source")["Amount"].sum().reset_index()
        result.columns = ["Source", "Total Spend"]

        fig = px.bar(result, x='Source', y='Total Spend',
                    title=f'Online vs Offline Spend in {year if year else "All Years"}',
                    labels={'Total Spend': 'Spend ($)'},
                    height=400,
                    color='Source',
                    color_discrete_map={'Online': 'skyblue', 'Offline': 'lightcoral'})
        st.plotly_chart(fig)

        online_spend = result[result['Source'].str.lower() == 'online']['Total Spend'].iloc[0] if 'online' in result['Source'].str.lower().values else 0
        offline_spend = result[result['Source'].str.lower() == 'offline']['Total Spend'].iloc[0] if 'offline' in result['Source'].str.lower().values else 0
        
        # Corrected string formatting here:
        response_text = (f"**Online vs Offline Spend ({year if year else 'All Years'}):**\n\n"
                    f"Online spend was: **${online_spend:,.2f}**\n\n"
                    f"Offline spend was: **${offline_spend:,.2f}**")
        
        response_text += "\n\nWould you like to:"
        response_text += "\n- See the total spend by year?"
        response_text += "\n- Compare total spend by brand?"
        response_text += "\n- See top sectors?"
        return response_text

# ✅ 4. Fix: Daily Trend for Brand
def plot_daily_trend(df, brand_name):
    if "Brand" not in df.columns or "Month" not in df.columns or "Amount" not in df.columns:
        return "Required columns are missing."

    filtered = df[df["Brand"].str.contains(brand_name, case=False, na=False)]
    if filtered.empty:
        return f"No data found for brand: {brand_name}"

    # Ensure Month column is datetime
    filtered["Month"] = pd.to_datetime(filtered["Month"])
    trend = filtered.groupby("Month")["Amount"].sum().sort_index()

    fig, ax = plt.subplots()
    trend.plot(ax=ax)
    ax.set_title(f"Daily Spend Trend for {brand_name}")
    ax.set_ylabel("Spend")
    ax.set_xlabel("Date")
    st.pyplot(fig)
    return ""

def handle_trend(query):
    """Handles queries for daily/monthly trend, usually for a brand, within the current filtered data."""
    # Use the currently filtered DataFrame if available, otherwise use the full processed_df
    data_to_analyze = st.session_state.current_filtered_df if st.session_state.current_filtered_df is not None and not st.session_state.current_filtered_df.empty else st.session_state.processed_df

    if data_to_analyze is None or data_to_analyze.empty:
        return "No data available to perform this analysis. Please upload your data first."

    brand = None
    brands_in_data = data_to_analyze["Brand"].dropna().unique() if "Brand" in data_to_analyze.columns else []

    found_brand_in_query, score = safe_fuzzy_extract_one(query, brands_in_data)
    if found_brand_in_query:
        brand = found_brand_in_query
    
    if not brand and st.session_state.last_brand:
        brand = st.session_state.last_brand
    
    if not brand:
        logger.warning(f"Brand not specified for trend analysis in query: {query}")
        available_brands_sample = ", ".join([str(b).title() for b in brands_in_data[:3]]) if len(brands_in_data) > 0 else "No brands found in data."
        return f"Brand not specified for trend analysis. Please tell me which brand, e.g., 'Show me the daily trend for {available_brands_sample}'."
    
    logger.info(f"Getting daily trend for brand: {brand}")
    # Call the new matplotlib plotting function
    response_text = plot_daily_trend(data_to_analyze, brand)

    if not response_text: # Empty string means success
        response_text = f"Here's the daily spend trend for **{brand.title()}**."
        if score is not None and score < 100 and score >= config.FUZZY_MATCH_THRESHOLD:
            response_text += f"\n\n*(I found trend data for **{brand.title()}** based on a close match to your query.)*"

        response_text += "\n\nWould you like to:"
        response_text += "\n- See total spend by this brand?"
        response_text += "\n- Compare this brand with another?"
        response_text += "\n- See top sectors?"
    
    return response_text


def get_daily_trend_data(filter_by=None, value=None):
    """
    Generates daily spend trend data, optionally filtered by a column and value, within the current filtered data.
    Returns a Pandas Series with daily sums.
    """
    # Use the currently filtered DataFrame if available, otherwise use the full processed_df
    data_to_analyze = st.session_state.current_filtered_df if st.session_state.current_filtered_df is not None and not st.session_state.current_filtered_df.empty else st.session_state.processed_df

    if data_to_analyze is None or data_to_analyze.empty:
        return "No data available to perform this analysis. Please upload your data first."
    if filter_by not in data_to_analyze.columns:
        return f"The column '{filter_by}' is not available in the dataset for trend analysis."

    temp_df = data_to_analyze.copy()
    if filter_by and value:
        if filter_by in temp_df.columns:
            temp_df = temp_df[temp_df[filter_by].astype(str).str.lower() == str(value).lower()]
        else:
            logger.warning(f"Filter column '{filter_by}' not found in data for trend analysis.")
            return pd.Series(dtype='float64')
    
    if temp_df.empty:
        return pd.Series(dtype='float64')

    trend = temp_df.groupby(temp_df["Month"].dt.date)["Amount"].sum().sort_index()
    return trend

def get_highest_spend_day():
    """Identifies and reports the highest spend day, weekday, and month within the current filtered data."""
    # Use the currently filtered DataFrame if available, otherwise use the full processed_df
    data_to_analyze = st.session_state.current_filtered_df if st.session_state.current_filtered_df is not None and not st.session_state.current_filtered_df.empty else st.session_state.processed_df

    if data_to_analyze is None or data_to_analyze.empty:
        return "No data available to determine highest spend day. Please upload your data first."
    
    if not pd.api.types.is_numeric_dtype(data_to_analyze['Amount']):
        data_to_analyze['Amount'] = pd.to_numeric(data_to_analyze['Amount'], errors='coerce').fillna(0)

    max_row = data_to_analyze.loc[data_to_analyze["Amount"].idxmax()]
    
    highest_total_spend_month_name = "N/A"
    valid_months_df = data_to_analyze.dropna(subset=['Month'])
    if not valid_months_df.empty:
        highest_total_spend_month_name = valid_months_df.groupby(valid_months_df['Month'].dt.month_name())['Amount'].sum().idxmax()

    response_text = f"\U0001F4C5 **Highest spend day ever**: {max_row['Day'].strftime('%Y-%m-%d')} with Amount **${max_row['Amount']:,.2f}**" + \
                f"\n\U0001F4C8 **Highest spend weekday**: {data_to_analyze.groupby('Weekday')['Amount'].sum().idxmax()}" + \
                f"\n\U0001F4A1 **Highest total spend month**: {highest_total_spend_month_name}"
    
    response_text += "\n\nWould you like to:"
    response_text += "\n- See daily trends for a specific brand?"
    response_text += "\n- Forecast spend for next week?"
    return response_text

def handle_month_comparison(query):
    """Handles queries to compare spend between two specific months/years within the current filtered data."""
    # Use the currently filtered DataFrame if available, otherwise use the full processed_df
    data_to_analyze = st.session_state.current_filtered_df if st.session_state.current_filtered_df is not None and not st.session_state.current_filtered_df.empty else st.session_state.processed_df

    if data_to_analyze is None or data_to_analyze.empty:
        return "No data available to perform this analysis. Please upload your data first."
    
    date_candidates_raw = re.findall(r"(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s*\d{2,4}|\b\d{4}\b", query.lower())
    
    dates = []
    for date_str in date_candidates_raw:
        parsed_date = dateparser.parse(date_str, settings={'PREFER_DAY_OF_MONTH': 'first', 'RELATIVE_BASE': datetime.now()})
        if parsed_date:
            if not any(d.month == parsed_date.month and d.year == parsed_date.year for d in dates):
                dates.append(parsed_date)
        if len(dates) >= 2:
            break
    
    if len(dates) >= 2:
        dates.sort()
        
        month1 = dates[0].month
        year1 = dates[0].year
        month2 = dates[1].month
        year2 = dates[1].year
        
        logger.info(f"Comparing months: {datetime(year1, month1, 1).strftime('%B %Y')} and {datetime(year2, month2, 1).strftime('%B %Y')}")
        response_text = compare_months(month1, year1, month2, year2) # compare_months will use data_to_analyze
    else:
        logger.warning(f"Not enough distinct month-year pairs found in query for comparison: {query}. Found dates: {[d.strftime('%Y-%m') for d in dates]}")
        response_text = "Please provide two distinct months and years to compare, like 'Compare March 2023 and March 2024'."
    
    if len(dates) >= 2:
        response_text += "\n\nWould you like to:"
        response_text += "\n- See overall spend trend?"
        response_text += "\n- Compare online vs offline spend?"
    return response_text

def compare_months(month1, year1, month2, year2):
    """Compares total spend between two specified months and years within the current filtered data."""
    # Use the currently filtered DataFrame if available, otherwise use the full processed_df
    data_to_analyze = st.session_state.current_filtered_df if st.session_state.current_filtered_df is not None and not st.session_state.current_filtered_df.empty else st.session_state.processed_df

    if data_to_analyze is None or data_to_analyze.empty:
        return "No data available to perform this analysis. Please upload your data first."

    amt1_df = data_to_analyze[(data_to_analyze['Month'].dt.month == month1) & (data_to_analyze['Month'].dt.year == year1)]
    amt1 = amt1_df["Amount"].sum()
    
    amt2_df = data_to_analyze[(data_to_analyze['Month'].dt.month == month2) & (data_to_analyze['Month'].dt.year == year2)]
    amt2 = amt2_df["Amount"].sum()

    month1_name = datetime(year1, month1, 1).strftime('%B %Y')
    month2_name = datetime(year2, month2, 1).strftime('%B %Y')

    response_text = (f"Spend in **{month1_name}**: **${amt1:,.0f}**\n"
                f"Spend in **{month2_name}**: **${amt2:,.0f}**")

    if amt1 > amt2:
        response_text += f"\n\n**{month1_name}** had higher spend."
    elif amt2 > amt1:
        response_text += f"\n\n**{month2_name}** had higher spend."
    else:
        response_text += f"\n\nSpend was equal for both months."

    response_text += "\n\nWould you like to:"
    response_text += "\n- See overall spend trend?"
    response_text += "\n- Compare online vs offline spend?"
    return response_text

def forecast_next_7_days_func():
    """Forecasts spend for the next 7 days using linear regression and generates a plot within the current filtered data."""
    # Use the currently filtered DataFrame if available, otherwise use the full processed_df
    data_to_analyze = st.session_state.current_filtered_df if st.session_state.current_filtered_df is not None and not st.session_state.current_filtered_df.empty else st.session_state.processed_df

    if data_to_analyze is None or data_to_analyze.empty:
        return "No data available to perform this analysis. Please upload your data first."

    df_daily = data_to_analyze.groupby("Day")["Amount"].sum().reset_index()
    df_daily["Day"] = pd.to_datetime(df_daily["Day"])
    df_daily = df_daily.sort_values("Day").reset_index(drop=True)
    df_daily["Ordinal"] = df_daily["Day"].map(datetime.toordinal)
    
    if len(df_daily) < 2:
        return "Not enough historical data (at least 2 data points needed) to forecast for the next 7 days."

    X = df_daily["Ordinal"].values.reshape(-1, 1)
    y = df_daily["Amount"].values
    
    try:
        model_lr = LinearRegression().fit(X, y)
    except ValueError as e:
        logger.error(f"Error fitting linear regression model: {e}", exc_info=True)
        return "Could not generate a forecast due to an issue with the historical data. Please ensure there's sufficient variation in spend over time."

    last_known_date = df_daily["Day"].max()
    future_dates = [last_known_date + pd.Timedelta(days=i) for i in range(1, 8)]
    future_ordinals = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
    preds = model_lr.predict(future_ordinals)
    
    forecast_results = []
    for d, p in zip(future_dates, preds):
        forecast_results.append(f"{d.strftime('%Y-%m-%d')}: ${max(0, p):,.0f}")
    
    response_text = "Here's the predicted spend for the next 7 days:\n"
    response_text += "```\n" + "\n".join(forecast_results) + "\n```"
    
    disclaimer = "\n\n*Disclaimer: This is a linear projection based on historical data and may not account for seasonality, market shifts, or other external factors.*"
    response_text += disclaimer
    
    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Amount': [max(0, p) for p in preds]
    })
    
    plot_df = pd.concat([df_daily.rename(columns={'Day': 'Date'}), forecast_df])
    
    fig = px.line(plot_df, x='Date', y='Amount',
                title='Daily Spend Forecast (Next 7 Days)',
                labels={'Date': 'Date', 'Amount': 'Amount ($)'},
                markers=True,
                height=500)
    
    fig.add_vline(x=last_known_date, line_dash="dash", line_color="red", annotation_text="Last Known Data")
    
    st.plotly_chart(fig)

    response_text += "\n\nWould you like to:"
    response_text += "\n- See the highest spend day?"
    response_text += "\n- Compare total spend by brand?"
    return response_text

def compare_two_brands(query):
    """Compares total spend between two brands extracted from the query and generates a plot within the current filtered data."""
    # Use the currently filtered DataFrame if available, otherwise use the full processed_df
    data_to_analyze = st.session_state.current_filtered_df if st.session_state.current_filtered_df is not None and not st.session_state.current_filtered_df.empty else st.session_state.processed_df

    if data_to_analyze is None or data_to_analyze.empty:
        return "No data available to perform this analysis. Please upload your data first."
    if "Brand" not in data_to_analyze.columns:
        return "Brand column not available in the dataset for brand comparison."

    brands_in_data = data_to_analyze["Brand"].dropna().unique()
    
    potential_matches = process.extract(query, brands_in_data, scorer=fuzz.token_set_ratio)
    
    found_brands_in_query = []
    for brand_name, score in potential_matches:
        if score >= config.FUZZY_MATCH_THRESHOLD:
            if not any(brand_name.lower() == existing_brand_in_list.lower() for existing_brand_in_list in found_brands_in_query): # Corrected variable name
                    found_brands_in_query.append(brand_name)
            if len(found_brands_in_query) >= 2:
                break

    if len(found_brands_in_query) >= 2:
        brand1 = found_brands_in_query[0]
        brand2 = found_brands_in_query[1]
        logger.info(f"Final brands selected for comparison: {brand1}, {brand2}")

        # Ensure these calls use the data_to_analyze, not the global df
        spend1_df = data_to_analyze[data_to_analyze["Brand"].astype(str).str.lower() == brand1.lower()]
        spend1 = spend1_df["Amount"].sum()

        spend2_df = data_to_analyze[data_to_analyze["Brand"].astype(str).str.lower() == brand2.lower()]
        spend2 = spend2_df["Amount"].sum()


        data = {'Brand': [brand1, brand2], 'Total Spend': [spend1, spend2]}
        compare_df = pd.DataFrame(data)

        fig = px.bar(compare_df, x='Brand', y='Total Spend',
                    title=f'Total Spend Comparison Between {brand1.title()} and {brand2.title()}',
                    labels={'Total Spend': 'Spend ($)'},
                    color='Brand',
                    height=400)
        st.plotly_chart(fig)

        # Corrected string formatting here to ensure proper line breaks and bolding
        response_text = (
            f"Spend of {brand1.title()}: **${spend1:,.0f}**\n\n"
            f"Spend of {brand2.title()}: **${spend2:,.0f}**"
        )
        
        if spend1 > spend2:
            response_text += f"\n\n**{brand1.title()}** had higher spend."
        elif spend2 > spend1:
            response_text += f"\n\n**{brand2.title()}** had higher spend."
        else:
            response_text += f"\n\nSpend was equal for both brands."

        response_text += "\n\nWould you like to:"
        response_text += f"\n- See the daily trend for {brand1.title()}?"
        response_text += "\n- See top sectors?"
        return response_text
    
    logger.warning(f"Not enough valid brands found for comparison in query: {query}. Found: {found_brands_in_query}")
    available_brands_sample = ", ".join([str(b).title() for b in data_to_analyze["Brand"].dropna().unique()[:3]]) if not data_to_analyze["Brand"].dropna().empty else "No brands found in data."
    return f"Please specify two valid brand names to compare. Try: 'Compare {available_brands_sample}' or 'Compare X vs Y'."

# ✅ 2. Fix: List All Brands
def list_all_brands():
    """Lists all unique brands available in the dataset within the current filtered data."""
    df = st.session_state.current_filtered_df if st.session_state.current_filtered_df is not None and not st.session_state.current_filtered_df.empty else st.session_state.processed_df
    if df is None:
        return "No data available."
    if "Brand" not in df.columns:
        return "Missing 'Brand' column in data."

    brands = df["Brand"].dropna().unique()
    brands_sorted = sorted(brands)
    return "Here are all brands in the dataset:\n" + ", ".join(brands_sorted)

def list_all_sectors():
    """Lists all unique sectors available in the dataset within the current filtered data."""
    # Use the currently filtered DataFrame if available, otherwise use the full processed_df
    data_to_analyze = st.session_state.current_filtered_df if st.session_state.current_filtered_df is not None and not st.session_state.current_filtered_df.empty else st.session_state.processed_df

    if data_to_analyze is None or data_to_analyze.empty:
        return "No data available to perform this analysis. Please upload your data first."
    if "Sector" not in data_to_analyze.columns:
        return "Sector column not available in the dataset."
    sectors = data_to_analyze["Sector"].dropna().unique()
    if len(sectors) == 0:
        return "No sectors found in the dataset."
    string_sectors = [str(s) for s in sectors]
    return "Available Sectors:\n" + "\n".join([f"- {s.title()}" for s in sorted(string_sectors)])

# ✅ 3. Fix: Highest Offline Spend by a Brand
def top_brand_by_channel(channel="OFFLINE"):
    df = st.session_state.processed_df
    if df is None:
        return "No data available."
    if "Channel" not in df.columns or "Brand" not in df.columns or "Amount" not in df.columns:
        return "Missing required columns."

    filtered = df[df["Channel"].str.upper() == channel.upper()]
    if filtered.empty:
        return f"No data found for channel: {channel}"

    brand_spend = filtered.groupby("Brand")["Amount"].sum().sort_values(ascending=False).head(1)
    if brand_spend.empty:
        return f"No brand spend data found for channel: {channel}"
    brand, amount = brand_spend.index[0], brand_spend.iloc[0]
    return f"The brand with the highest spend in {channel.title()} is {brand} with ${amount:,.2f}"

# --- Generic Data Analysis Functions for LLM Tool Use ---
def calculate_column_statistic(column_name, statistic_type):
    """
    Calculates a specified statistic (sum, mean, min, max, count) for a given column within the current filtered data.
    Args:
        column_name (str): The name of the column to analyze.
        statistic_type (str): The type of statistic to calculate ('sum', 'mean', 'min', 'max', 'count').
    """
    # Use the currently filtered DataFrame if available, otherwise use the full processed_df
    data_to_analyze = st.session_state.current_filtered_df if st.session_state.current_filtered_df is not None and not st.session_state.current_filtered_df.empty else st.session_state.processed_df

    if data_to_analyze is None or data_to_analyze.empty:
        return "No data available to perform this analysis. Please upload your data first."
    if column_name not in data_to_analyze.columns:
        return f"Column '{column_name}' not found in your data. Please check the column name."
    
    col_data = data_to_analyze[column_name]
    
    if statistic_type == 'sum':
        if pd.api.types.is_numeric_dtype(col_data):
            result = col_data.sum()
            return f"The sum of '{column_name}' is: **{result:,.2f}**"
        else:
            return f"Cannot calculate sum for non-numeric column '{column_name}'."
    elif statistic_type == 'mean':
        if pd.api.types.is_numeric_dtype(col_data):
            result = col_data.mean()
            return f"The average of '{column_name}' is: **{result:,.2f}**"
        else:
            return f"Cannot calculate mean for non-numeric column '{column_name}'."
    elif statistic_type == 'min':
        if pd.api.types.is_numeric_dtype(col_data):
            result = col_data.min()
            return f"The minimum value in '{column_name}' is: **{result:,.2f}**"
        else:
            return f"Cannot find minimum for non-numeric column '{column_name}'."
    elif statistic_type == 'max':
        if pd.api.types.is_numeric_dtype(col_data):
            result = col_data.max()
            return f"The maximum value in '{column_name}' is: **{result:,.2f}**"
        else:
            return f"Cannot find maximum for non-numeric column '{column_name}'."
    elif statistic_type == 'count':
        result = col_data.count()
        return f"The count of non-empty values in '{column_name}' is: **{result:,}**"
    else:
        return f"Unsupported statistic type: '{statistic_type}'. Supported types are: 'sum', 'mean', 'min', 'max', 'count'."

def plot_column_distribution(column_name, plot_type='bar'):
    """
    Generates a plot for the distribution of values in a given column within the current filtered data.
    Args:
        column_name (str): The name of the column to plot.
        plot_type (str): The type of plot ('bar' for categorical/counts, 'histogram' for numeric distribution).
    """
    # Use the currently filtered DataFrame if available, otherwise use the full processed_df
    data_to_analyze = st.session_state.current_filtered_df if st.session_state.current_filtered_df is not None and not st.session_state.current_filtered_df.empty else st.session_state.processed_df

    if data_to_analyze is None or data_to_analyze.empty:
        return "No data available to perform this analysis. Please upload your data first."
    if column_name not in data_to_analyze.columns:
        return f"Column '{column_name}' not found in your data. Please check the column name."
    
    col_data = data_to_analyze[column_name]

    if plot_type == 'bar':
        counts = col_data.value_counts().reset_index()
        counts.columns = [column_name, 'Count']
        fig = px.bar(counts, x=column_name, y='Count',
                     title=f'Distribution of {column_name}',
                     labels={'Count': 'Number of Occurrences'},
                     color=column_name)
        st.plotly_chart(fig)
        return f"Here is the bar chart showing the distribution of '{column_name}'."
    elif plot_type == 'histogram':
        if pd.api.types.is_numeric_dtype(col_data):
            fig = px.histogram(data_to_analyze, x=column_name,
                               title=f'Histogram of {column_name}',
                               labels={column_name: column_name, 'count': 'Frequency'})
            st.plotly_chart(fig)
            return f"Here is the histogram showing the distribution of '{column_name}'."
        else:
            return f"Cannot create a histogram for non-numeric column '{column_name}'. Try 'bar' chart instead."
    else:
        return f"Unsupported plot type: '{plot_type}'. Supported types are: 'bar', 'histogram'."

def get_top_sectors_by_spend(n=5):
    """Returns the top N sectors by total spend, in thousands."""
    global df
    if "Sector" not in df.columns:
        return "Sector column not available in the dataset for this analysis."
    
    # Ensure 'Amount' column is numeric before summing
    if not pd.api.types.is_numeric_dtype(df['Amount']):
        df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce').fillna(0)

    top_sectors_data = df.groupby("Sector")["Amount"].sum().sort_values(ascending=False).head(n).reset_index()
    top_sectors_data.columns = ['Sector', 'Total Spend']  # Rename columns for clarity

    if top_sectors_data.empty:
        return "No sectors found in the dataset to analyze."

    response_text = f"Here are the top {n} sectors by total spend:\n"
    for index, row in top_sectors_data.iterrows():
        response_text += f"- **{row['Sector'].title()}**: ${row['Total Spend']:,.2f}\n"

    # Create a bar chart
    fig = px.bar(top_sectors_data, x='Sector', y='Total Spend',
                 title=f'Top {n} Sectors by Total Spend',
                 labels={'Total Spend': 'Total Spend ($)'},
                 color='Sector',
                 height=400)
    st.plotly_chart(fig)

    response_text += "\nWould you like to:"
    response_text += "\n- See top brands?"
    response_text += "\n- Compare online vs offline spend?"
    return response_text

def list_unique_values(column_name):
    """
    Lists all unique values in a given column within the current filtered data.
    Args:
        column_name (str): The name of the column.
    """
    # Use the currently filtered DataFrame if available, otherwise use the full processed_df
    data_to_analyze = st.session_state.current_filtered_df if st.session_state.current_filtered_df is not None and not st.session_state.current_filtered_df.empty else st.session_state.processed_df

    if data_to_analyze is None or data_to_analyze.empty:
        return "No data available to perform this analysis. Please upload your data first."
    if column_name not in data_to_analyze.columns:
        return f"Column '{column_name}' not found in your data. Please check the column name."
    
    unique_vals = data_to_analyze[column_name].dropna().unique()
    if len(unique_vals) == 0:
        return f"No unique values found in column '{column_name}'."
    
    display_limit = 20
    if len(unique_vals) > display_limit:
        display_vals = sorted([str(v) for v in unique_vals[:display_limit]])
        return f"Unique values in '{column_name}' (showing first {display_limit}):\n" + "\n".join([f"- {v}" for v in display_vals]) + "\n..."
    else:
        display_vals = sorted([str(v) for v in unique_vals])
        return f"Unique values in '{column_name}':\n" + "\n".join([f"- {v}" for v in display_vals])

# Helper to extract column name from query for generic functions
def handle_top_sectors(query):
    n_match = re.search(r'top\s*(\d+)\s*sectors', query, re.IGNORECASE)
    n = int(n_match.group(1)) if n_match else 5
    return get_top_sectors_by_spend(n)

def extract_column_name(query):
    # Use the currently filtered DataFrame's columns for context
    data_columns = st.session_state.current_filtered_df.columns.tolist() if st.session_state.current_filtered_df is not None else (st.session_state.processed_df.columns.tolist() if st.session_state.processed_df is not None else [])
    
    if not data_columns:
        return "Amount" # Default if no data loaded

    # Look for exact column names in the query first (case-insensitive)
    for col in ALL_INTERNAL_COLUMNS: # Check against all possible internal names
        if col.lower() in query.lower() and col in data_columns: # Ensure it's in the actual data
            return col
    
    # If not found directly, try to identify words that might be column names
    words_in_query = re.findall(r'\b(?:[a-zA-Z0-9_]+)\b', query, re.IGNORECASE)
    
    best_match_col = None
    highest_score = -1

    for col in data_columns: # Iterate through actual columns in the current data
        for word in words_in_query:
            score = fuzz.ratio(word.lower(), col.lower())
            if score > highest_score and score >= 70: # Use a reasonable threshold for fuzzy matching
                highest_score = score
                best_match_col = col
    
    if best_match_col:
        return best_match_col

    return "Amount" # Default to 'Amount' if no specific column found

# --- Semantic Mappings ---
# Define common synonyms or related terms for data values that might appear in queries
SEMANTIC_MAPPINGS = {
    "source": {
        "online": ["online", "digital", "web", "internet"],
        "offline": ["offline", "traditional", "print", "tv", "radio", "ooh"]
    },
    # Add other semantic mappings as needed, e.g., for channels or categories
    # "channel": {
    #     "social media": ["facebook", "instagram", "twitter", "linkedin"],
    #     "search": ["google ads", "bing ads"]
    # }
}

def get_total_spend_by_brand_and_channel(df, brand, channel_or_source):
    """
    Supports queries like 'total online spend by BMW'.
    Uses 'Source' (ONLINE/OFFLINE) when appropriate; otherwise falls back to 'Channel'.
    """
    required = {"Brand", "Amount"}
    if not required.issubset(df.columns):
        return "Missing required columns: Brand/Amount."

    brand_q = str(brand).strip().lower()
    cos_q   = str(channel_or_source).strip().lower()

    use_source = ("Source" in df.columns) and (cos_q in {"online", "offline"})
    col = "Source" if use_source else ("Channel" if "Channel" in df.columns else None)
    if col is None:
        return "Neither Source nor Channel available in the dataset."

    brand_mask = df["Brand"].astype(str).str.lower().str.contains(brand_q, na=False)
    col_mask   = df[col].astype(str).str.lower().str.contains(cos_q,   na=False)

    filtered = df[brand_mask & col_mask]
    if filtered.empty:
        nice_col = "Source" if use_source else "Channel"
        return f"No data found for brand '{brand}' with {nice_col}='{channel_or_source}'."

    if not pd.api.types.is_numeric_dtype(filtered["Amount"]):
        filtered["Amount"] = pd.to_numeric(filtered["Amount"], errors="coerce").fillna(0)

    total = filtered["Amount"].sum()
    label = "Online/Offline (Source)" if use_source else "Channel"
    return f"Total spend by **{brand}** on **{channel_or_source}** ({label}) is **${total:,.2f}**."



# Updated function to extract filters from query
def extract_filters_from_query(query, df_columns, brands_in_data, sectors_in_data, sources_in_data, categories_in_data, products_in_data, media_in_data, agencies_in_data, producers_in_data, countries_in_data, channels_in_data):
    logger.info(f"DEBUG: Entering extract_filters_from_query for query: '{query}')")
    import calendar

    # Month-based filtering
    for month_name in calendar.month_name[1:]:  # Skips index 0 (empty string)
        if month_name.lower() in query.lower():
            st.sidebar.write(f"🔎 Detected month filter: {month_name}")
            return {"Month_str": month_name}  # You can also use filters['Month_str'] = month_name if building multiple filters

    st.sidebar.write(f"DEBUG - extract_filters_from_query: Query: '{query}'")
    st.sidebar.write(f"DEBUG - df_columns: {df_columns}")
    st.sidebar.write(f"DEBUG - Brands in data (sample): {brands_in_data[:5]}...")
    st.sidebar.write(f"DEBUG - Sources in data (sample): {sources_in_data[:5]}...")
    st.sidebar.write(f"DEBUG - Channels in data (sample): {channels_in_data[:5]}...")
    st.sidebar.write(f"DEBUG - Categories in data (sample): {categories_in_data[:5]}...")


    filters = {}
    query_lower = query.lower()
    
    potential_matches = [] # List to store (column_name, matched_value, score, query_phrase_matched)
    used_query_words = set() # To track words that have led to a selected filter

    entity_column_data = {
        "Brand": brands_in_data,
        "Source": sources_in_data,
        "Channel": channels_in_data,
        "Product": products_in_data,
        "Sector": sectors_in_data,
        "Category": categories_in_data,
        "Media": media_in_data,
        "Media Agency": agencies_in_data, # Corrected from "Agency"
        "Producer": producers_in_data,
        "Country": countries_in_data,
    }

    # Define a priority for columns when resolving conflicts (lower index = higher priority)
    column_priority = {col: i for i, col in enumerate([
        "Brand", "Source", "Sector", "Channel", "Product", "Category", "Media", "Media Agency", "Producer", "Country" # Corrected "Agency" to "Media Agency"
    ])}

    # Generate all possible n-grams (up to 3 words) from the query
    query_words_list = re.findall(r'\b\w+\b', query_lower)
    query_ngrams = set()
    for i in range(len(query_words_list)):
        query_ngrams.add(query_words_list[i]) # 1-gram
        if i + 1 < len(query_words_list):
            query_ngrams.add(" ".join(query_words_list[i:i+2])) # 2-gram
        if i + 2 < len(query_words_list):
            query_ngrams.add(" ".join(query_words_list[i:i+3])) # 3-gram
    
    # Sort n-grams by length descending to prefer longer, more specific matches
    sorted_ngrams = sorted(list(query_ngrams), key=len, reverse=True)
    st.sidebar.write(f"DEBUG - Sorted N-grams from query: {sorted_ngrams}")

    # Step 1: Collect all potential matches with their scores and original query phrases
    for col_name in column_priority.keys(): # Iterate through prioritized columns
        if col_name not in df_columns:
            continue # Skip if the column doesn't exist in the actual DataFrame
        
        entities_in_col = entity_column_data.get(col_name, [])
        
        if not entities_in_col: # Check if the list of entities is empty
            continue

        for phrase in sorted_ngrams:
            # NEW: Check if the phrase is a common query word and should be ignored for this column
            is_common_word = phrase in COMMON_QUERY_WORDS
            is_source_col_and_online_offline = col_name == "Source" and (phrase == "online" or phrase == "offline")

            if is_common_word and not is_source_col_and_online_offline:
                # If it's a common query word and not 'online'/'offline' for 'Source' column, skip
                logger.debug(f"Skipping common query word '{phrase}' for column '{col_name}'.")
                st.sidebar.write(f"DEBUG - Skipping common query word '{phrase}' for column '{col_name}'.")
                continue

            matched_entity_val = None
            score = -1

            # --- 1. Exact Match Check (Highest Priority) ---
            # Check if the phrase is an exact match (case-insensitive) to any entity in the column
            for entity in entities_in_col:
                if str(entity).strip().lower() == phrase:
                    matched_entity_val = str(entity).strip()
                    score = 101 # Give it a score higher than any fuzzy match (100 max)
                    logger.debug(f"Exact Match: '{phrase}' -> '{matched_entity_val}' for {col_name} (Score: {score})")
                    st.sidebar.write(f"DEBUG - Exact Match: '{phrase}' -> '{matched_entity_val}' for {col_name} (Score: {score})")
                    break # Found exact match, no need to check other entities in this column for this phrase
            
            # --- 2. Semantic Mapping Check (High Priority, if no exact match) ---
            if not matched_entity_val:
                if col_name.lower() in SEMANTIC_MAPPINGS:
                    for canonical_term, synonyms in SEMANTIC_MAPPINGS[col_name.lower()].items():
                        if phrase in synonyms:
                            # Check if the canonical term or any of its synonyms exist in the actual data
                            for data_entity in entities_in_col: # Use original-cased list here
                                if str(data_entity).strip().lower() == canonical_term or str(data_entity).strip().lower() in synonyms:
                                    matched_entity_val = str(data_entity).strip() # Use the actual data entity, not the canonical
                                    score = 100 # High score for semantic matches
                                    logger.debug(f"Semantic Match: '{phrase}' -> '{matched_entity_val}' for {col_name} (Score: {score})")
                                    st.sidebar.write(f"DEBUG - Semantic Match: '{phrase}' -> '{matched_entity_val}' for {col_name} (Score: {score})")
                                    break 
                        if matched_entity_val:
                            break 
            
            # --- 3. Fuzzy Matching (if no exact or semantic match) ---
            if not matched_entity_val:
                # Pass the original-cased entities_in_col to safe_fuzzy_extract_one
                matched_entity_val, score = safe_fuzzy_extract_one(phrase, entities_in_col)
            
            if matched_entity_val:
                # Exclude "marketing" as a Category filter
                if col_name == "Category" and matched_entity_val.lower() == "marketing":
                    logger.debug(f"Skipping 'marketing' as a Category filter for query '{query}'.")
                    continue 
                potential_matches.append((col_name, matched_entity_val, score, phrase))
                st.sidebar.write(f"DEBUG - Added potential match: ({col_name}, '{matched_entity_val}', {score}, '{phrase}')")

    # Step 2: Sort potential matches for conflict resolution
    # Sort by score (desc), then by phrase length (desc), then by column priority (asc)
    potential_matches.sort(key=lambda x: (x[2], len(x[3]), column_priority.get(x[0], len(column_priority))), reverse=True)
    st.sidebar.write(f"DEBUG - Sorted Potential Matches: {potential_matches}")

    # Step 3: Select the best matches, ensuring only one filter per column type and consuming query words
    for col_name, matched_value, score, query_phrase_matched in potential_matches:
        if col_name in filters: # If this column type has already been filtered by a higher priority match
            continue 
        
        # Check if the words in `query_phrase_matched` have already been "used" by a higher-priority match.
        words_in_current_phrase = set(re.findall(r'\b\w+\b', query_phrase_matched.lower()))
        
        overlap_found = False
        for word in words_in_current_phrase:
            if word in used_query_words:
                overlap_found = True
                break
        
        if overlap_found:
            logger.debug(f"Skipping '{query_phrase_matched}' for {col_name} as its words are already used by a higher priority match.")
            st.sidebar.write(f"DEBUG - Skipping '{query_phrase_matched}' for {col_name} as its words are already used by a higher priority match.")
            continue

        # If we reach here, this is the best match for `col_name` AND its words aren't already consumed.
        filters[col_name] = matched_value.title()
        used_query_words.update(words_in_current_phrase) # Mark words as used
        logger.debug(f"Selected filter: {col_name} = {filters[col_name]} (Phrase: '{query_phrase_matched}'). Used words: {words_in_current_phrase}")
        st.sidebar.write(f"DEBUG - Selected filter: {col_name} = {filters[col_name]} (Phrase: '{query_phrase_matched}'). Used words: {used_query_words}")


    # Year and Month extraction (these are less likely to conflict and can be done after entity extraction)
    year_match = re.search(r'\b(?:in|for|year)\s+(\d{4})\b', query_lower)
    if year_match:
        filters["Year"] = int(year_match.group(1))
        logger.debug(f"Extracted filter: Year = {filters['Year']}")
        st.sidebar.write(f"DEBUG - extract_filters_from_query: Extracted filter: Year = {filters['Year']}")

    month_match = re.search(r'\b(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b', query_lower)
    if month_match:
        month_name = month_match.group(1).lower()
        month_dict = {
            'january': 1, 'jan': 1, 'february': 2, 'feb': 2, 'march': 3, 'mar': 3,
            'april': 4, 'apr': 4, 'may': 5, 'june': 6, 'jun': 6, 'july': 7, 'jul': 7,
            'august': 8, 'aug': 8, 'september': 9, 'sep': 9, 'october': 10, 'oct': 10,
            'november': 11, 'nov': 11, 'december': 12, 'dec': 12
        }
        filters["Month"] = month_dict.get(month_name)
        logger.debug(f"Extracted filter: Month = {filters['Month']}")
        st.sidebar.write(f"DEBUG - extract_filters_from_query: Extracted filter: Month = {filters['Month']}")

    # Amount range (remains the same)
    amount_range_match = re.search(r'between\s+\$*(\d[\d,.]*)\s+and\s+\$*(\d[\d,.]*)', query_lower)
    if amount_range_match: 
        min_amount = float(amount_range_match.group(1).replace(',', ''))
        max_amount = float(amount_range_match.group(2).replace(',', ''))
        filters["Amount_Range"] = (min_amount, max_amount)
        logger.debug(f"Extracted filter: Amount_Range = {filters['Amount_Range']}")
        st.sidebar.write(f"DEBUG - extract_filters_from_query: Extracted filter: Amount_Range = {filters['Amount_Range']}")
    
    greater_than_match = re.search(r'(?:greater than|more than|over)\s+\$*(\d[\d,.]*)', query_lower)
    if greater_than_match:
        min_amount = float(greater_than_match.group(1).replace(',', ''))
        if "Amount_Range" in filters:
            filters["Amount_Range"] = (max(filters["Amount_Range"][0], min_amount), filters["Amount_Range"][1])
        else:
            filters["Amount_Range"] = (min_amount, float('inf'))
        logger.debug(f"Extracted filter: Amount_Range (GT) = {filters['Amount_Range']}")
        st.sidebar.write(f"DEBUG - extract_filters_from_query: Extracted filter: Amount_Range (GT) = {filters['Amount_Range']}")

    less_than_match = re.search(r'(?:less than|under)\s+\$*(\d[\d,.]*)', query_lower)
    if less_than_match:
        max_amount = float(less_than_match.group(1).replace(',', ''))
        if "Amount_Range" in filters:
            filters["Amount_Range"] = (filters["Amount_Range"][0], min(filters["Amount_Range"][1], max_amount))
        else:
            filters["Amount_Range"] = (0, max_amount)
        logger.debug(f"Extracted filter: Amount_Range (LT) = {filters['Amount_Range']}")
        st.sidebar.write(f"DEBUG - extract_filters_from_query: Extracted filter: Amount_Range (LT) = {filters['Amount_Range']}")

    st.sidebar.write(f"DEBUG - extract_filters_from_query: Final filters: {filters}")
    logger.info(f"DEBUG: Final filters extracted by extract_filters_from_query: {filters}")
    return filters

# New function to apply filters dynamically
def apply_dynamic_filters(df_to_filter, filters):
    filtered_df = df_to_filter.copy()
    
    for col, val in filters.items():
        if col == "Year":
            filtered_df = filtered_df[filtered_df["Year"] == val]
        elif col == "Month":
            filtered_df = filtered_df[filtered_df["Month"].dt.month == val]
        elif col == "Amount_Range":
            min_amt, max_amt = val
            filtered_df = filtered_df[(filtered_df["Amount"] >= min_amt) & (filtered_df["Amount"] <= max_amt)]
        elif col in filtered_df.columns: # For all other categorical columns
            filtered_df = filtered_df[filtered_df[col].astype(str).str.lower() == str(val).lower()]
        else:
            logger.warning(f"Attempted to filter by unknown column or unhandled filter type: {col}")
            # Optionally, inform the user that a filter could not be applied
            # return f"Warning: Could not apply filter for '{col}'. Column not found or filter type not supported."
            
    return filtered_df

def format_spend_response(spend, filters):
    if spend == 0:
        return "No data found matching your criteria. Please try different filters."
    
    filters_text = "\n\n🔎 **Applied Filters:**\n" + "\n".join([f"- **{k}**: {v}" for k, v in filters.items()])
    return f"**The total spend for your specified criteria was ${spend:,.2f}.**{filters_text}"

# New generic handler for complex queries
def handle_complex_query(query=None, filters_dict=None):  # Added filters_dict parameter
    logger.info(f"DEBUG: Entering handle_complex_query. Query param: '{query}', filters_dict param: {filters_dict}")
    
    response_text = ""

    # Determine the base DataFrame to apply filters to
    if st.session_state.current_filtered_df is not None and not st.session_state.current_filtered_df.empty:
        base_df = st.session_state.current_filtered_df.copy()
        logger.info("Using current_filtered_df as base for complex query.")
    else:
        base_df = st.session_state.processed_df.copy()  # Use the full processed_df
        logger.info("Using full processed_df as base for complex query.")

    logger.info(f"DEBUG: Initial base_df shape in handle_complex_query: {base_df.shape}")
    st.sidebar.write(f"DEBUG - handle_complex_query: Initial base_df shape: {base_df.shape}")

    import re

    # --- short-circuit: "top/bottom N sectors" ---
    q = query.lower().strip()
    m = re.search(r'\b(top|bottom|lowest)\s+(\d+)\s+sectors?\b', q)
    if m:
        n = int(m.group(2))
        lowest = m.group(1) in ("bottom", "lowest")
        # If you want to respect current filters, set use_filters=True
        resp = get_top_sectors(n=n, ascending=lowest, use_filters=False)
        return resp  # or: response_text = resp; return response_text

    # --- short-circuit: "top/bottom N <column> by spend" ---
    m2 = re.search(r'\b(top|bottom|lowest)\s+(\d+)\s+([a-zA-Z &/]+?)\s+by\s+spend\b', q)
    if m2:
        n = int(m2.group(2))
        lowest = m2.group(1) in ("bottom", "lowest")
        col = m2.group(3).strip().title()  # e.g., "Sector", "Category", "Brand"
        resp = get_top_n_by_column(column_name=col, n=n, ascending=lowest, use_filters=False)
        return resp  # or: response_text = resp; return response_text


    # ==== ONLINE/OFFLINE spend by BRAND fast-path ====
    ql = (query or "").lower()
    if "spend" in ql and ("online" in ql or "offline" in ql):
        if "Brand" in base_df.columns:
            # quick brand guess via substring; replace with your stronger extractor if you have one
            brand_guess = next(
                (b for b in base_df["Brand"].dropna().unique()
                 if isinstance(b, str) and b.lower() in ql),
                None
            )
            if brand_guess:
                which = "online" if "online" in ql else "offline"
                resp = get_total_spend_by_brand_and_channel(base_df, brand_guess, which)
                if isinstance(resp, str) and resp and not resp.lower().startswith("no "):
                    return resp
    # ==== END fast-path ====

    # --- Sidebar filters path ---
    if filters_dict:  # If filters are provided directly (from sidebar)
        filters = {k: v for k, v in filters_dict.items() if v != "All" and v is not None}
        if not filters:
            st.session_state.current_filtered_df = None
            total_spend = st.session_state.processed_df["Amount"].sum()
            return f"No specific filters selected. The total spend for the entire dataset is **${total_spend:,.2f}**."
    elif query:  # Extract filters from query text
        brands_in_data    = base_df["Brand"].dropna().unique().tolist()    if "Brand"   in base_df.columns else []
        sectors_in_data   = base_df["Sector"].dropna().unique().tolist()   if "Sector"  in base_df.columns else []
        sources_in_data   = base_df["Source"].dropna().unique().tolist()   if "Source"  in base_df.columns else []
        categories_in_data= base_df["Category"].dropna().unique().tolist() if "Category"in base_df.columns else []
        products_in_data  = base_df["Product"].dropna().unique().tolist()  if "Product" in base_df.columns else []
        media_in_data     = base_df["Media"].dropna().unique().tolist()    if "Media"   in base_df.columns else []
        agencies_in_data  = base_df["Media Agency"].dropna().unique().tolist() if "Media Agency" in base_df.columns else []
        producers_in_data = base_df["Producer"].dropna().unique().tolist() if "Producer"in base_df.columns else []
        countries_in_data = base_df["Country"].dropna().unique().tolist()  if "Country" in base_df.columns else []
        channels_in_data  = base_df["Channel"].dropna().unique().tolist()  if "Channel" in base_df.columns else []

        filters = extract_filters_from_query(
            query, base_df.columns.tolist(),
            brands_in_data, sectors_in_data, sources_in_data,
            categories_in_data, products_in_data, media_in_data,
            agencies_in_data, producers_in_data, countries_in_data,
            channels_in_data
        )

        # contextual understanding
        contextual_filters = infer_contextual_filters(query, base_df, st.session_state.column_mapping)
        filters.update(contextual_filters)

        # Detect all months mentioned in the query -> list of months
        months_in_query = [month for month in calendar.month_name[1:] if month.lower() in ql]
        if months_in_query and 'Month_str' in base_df.columns:
            filters['Month_str'] = months_in_query
            logger.info(f"Month filter(s) applied from query: {months_in_query}")

        # Override conflicting filters with contextual understanding
        if contextual_filters:
            for key in contextual_filters:
                filters[key] = contextual_filters[key]

        # Optional cleanup: remove Brand/Producer if not supported by context
        for bad_col in ['Brand', 'Producer']:
            actual_col = st.session_state.column_mapping.get(bad_col)
            if actual_col in filters and actual_col not in contextual_filters:
                del filters[actual_col]

        if not filters:
            st.session_state.current_filtered_df = None
            total_spend = st.session_state.processed_df["Amount"].sum()
            return ( "I couldn't identify any specific filters (like brand, sector, year, etc.) in your query to perform a detailed analysis. "
                     f"The total spend for the entire dataset is **${total_spend:,.2f}**. Can you please be more specific?" )
    else:
        return "No query or filters provided for complex analysis."

    logger.info(f"DEBUG: Filters identified/received in handle_complex_query: {filters}")
    st.sidebar.write(f"DEBUG - handle_complex_query: Filters identified/received: {filters}")

    # --- Apply filters & respond ---
    try:
        filtered_df = base_df.copy()
        for key, value in filters.items():
            if key in filtered_df.columns:
                filtered_df = filtered_df[filtered_df[key] == value]

        st.session_state.current_filtered_df = filtered_df.copy()

        spend_total = filtered_df["Amount"].sum() if "Amount" in filtered_df.columns else 0
        response = format_spend_response(spend_total, filters)
        return response

    except Exception as e:
        logger.error(f"Error during spend calculation: {e}")
        return "An error occurred while trying to perform that analysis. I'll try to give a general answer."


    # Apply the extracted filters to the base_df

    filtered_df = apply_dynamic_filters(base_df, filters)
    logger.info(f"DEBUG: Filtered df shape after apply_dynamic_filters: {filtered_df.shape}")
    logger.info(f"DEBUG: Columns of DataFrame after filtering: {filtered_df.columns.tolist()}")
    st.sidebar.write(f"DEBUG - handle_complex_query: Filtered df shape after apply_dynamic_filters: {filtered_df.shape}")


    if filtered_df.empty:
        st.session_state.current_filtered_df = None # Clear if no data matches
        filter_details_str = ", ".join([f"{col}: {val}" for col, val in filters.items()])
        return f"No data found matching your criteria: {filter_details_str}. Please try different filters."

    # --- Grouping and Visualization Logic ---
    group_by_column = None
    top_n_request = None # (n, ascending)
    
    if query: # Only attempt to find grouping or top N if a query string was provided
        # Check for specific "highest/lowest/top N X" patterns
        top_n_match = re.search(r'(?:highest|lowest|top|least)\s*(?:(\d+)\s*)?([a-zA-Z\s]+?)(?:\s+by\s+spend|\s+spending)?', query, re.IGNORECASE)
        if top_n_match:
            n_val = int(top_n_match.group(1)) if top_n_match.group(1) else 1 # Default to 1 for highest/lowest
            potential_col_raw = top_n_match.group(2).strip()
            ascending_val = "lowest" in top_n_match.group(0).lower() or "least" in top_n_match.group(0).lower()

            # Try to map potential_col_raw to an actual column in filtered_df
            for col in filtered_df.columns:
                if fuzz.ratio(potential_col_raw.lower(), col.lower()) >= config.FUZZY_MATCH_THRESHOLD and col in OPTIONAL_ANALYTICAL_COLUMNS:
                    group_by_column = col
                    top_n_request = (n_val, ascending_val)
                    logger.info(f"Detected top N request: {top_n_request} for column: {group_by_column}")
                    st.sidebar.write(f"DEBUG - handle_complex_query: Detected top N request: {top_n_request}")
                    break
            if not group_by_column:
                logger.warning(f"Could not find a valid column for top N analysis from query: '{potential_col_raw}'")
                st.sidebar.write(f"DEBUG - handle_complex_query: Could not find valid column for top N: '{potential_col_raw}')")

        # If no specific "top N" request, check for general grouping keywords
        if not group_by_column:
            grouping_match = re.search(r'(?:by|for each|grouped by|break down by)\s+([a-zA-Z\s]+)', query, re.IGNORECASE)
            if grouping_match:
                potential_group_col_raw = grouping_match.group(1).strip()
                # Try to map potential_group_col_raw to an actual column in filtered_df
                for col in filtered_df.columns:
                    if fuzz.ratio(potential_group_col_raw.lower(), col.lower()) >= config.FUZZY_MATCH_THRESHOLD and col in OPTIONAL_ANALYTICAL_COLUMNS:
                        group_by_column = col
                        break
                if group_by_column:
                    logger.info(f"Detected grouping by: {group_by_column}")
                    st.sidebar.write(f"DEBUG - handle_complex_query: Detected grouping by: {group_by_column}")
                else:
                    logger.warning(f"Could not find a valid column to group by from query: '{potential_group_col_raw}'")
                    st.sidebar.write(f"DEBUG - handle_complex_query: Could not find valid grouping column: '{potential_group_col_raw}')")


    if group_by_column and group_by_column in filtered_df.columns:
        # Perform grouping and display results
        grouped_data = filtered_df.groupby(group_by_column)["Amount"].sum().reset_index()
        grouped_data.columns = [group_by_column, 'Total Spend'] # Rename columns for clarity

        # Apply top N logic if requested
        if top_n_request:
            n_val, ascending_val = top_n_request
            filtered_grouped_data = grouped_data[grouped_data['Total Spend'] > 0]
            grouped_data = filtered_grouped_data.sort_values("Total Spend", ascending=ascending_val).head(n_val)
            
            if grouped_data.empty:
                response_text = f"No {group_by_column.lower()} with non-zero total spend found after applying filters and top N request."
            else:
                if ascending_val:
                    response_text = f"Here are the lowest {n_val} {group_by_column.lower()} by total spend (excluding zero spend):\n"
                else:
                    response_text = f"Here are the top {n_val} {group_by_column.lower()} by total spend (excluding zero spend):\n"
                
                for index, row in grouped_data.iterrows():
                    response_text += f"- **{row[group_by_column].title()}**: ${row['Total Spend']:,.2f}\n"

                fig = px.bar(grouped_data, x=group_by_column, y='Total Spend',
                             title=f'{"Lowest" if ascending_val else "Top"} {n_val} {group_by_column.title()} by Total Spend (Excluding Zero Spend)',
                             labels={'Total Spend': 'Total Spend ($)'},
                             color=group_by_column,
                             height=400)
                st.plotly_chart(fig)

        else: # General grouping without specific top N
            if grouped_data.empty:
                response_text = f"No data found for grouping by '{group_by_column}' after applying filters."
            else:
                response_text = f"Here's the spend breakdown by **{group_by_column.title()}**:\n"
                for index, row in grouped_data.iterrows():
                    response_text += f"- **{row[group_by_column].title()}**: ${row['Total Spend']:,.2f}\n"

                fig = px.bar(grouped_data, x=group_by_column, y='Total Spend',
                             title=f'Spend Breakdown by {group_by_column.title()}',
                             labels={'Total Spend': 'Total Spend ($)'},
                             color=group_by_column,
                             height=400)
                st.plotly_chart(fig)
    else:
        # Existing summary logic (total/average/min/max)
        summary_type = "total spend" # Default
        if query: # Only check for summary type if a query string was provided
            if "average" in query.lower():
                summary_type = "average spend"
            elif "count" in query.lower():
                summary_type = "count of records"
            elif "max" in query.lower() or "highest" in query.lower():
                summary_type = "maximum spend"
            elif "min" in query.lower() or "lowest" in query.lower():
                summary_type = "minimum spend"

        result = 0
        if summary_type == "total spend":
            result = filtered_df["Amount"].sum()
            response_text = f"The total spend for your specified criteria was **${result:,.2f}**."
        elif summary_type == "average spend":
            result = filtered_df["Amount"].mean()
            response_text = f"The average spend for your specified criteria was **${result:,.2f}**."
        elif summary_type == "count of records":
            result = len(filtered_df)
            response_text = f"There are **{result:,}** records matching your criteria."
        elif summary_type == "maximum spend":
            result = filtered_df["Amount"].max()
            response_text = f"The maximum spend for your specified criteria was **${result:,.2f}**."
        elif summary_type == "minimum spend":
            result = filtered_df["Amount"].min()
            response_text = f"The minimum spend for your specified criteria was **${result:,.2f}**."

    # Store the filtered DataFrame in session state for subsequent queries
    st.session_state.current_filtered_df = filtered_df.copy() # Store a copy

    filter_details_list = [f"**{col.replace('_', ' ')}**: {val if not isinstance(val, tuple) else f'between ${val[0]:,.0f} and ${val[1]:,.0f}'}" for col, val in filters.items()]
    filter_details_summary = ", ".join(filter_details_list)
    
    if len(filters) > 0: # Always add filter summary if filters were applied
        response_text += "\n\n🔎 **Applied Filters:**\n"
        for key, value in filters.items():
            response_text += f"- **{key.replace('_', ' ')}**: {value if not isinstance(value, tuple) else f'between ${value[0]:,.0f} and ${value[1]:,.0f}'}\n"
    
    response_text += "\n\nWould you like to:"
    response_text += "\n- See the daily trend for a specific brand?"
    response_text += "\n- Compare total spend by brand?"
    response_text += "\n- See top sectors?"
    logger.info(f"DEBUG: Final response_text generated by handle_complex_query: {response_text}")
    st.sidebar.write(f"DEBUG - handle_complex_query: Final response_text: {response_text}")
    return response_text


# Define the missing handle functions
# These handlers are now simplified to just call the generic get_top_n_by_column
def handle_top_sectors(query):
    """
    FIXED: Parses query to extract 'n' and 'ascending' for top/lowest sectors 
    and calls the new get_top_sectors function.
    """
    n_match = re.search(r'(?:top|highest|lowest|least)\s*(\d+)\s*sectors', query, re.IGNORECASE)
    # Default to 3 as per user's request context
    n = int(n_match.group(1)) if n_match and n_match.group(1) else 3
    
    ascending = "lowest" in query.lower() or "least" in query.lower()
    
    # Call the new, specific function for sectors
    return get_top_sectors(n=n, ascending=ascending)


def handle_top_brands(query):
    """Parses query to extract 'n' and 'ascending' for top/lowest brands and calls get_top_n_by_column."""
    n_match = re.search(r'(?:top|highest)\s*(\d+)\s*brands', query, re.IGNORECASE)
    n = int(n_match.group(1)) if n_match else 5
    
    ascending = "lowest" in query.lower() or "least" in query.lower()

    return get_top_n_by_column("Brand", n, ascending)


def handle_compare_online_offline_query(query):
    """Parses query to extract year for online/offline comparison and calls compare_online_offline_func."""
    year = None
    year_match = re.search(r'in\s+(\d{4})', query, re.IGNORECASE)
    if year_match:
        year = int(year_match.group(1))
    
    return compare_online_offline_func(year)


# Embed base intents once
if model:
    base_embeddings = load_sentence_transformer_model().encode(config.BASE_INTENTS, convert_to_tensor=True)
else:
    base_embeddings = None

# Map intents to functions (for SentenceTransformer path)
# keys in intent_funcs directly match the strings in config.BASE_INTENTS
intent_funcs = {
    "total spend by brand": lambda q: get_total_spend_by_column("Brand", q), # Updated to generic
    "spend by company": lambda q: get_total_spend_by_column("Brand", q), # Updated to generic
    "spend of a brand": lambda q: get_total_spend_by_column("Brand", q), # Updated to generic
    "top sectors by spend": lambda q: handle_top_sectors(q),
    "top 3 sectors": lambda q: handle_top_sectors(q),
    "top 5 sectors": lambda q: handle_top_sectors(q),
    "top brands by spend": lambda q: handle_top_brands(q),
    "compare online offline spend": lambda q: handle_compare_online_offline_query(q),
    "monthly trend for brand": lambda q: handle_trend(q),
    "daily trend for brand": lambda q: handle_trend(q),
    "highest spend day": get_highest_spend_day,
    "compare two months": lambda q: handle_month_comparison(q),
    "predict next week spend": forecast_next_7_days_func,
    "forecast spend next week": forecast_next_7_days_func,
    "compare two brands": lambda q: compare_two_brands(q), # Corrected typo here
    "list all brands": list_all_brands,
    "list all sectors": list_all_sectors, # Corrected typo here
    "highest offline spend by a brand": top_brand_by_channel,
    "general question about marketing": lambda q, conv: get_openai_response(q, conv),
    "what is marketing spend analysis": lambda q, conv: get_openai_response(q, conv),
    "tell me about AI in marketing.": lambda q, conv: get_openai_response(q, conv),
    "calculate sum for column": lambda q: calculate_column_statistic(extract_column_name(q), 'sum'),
    "what is the average of column": lambda q: calculate_column_statistic(extract_column_name(q), 'mean'),
    "find the maximum value in column": lambda q: calculate_column_statistic(extract_column_name(q), 'max'),
    "show me a histogram of column": lambda q: plot_column_distribution(extract_column_name(q), 'histogram'),
    "list unique values in column": lambda q: list_unique_values(extract_column_name(q)),
    "how many unique values in column": lambda q: list_unique_values(extract_column_name(q)), # Corrected from list_unique_statistic
    "complex data query": lambda q: handle_complex_query(query=q) # Ensure query is passed for parsing
}

# Define tools for LLM Function Calling
AVAILABLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_total_spend_by_column",
            "description": "Calculates the total marketing spend for a specific value within any categorical column (e.g., total spend for 'Instagram' in the 'Channel' column, or 'Retail' in the 'Sector' column). Use this for queries like 'total spend on Instagram', 'total spend in Retail sector', 'total spend by Brand X'. Only use if the specified column is available.",
            "parameters": {
                "type": "object",
                "properties": {
                    "column_name": {
                        "type": "string",
                        "description": "The name of the categorical column (e.g., 'Brand', 'Channel', 'Sector', 'Source', 'Category', 'Product'). Infer this directly from the user's query."
                    },
                    "value": {
                        "type": "string",
                        "description": "The specific value within the column (e.g., 'Instagram', 'Retail', 'Sharjah Co-op'). Infer this directly from the user's query."
                    }
                },
                "required": ["column_name", "value"]
            }
        }
    },
    {
        "type": "function",
        "name": "handle_complex_query",
        "description": "Handles complex data queries involving multiple filters (e.g., brand, sector, source, year, month, country, channel, etc.) and various summary types (total spend, average spend, count, min, max), AND/OR grouping/breakdown requests (e.g., 'group by month', 'break down by source'), AND/OR 'highest/lowest/top N' requests for any categorical column when combined with other filters (e.g., 'highest spending digital channel', 'top 5 brands in 2024'). DO NOT use this tool for simple 'top N X' requests that do not include any other filtering criteria; for those, use `get_top_n_by_column`.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The original user query string (e.g., 'What was the online spend by Etisalat in 2023 for the Retail sector, grouped by month?', 'Show me the average spend for Brand X in Country Y, broken down by category', 'highest spending digital channel', 'top 5 brands in 2024'). This is used when the function needs to parse filters from a natural language query.",
                    "nullable": True # Make query nullable as filters_dict can be used
                },
                "filters_dict": {
                    "type": "object",
                    "description": "A dictionary of pre-parsed filters (e.g., {'Brand': 'Jaguar', 'Channel': 'Instagram'}). This is used when filters come directly from UI selections.",
                    "additionalProperties": {"type": "string"}, # Assuming filter values are strings for simplicity
                    "nullable": True # Make filters_dict nullable as query can be used
                }
            },
            "oneOf": [ # Ensure either query or filters_dict is provided, but not both at once by the LLM
                {"required": ["query"]},
                {"required": ["filters_dict"]}
            ]
        }
    },
    {
        "type": "function",
        "name": "get_top_n_by_column",
        "description": "Returns the top N (highest spend) or lowest N (lowest spend) items by total marketing spend for ANY specified categorical column (e.g., 'Brand', 'Sector', 'Channel', 'Source', 'Category', 'Media', 'Agency', 'Producer', 'Country'). Use this ONLY for simple 'top N X' requests that do not include any other filtering criteria (e.g., no year, no other brand filter, no source filter).",
        "parameters": {
            "type": "object",
            "properties": {
                "column_name": {
                    "type": "string",
                    "description": "The name of the categorical column to analyze (e.g., 'Brand', 'Sector', 'Channel', 'Source', 'Category'). Must be one of the available columns in the dataset. Infer this directly from the user's query."
                },
                "n": {
                    "type": "integer",
                    "description": "The number of items to return (e.g., 1, 3, 5, 10). Defaults to 5 if not specified."
                },
                "ascending": {
                    "type": "boolean",
                    "description": "Set to true to get the lowest spending items. Defaults to false (highest spending)."
                }
            },
            "required": ["column_name"]
        }
    },
    {
        "type": "function",
        "name": "compare_online_offline_func", 
        "description": "Compares total marketing spend between online and offline sources. Can filter by year. Use this ONLY for general online/offline comparisons without any specific brand mentioned. If a brand is mentioned, use 'handle_complex_query' with 'source' as a filter. Only use if the 'Source' column is available.",
        "parameters": {
            "type": "object",
            "properties": {
                "year": {
                    "type": "integer",
                    "description": "The year to filter the comparison by (e.g., 2023, 2024). Optional. If not provided, compares across all years."
                }
            }
        }
    },
    {
        "type": "function",
        "name": "get_daily_trend_data", # Renamed function
        "description": "Generates the daily spend trend for a specific brand, sector, or source. Requires both 'filter_by' column and 'value' to be specified.",
        "parameters": {
            "type": "object",
            "properties": {
                "filter_by": {
                    "type": "string",
                    "description": "The column to filter by ('Brand', 'Sector', or 'Source'). Must be one of the available columns in the dataset."
                },
                "value": {
                    "type": "string",
                    "description": "The specific value within the filter_by column (e.g., 'Oasis Mall' for 'Brand', 'Retail' for 'Sector', 'Online' for 'Source')."
                }
            },
            "required": ["filter_by", "value"]
        }
    },
    {
        "type": "function",
        "name": "get_highest_spend_day",
        "description": "Identifies and reports the single highest spend day, highest spend weekday, and highest total spend month in the dataset. No parameters needed.",
        "parameters": {"type": "object", "properties": {}}
    },
    {
        "type": "function",
        "name": "compare_months",
        "description": "Compares total marketing spend between two specified months and years.",
        "parameters": {
            "type": "object",
            "properties": {
                "month1": {"type": "integer", "description": "The month number (1-12) for the first period."},
                "year1": {"type": "integer", "description": "The year for the first period."},
                "month2": {"type": "integer", "description": "The month number (1-12) for the second period."},
                "year2": {"type": "integer", "description": "The year for the second period."}
            },
            "required": ["month1", "year1", "month2", "year2"]
        }
    },
    {
        "type": "function",
        "name": "forecast_next_7_days_func",
        "description": "Forecasts marketing spend for the next 7 days using historical data and generates a plot. No parameters needed.",
        "parameters": {"type": "object", "properties": {}}
    },
    {
        "type": "function",
        "name": "compare_two_brands",
        "description": "Compares total marketing spend between two specific brands and generates a bar chart. Only use if the 'Brand' column is available.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The original user query containing the two brand names to compare (e.g., 'Compare Dubai Duty Free vs Daiso Japan')."
                }
            },
            "required": ["query"]
        }
    },
    {
        "type": "function",
        "name": "list_all_brands",
        "description": "Lists all unique brand names available in the dataset. Only use if the 'Brand' column is available.",
        "parameters": {"type": "object", "properties": {}}
    },
    {
        "type": "function",
        "name": "list_all_sectors",
        "description": "Lists all unique sector names available in the dataset. Only use if the 'Sector' column is available.",
        "parameters": {"type": "object", "properties": {}}
    },
    {
        "type": "function",
        "name": "calculate_column_statistic",
        "description": "Calculates a specified statistic (sum, mean, min, max, count) for any numeric column in the dataset.",
        "parameters": {
            "type": "object",
            "properties": {
                "column_name": {
                    "type": "string",
                    "description": "The name of the column to analyze. Must be one of the available columns in the dataset."
                },
                "statistic_type": {
                    "type": "string",
                    "enum": ["sum", "mean", "min", "max", "count"],
                    "description": "The type of statistic to calculate."
                }
            },
            "required": ["column_name", "statistic_type"]
        }
    },
    {
        "type": "function",
        "name": "plot_column_distribution",
        "description": "Generates a plot (bar for categorical, histogram for numeric) for the distribution of values in a given column.",
        "parameters": {
            "type": "object",
            "properties": {
                "column_name": {
                    "type": "string",
                    "description": "The name of the column to plot. Must be one of the available columns in the dataset."
                },
                "plot_type": {
                    "type": "string",
                    "enum": ["bar", "histogram"],
                    "description": "The type of plot to generate. 'bar' is suitable for categorical data, 'histogram' for numeric. Defaults to 'bar'."
                }
            },
            "required": ["column_name"]
        }
    },
    {
        "type": "function",
        "name": "list_unique_values",
        "description": "Lists all unique values present in a specified column.",
        "parameters": {
            "type": "object",
            "properties": {
                "column_name": {
                    "type": "string",
                    "description": "The name of the column to list unique values from. Must be one of the available columns in the dataset."
                }
            },
            "required": ["column_name"]
        }
    }
]

# Call LLM for tool selection and argument extraction
def call_llm_for_function_call(user_query, available_tools, conversation_history, dataframe_columns,
                               brands_in_data, sectors_in_data, sources_in_data, categories_in_data,
                               products_in_data, media_in_data, agencies_in_data, producers_in_data,
                               countries_in_data, channels_in_data): 
    """
    Uses OpenAI's GPT to determine which tool to call and to extract its arguments.
    Provides the LLM with the current DataFrame columns.
    """
    logger.info(f"DEBUG: call_llm_for_function_call received query: '{user_query}'")
    st.sidebar.write(f"DEBUG - current_filtered_df before reset: {st.session_state.current_filtered_df.shape if st.session_state.current_filtered_df is not None else 'None'}")

    # --- IMPORTANT: Reset current_filtered_df at the start of every new natural language query ---
    # This ensures that each new query starts with the full dataset unless explicitly filtered by the query itself
    # or by the sidebar. Sidebar button will call handle_complex_query directly with filters_dict.
    if st.session_state.processed_df is not None:
        st.session_state.current_filtered_df = st.session_state.processed_df.copy()
        logger.info("Resetting current_filtered_df to full processed_df for new LLM query.")
    st.sidebar.write(f"DEBUG - current_filtered_df after reset: {st.session_state.current_filtered_df.shape if st.session_state.current_filtered_df is not None else 'None'}")

    query_lower = user_query.lower()
    st.sidebar.write(f"DEBUG - Original query_lower: '{query_lower}'")

    # --- HIGH PRIORITY PRE-CHECKS FOR DIRECT TOOL CALLS ---

    # NEW PRE-CHECK: Compare two brands - ABSOLUTE HIGHEST PRIORITY
    compare_brands_match = re.search(r'compare(?:\s+spend)?(?:\s+of)?\s*(.*?)\s+vs\s+(.*)', query_lower)
    if compare_brands_match:
        brand1_raw = compare_brands_match.group(1).strip()
        brand2_raw = compare_brands_match.group(2).strip()
        
        st.sidebar.write(f"DEBUG - Compare brands match found. Brand1 raw: '{brand1_raw}', Brand2 raw: '{brand2_raw}'")

        # Get all brands from the dataset
        all_brands_in_data = [str(e) for e in st.session_state.processed_df["Brand"].dropna().unique().tolist()] if "Brand" in st.session_state.processed_df.columns else []

        found_brand1, score1 = safe_fuzzy_extract_one(brand1_raw, all_brands_in_data, threshold=90) # High threshold for direct brand match
        found_brand2, score2 = safe_fuzzy_extract_one(brand2_raw, all_brands_in_data, threshold=90) # High threshold for direct brand match

        if found_brand1 and found_brand2 and found_brand1.lower() != found_brand2.lower():
            logger.info(f"DEBUG: HIT - Direct call for compare_two_brands. Brands: {found_brand1}, {found_brand2}")
            st.sidebar.write(f"DEBUG - Hardcoded Tool Call: compare_two_brands, Args: {{'query': '{user_query}'}}")
            return {
                "function_name": "compare_two_brands",
                "args": {"query": user_query}
            }
        logger.warning(f"DEBUG: Pre-check for 'compare two brands' failed. Found brands: {found_brand1}, {found_brand2}")


    # Pre-check for "Compare online and offline spend"
    if re.search(r'compare\s+(?:online|offline)\s+and\s+(?:online|offline)\s+spend', query_lower) or \
       re.search(r'(?:online|offline)\s+vs\s+(?:online|offline)\s+spend', query_lower):
        
        # Check if there are any *other* specific entities (brands, sectors, products) in the query.
        # If there are, then it's a complex query, not a simple online/offline comparison.
        other_entities_found = False
        for col_name, entities in {
            "Brand": brands_in_data, "Sector": sectors_in_data, "Product": products_in_data,
            "Category": categories_in_data, "Media": media_in_data, "Media Agency": agencies_in_data, # Corrected "Agency" to "Media Agency"
            "Producer": producers_in_data, "Country": countries_in_data, "Channel": channels_in_data
        }.items():
            # Exclude 'Source' column from this check, as 'online'/'offline' are part of it.
            if col_name == "Source":
                continue
            if col_name in dataframe_columns:
                # Use safe_fuzzy_extract_one for this check too, with a higher threshold
                matched_entity, score = safe_fuzzy_extract_one(query_lower, [str(e) for e in entities if pd.notna(e)], threshold=90) # Increased threshold
                if matched_entity:
                    # Make sure the matched entity is not 'online' or 'offline' itself
                    if matched_entity.lower() not in ['online', 'offline', 'digital', 'traditional']:
                        other_entities_found = True
                        break
        
        if not other_entities_found:
            year = None
            year_match = re.search(r'\b(?:in|for)\s+(\d{4})\b', query_lower)
            if year_match:
                year = int(year_match.group(1))
            
            logger.info(f"DEBUG: HIT - Direct call for compare_online_offline_func. Year: {year}")
            st.sidebar.write(f"DEBUG - Hardcoded Tool Call: compare_online_offline_func, Args: {{'year': {year}}}")
            return {
                "function_name": "compare_online_offline_func",
                "args": {"year": year}
            }
        else:
            logger.info(f"DEBUG: Online/Offline query detected, but other entities found. Will route via LLM/complex handler.")
            st.sidebar.write(f"DEBUG - Online/Offline query detected, but other entities found. Routing to LLM/complex handler.")


    # Pre-check for "total spend on/by/through X" (for the new generic function)
    total_spend_match = re.search(r'total spend (?:on|by|through)\s+([a-zA-Z0-9\s]+)', query_lower)
    if total_spend_match:
        target_value_raw = total_spend_match.group(1).strip()
        st.sidebar.write(f"DEBUG - Total spend match found. Raw value: '{target_value_raw}'")

        # New: Store all potential matches with their scores and column names
        potential_matches_across_columns = []

        # Prioritize Brand, then Source, then Sector, then Channel, then other analytical columns
        # This order is crucial for cases like "Sharjah Co-op" vs "Sharjah Sports TV" or "online" vs "online banking"
        prioritized_total_spend_cols = ["Brand", "Source", "Sector", "Channel", "Product", "Category", "Media", "Media Agency", "Producer", "Country"]
        
        for col_name in prioritized_total_spend_cols:
            if col_name in dataframe_columns:
                entities_in_column = [str(e) for e in st.session_state.processed_df[col_name].dropna().unique().tolist()]
                
                # Use safe_fuzzy_extract_one for this check
                matched_entity, score = safe_fuzzy_extract_one(target_value_raw, entities_in_column)
                
                if matched_entity and score >= config.FUZZY_MATCH_THRESHOLD:
                    potential_matches_across_columns.append({
                        "column_name": col_name,
                        "value": matched_entity,
                        "score": score
                    })
        
        st.sidebar.write(f"DEBUG - Potential matches across columns: {potential_matches_across_columns}")

        # Now, evaluate all collected potential matches
        best_match = None
        highest_score_found = -1
        
        # First pass: Look for an exact match in 'Brand'
        for match in potential_matches_across_columns:
            if match["column_name"] == "Brand" and match["value"].lower() == target_value_raw.lower():
                best_match = match
                logger.info(f"DEBUG: Exact match found for Brand: {best_match['value']}")
                st.sidebar.write(f"DEBUG - Exact Brand match found: {best_match['value']}")
                break # Found the best possible match, stop
        
        if not best_match: # If no exact brand match, find the the best scoring match overall
            for match in potential_matches_across_columns:
                if match["score"] > highest_score_found:
                    highest_score_found = match["score"]
                    best_match = match
                elif match["score"] == highest_score_found and best_match:
                    # Tie-breaker: if scores are equal, prioritize based on the `prioritized_total_spend_cols` order
                    current_col_priority = prioritized_total_spend_cols.index(match["column_name"])
                    best_match_col_priority = prioritized_total_spend_cols.index(best_match["column_name"])
                    if current_col_priority < best_match_col_priority: # Lower index means higher priority
                        best_match = match
        
        if best_match:
            logger.info(f"DEBUG: HIT - Direct call for get_total_spend_by_column. Column: {best_match['column_name']}, Value: {best_match['value']} (Score: {best_match['score']})")
            st.sidebar.write(f"DEBUG - Hardcoded Tool Call: get_total_spend_by_column, Args: {{'column_name': '{best_match['column_name']}', 'value': '{best_match['value']}'}}")
            return {
                "function_name": "get_total_spend_by_column",
                "args": {"column_name": best_match["column_name"], "value": best_match["value"]}
            }
        logger.warning(f"DEBUG: Pre-check for 'total spend on X' failed to find a matching column/value for '{target_value_raw}'")


    # Pre-check for "top N X by spend" - REFINED LOGIC
    # Use a more general search for the core pattern
    # Adjusted regex to capture the full "top N X by spend" phrase more reliably
    core_top_n_match = re.search(r'(top|lowest|least|highest)\s*(\d*)\s*([a-zA-Z\s]+?)(?:\s+by\s+spend|\s+spending)?', query_lower)
    
    if core_top_n_match:
        # Extract components
        n_val = int(core_top_n_match.group(2)) if core_top_n_match.group(2) else (1 if "highest" in core_top_n_match.group(1) or "lowest" in core_top_n_match.group(1) else 5)
        potential_col_raw = core_top_n_match.group(3).strip()
        ascending_val = "lowest" in core_top_n_match.group(1) or "least" in core_top_n_match.group(1)

        # Attempt to map potential_col_raw to an actual column in the dataframe
        target_top_n_column = None
        for col in OPTIONAL_ANALYTICAL_COLUMNS:
            if col in dataframe_columns and fuzz.token_set_ratio(potential_col_raw.lower(), col.lower()) >= 85:
                target_top_n_column = col
                break
        
        if target_top_n_column:
            # Get the exact matched string from the regex group 0
            matched_phrase_exact = core_top_n_match.group(0)
            
            # Remove the exact matched phrase from the query
            # Use re.sub with a word boundary to ensure it's a whole word match
            remaining_query_after_top_n = re.sub(r'\b' + re.escape(matched_phrase_exact) + r'\b', '', query_lower, 1).strip()
            
            # Define words that are allowed to remain in a simple query (filler words)
            allowed_filler_words = {"what", "are", "the", "by", "spend", "spending", "questions", "of", "in", "for", "a", "an", "and", "or", "is", "was", "show", "me", "list", "tell", "about"}
            
            # Filter out allowed filler words from the remaining query
            remaining_words_filtered = [word for word in remaining_query_after_top_n.split() if word not in allowed_filler_words]
            
            # If anything remains after filtering out filler words, then other filters are detected
            other_filters_detected = bool(remaining_words_filtered)

            st.sidebar.write(f"DEBUG - Core top N match found: '{matched_phrase_exact}'")
            st.sidebar.write(f"DEBUG - Remaining query after top N removal: '{remaining_query_after_top_n}'")
            st.sidebar.write(f"DEBUG - Remaining words filtered (after filler removal): {remaining_words_filtered}")
            st.sidebar.write(f"DEBUG - other_filters_detected: {other_filters_detected}")

            if not other_filters_detected:
                logger.info(f"DEBUG: HIT - Direct call for get_top_n_by_column. Column: {target_top_n_column}, N: {n_val}, Ascending: {ascending_val}")
                st.sidebar.write(f"DEBUG - Hardcoded Tool Call: get_top_n_by_column, Args: {{'column_name': '{target_top_n_column}', 'n': {n_val}, 'ascending': {ascending_val}}}")
                return {
                    "function_name": "get_top_n_by_column",
                    "args": {"column_name": target_top_n_column, "n": n_val, "ascending": ascending_val}
                }
    
    # --- END HIGH PRIORITY PRE-CHECKS ---


    # If no direct pre-check hit, proceed with LLM tool selection
    # The rest of this function (LLM tool selection) remains largely the same,
    # but the system prompt will be updated to reflect the stricter pre-checks.
    
    detected_filter_types = set()
    
    # Check for specific entities in analytical columns
    entity_column_data = {
        "Brand": brands_in_data,
        "Source": sources_in_data,
        "Sector": sectors_in_data,
        "Category": categories_in_data,
        "Product": products_in_data,
        "Media": media_in_data,
        "Media Agency": agencies_in_data, # Corrected "Agency" to "Media Agency"
        "Producer": producers_in_data,
        "Country": countries_in_data,
        "Channel": channels_in_data 
    }

    for col_name, entities in entity_column_data.items():
        if col_name in dataframe_columns: # Only check if column exists in user's data
            # Use safe_fuzzy_extract_one to check for presence of any entity from this column in the query
            matched_entity, score = safe_fuzzy_extract_one(query_lower, [str(e) for e in entities if pd.notna(e)])
            if matched_entity and score >= config.FUZZY_MATCH_THRESHOLD:
                # IMPORTANT: Exclude "marketing" as a category filter here
                if col_name == "Category" and matched_entity.lower() == "marketing":
                    logger.debug(f"Explicitly ignoring 'marketing' as a Category filter for query '{query_lower}'.")
                    continue
                detected_filter_types.add(col_name)

    # Check for year and month keywords
    if re.search(r'\b\d{4}\b', query_lower):
        detected_filter_types.add("Year")
    if re.search(r'\b(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b', query_lower):
        detected_filter_types.add("Month")
    
    # Check for explicit grouping keywords
    if re.search(r'(?:group by|break down by|by|for each)\s+([a-zA-Z\s]+)', query_lower):
        detected_filter_types.add("Grouping")

    # Check for amount ranges/comparisons
    if re.search(r'(?:between|greater than|more than|over|less than|under)\s+\$*(\d[\d,.]*)', query_lower):
        detected_filter_types.add("Amount_Range_or_Comparison")

    # Force handle_complex_query if:
    # 1. Two or more distinct filter types are detected.
    # 2. A grouping request is present.
    # 3. An amount range/comparison is present.
    # 4. 'online' or 'offline' is mentioned alongside a brand or sector (this is a specific complex case).

    force_complex_query = False
    
    if len(detected_filter_types) >= 2:
        force_complex_query = True
    
    if "Grouping" in detected_filter_types:
        force_complex_query = True
    if "Amount_Range_or_Comparison" in detected_filter_types:
        force_complex_query = True
    if "Source" in detected_filter_types and ("Brand" in detected_filter_types or "Sector" in detected_filter_types):
        force_complex_query = True

        
    if force_complex_query:
        logger.info(f"DEBUG: HIT - Force complex query due to multiple filters/grouping.")
        logger.info(f"Multiple filter types or specific combination detected ({detected_filter_types}). Forcing routing to handle_complex_query.")
        st.sidebar.write(f"DEBUG - Hardcoded Tool Call: handle_complex_query (multiple filters), Args: {{'query': '{user_query}'}}") # ADD THIS DEBUG LINE
        return {
            "function_name": "handle_complex_query",
            "args": {"query": user_query}
        }

    system_message_content = f"""You are an *extremely precise* marketing spend data analyst. Your *sole mission* is to select the *single, most appropriate tool* and extract *ALL and ONLY* the necessary parameters from the user's query.

    Available columns in the user's dataset are: {dataframe_columns}.

    **CRITICAL COLUMN SEMANTICS AND EXAMPLES:**
    - **Brand**: Specific company or product name (e.g., 'BMW', 'Jaguar', 'Sharjah Co-op').
    - **Channel**: Specific advertising platforms or channels (e.g., 'Instagram', 'Facebook', 'YouTube', 'TV', 'Radio', 'Website', 'LinkedIn', 'Google Ads', 'Bing Ads', 'Sharjah TV').
    - **Source**: General origin of spend (e.g., 'Online', 'Offline', 'Digital', 'Traditional').
    - **Category**: Broader classification of product/service or *non-marketing* type (e.g., 'Automotive', 'Apparel', 'Electronics', 'Home Goods'). **The word 'marketing' in a query generally indicates the *topic* of the query, not a filter value for this column. You MUST NOT use 'marketing' as a filter value for the 'Category' column unless the user explicitly refers to a specific named marketing category that exists in the data (e.g., 'marketing campaign category').**
    - **Product**: The specific product name or SKU.
    - **Amount**: Monetary value.
    - **Month**: Date/period.
    - Other columns: {', '.join([col for col in dataframe_columns if col not in ['Brand', 'Channel', 'Source', 'Category', 'Product', 'Amount', 'Month']])}.

    **ABSOLUTE TOOL SELECTION PROTOCOL (READ CAREFULLY AND FOLLOW WITHOUT DEVIATION):**

    1.  **RULE #1: COMPLEX DATA QUERIES (`handle_complex_query`) - FOR QUERIES WITH MULTIPLE FILTERS, EXPLICIT AGGREGATIONS (beyond simple total), OR GROUPING**
        * **TRIGGER:** Select this tool *ONLY* if the user's query *explicitly* requires:
            * **Two or more distinct filtering criteria** (e.g., "brand X in year Y", "sector A by source B", "brand X on channel Y", "brand X product Y on channel Z").
            * **Any request for grouping or breaking down data** (e.g., "group by month", "break down by source", "by country", "for each agency").
            * **Specific aggregation types other than a simple total for a single entity** (e.g., "average spend", "count of records", "max spend", "min spend", or "total spend" combined with *multiple* filters or grouping).
            * **Amount range or comparison** (e.g., "spend between 1000 and 5000", "spend greater than 1000").
            * **A 'highest/lowest/top N' request for a column *when combined with other filters*** (e.g., "highest spending digital channel", "top 5 brands in 2024").
        * **ACTION:** You **MUST IMMEDIATELY AND EXCLUSIVELY** select the `handle_complex_query` tool.
        * **PARAMETER EXTRACTION (CRITICAL):** Pass the *entire original user query string* as the `query` argument to `handle_complex_query`. The internal function will handle parsing all individual filters and grouping dimensions based on the CRITICAL COLUMN SEMANTICS above.
        * **CRITICAL NEGATIVE CONSTRAINT:** If a query can be fully answered by a *single, more specific tool* (e.g., `get_total_spend_by_column`, `compare_online_offline_func`, `get_daily_trend_data`, `forecast_next_7_days_func`, `calculate_column_statistic` for a single column/statistic, `list_all_brands`, `list_all_sectors`, or `get_top_n_by_column` for a *simple* 'top/lowest N X' request **that does not include any other filtering criteria**), you **MUST NOT** use `handle_complex_query`. Always prefer the most specific tool.

    2.  **RULE #2: TOTAL SPEND FOR A SINGLE ENTITY (`get_total_spend_by_column`)**
        * **TRIGGER:** Select this tool *ONLY IF* the query is **SOLELY** about the total spend of a **single, specific value within a single categorical column** (e.g., "total spend on Instagram", "total spend in Retail sector", "total spend by Brand X") and *does not include any other filtering criteria* (like year, month, other entities), other aggregation types, or grouping requests.
        * **EXAMPLE:** "What was the total spend by Sharjah Co-op?", "Total spend on Instagram", "Total spend in the Automotive sector."
        * **PARAMETER EXTRACTION:**
            * `column_name`: Infer the categorical column (e.g., 'Brand', 'Channel', 'Sector').
            * `value`: Extract the specific entity value (e.g., 'Sharjah Co-op', 'Instagram', 'Automotive').
        * **ABSOLUTE NEGATIVE CONSTRAINT:** If *any* additional filter (e.g., 'online', '2023', 'Retail', 'Instagram', 'engine'), aggregation type (e.g., 'average'), or grouping request (e.g., 'by month') is mentioned alongside the single entity, you **MUST NOT** use `get_total_spend_by_column`; instead, revert to `handle_complex_query` (as per Rule #1).

    3.  **RULE #3: GENERAL ONLINE/OFFLINE COMPARISON (`compare_online_offline_func`)**
        * **TRIGGER:** Select this tool *ONLY IF* the query is about a general "online vs offline spend" comparison **AND ABSOLUTELY DOES NOT MENTION ANY SPECIFIC BRAND, SECTOR, OR REQUEST GROUPING.**
        * **EXAMPLE:** "Compare online and offline spend in 2024."
        * **ABSOLUTE NEGATIVE CONSTRAINT:** If a specific `brand` or `sector` or `product` is present in the query, or if grouping is requested, you **MUST NEVER, UNDER ANY CIRCUMSTANCES, NOT EVEN ONCE,** select `compare_online_offline_func`. The 'online' or 'offline' keyword becomes a `source` filter for `handle_complex_query` in such cases.

    4.  **RULE #4: TOP N FOR ANY SINGLE CATEGORICAL COLUMN (`get_top_n_by_column`)**
        * **TRIGGER:** Select this tool *ONLY IF* the query is a **simple** request for the "top N" or "lowest N" items for a **single categorical column** (e.g., "top 5 channels", "highest spending sector", "lowest brand") **AND DOES NOT INCLUDE ANY OTHER FILTERS** (e.g., no year, no other brand filter, no source filter).
        * **EXAMPLES:** "top 5 channels by spend", "highest spending sector", "lowest 3 categories".
        * **PARAMETER EXTRACTION:**
            * `column_name`: Infer the categorical column from the query (e.g., 'Channel', 'Sector', 'Brand').
            * `n`: Extract the number (e.g., 5, 3). Default to 5 if "top" is used without a number, or 1 if "highest" or "lowest" is used without a number.
            * `ascending`: Set to `true` if "lowest" or "least" is in the query, otherwise `false`.
        * **ABSOLUTE NEGATIVE CONSTRAINT:** If *any* additional filter (e.g., "in 2024", "for Brand X") is present in the query alongside the "top N" request, you **MUST NOT** use `get_top_n_by_column`; instead, revert to `handle_complex_query` (as per Rule #1).

    5.  **RULE #5: OTHER SPECIFIC QUERIES (Tertiary Tools)**
        * `compare_two_brands`: For comparing two distinct brands (e.g., "Compare Brand X vs Brand Y"). Takes the *entire original user query string* as its `query` argument.
        * `get_daily_trend_data`: For daily/monthly trends for a specific entity (e.g., "Show me the daily trend for Oasis Mall").
        * `get_highest_spend_day`: For overall highest spend day/month.
        * `forecast_next_7_days_func`: For forecasting spend.
        * `list_all_brands`, `list_all_sectors`: For listing entities.
        * `calculate_column_statistic`, `plot_column_distribution`, `list_unique_values`: For generic column analysis on a single column.

    **Argument Formatting:**
    -   Convert month names (e.g., 'January') to their integer (e.g., 1).
    -   Extract 4-digit years.
    -   Ensure all extracted string arguments are direct matches to entities within the user's query, case-insensitively where appropriate for matching.

    **DISAMBIGUATION RULE (Channel vs. Media):** If a term in the query could refer to both a specific 'Channel' and a broader 'Media' type (e.g., 'TV' could be a generic channel or a broadcast media), and a more specific channel name is available (like 'Sharjah TV' in 'Channel'), prioritize the more specific 'Channel' match. If only the generic term is present (e.g., just 'TV' without a specific station name), consider the context to determine if it's a 'Channel' or 'Media'. For queries like "total spend through Sharjah TV", "Sharjah TV" should be mapped to the 'Channel' column.

    Your response MUST be a JSON object containing the `function_name` and `args`, or 'None' if no tool is suitable.
    """
    messages = [{"role": "system", "content": system_message_content}]
    
    context_messages_limit = getattr(config, 'OPENAI_CONTEXT_MESSAGES', 5) 
    start_index = max(0, len(conversation_history) - context_messages_limit)
    for msg in conversation_history[start_index:]:
        if msg["role"] != "assistant" or not msg["content"].startswith("Would you like to:"):
            messages.append({"role": msg["role"], "content": msg["content"]})
    
    messages.append({"role": "user", "content": user_query})

    try:
        response = client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=messages,
            tools=available_tools,
            tool_choice="auto",
            temperature=0.0,
            max_tokens=500
        )
        
        tool_calls = response.choices[0].message.tool_calls
        if tool_calls:
            tool_call = tool_calls[0]
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            logger.info(f"LLM suggested tool: {function_name} with args: {function_args}")
            st.sidebar.write(f"DEBUG - LLM Tool Call: {function_name}, Args: {function_args}") 
            return {"function_name": function_name, "args": function_args}
        else:
            logger.info("LLM did not suggest any tool.")
            st.sidebar.write("DEBUG - LLM Tool Call: None") 
            return None

    except json.JSONDecodeError:
        logger.error("LLM did not return valid JSON for tool call arguments.")
        return None
    except Exception as e:
        logger.error(f"Error during LLM tool calling: {e}", exc_info=True)
        return None

st.markdown("""
    <h1 style='text-align: center;'>📊 Kinesso Chatbot</h1>
    <p style='text-align: center;'>Upload your Excel marketing spend data to get started.</p>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx", "xls"])

df = st.session_state.processed_df 

if uploaded_file is not None:
    try:
        if st.session_state.raw_df is None or uploaded_file.name != st.session_state.get('uploaded_file_name'):
            st.session_state.raw_df = pd.read_excel(uploaded_file)
            st.session_state.uploaded_file_name = uploaded_file.name
            st.session_state.column_mapping = {col: None for col in ALL_INTERNAL_COLUMNS}
            st.session_state.data_mapped = False
            st.session_state.llm_mapping_attempted = False
            st.session_state.processed_df = None
            st.session_state.skip_mapping_ui = False
            st.session_state.current_filtered_df = None # Reset filtered data on new upload

        df_columns = list(st.session_state.raw_df.columns)

        is_already_correct_format = True
        for col in CORE_REQUIRED_COLUMNS:
            if col not in df_columns:
                is_already_correct_format = False
                break
        
        if is_already_correct_format:
            for col in OPTIONAL_ANALYTICAL_COLUMNS:
                if col in df_columns: # Check if the column exists in the raw_df
                    pass # If it exists, it's fine for this check.
                else:
                    pass
        
        if is_already_correct_format and not st.session_state.data_mapped:
            st.session_state.skip_mapping_ui = True
            for col in ALL_INTERNAL_COLUMNS:
                if col in df_columns:
                    st.session_state.column_mapping[col] = col
                else:
                    st.session_state.column_mapping[col] = None # Corrected typo here

            with st.spinner("File format recognized! Loading data..."):
                st.session_state.processed_df = load_and_map_data(uploaded_file, st.session_state.column_mapping)
                if st.session_state.processed_df is not None:
                    st.session_state.data_loaded = True
                    st.session_state.data_mapped = True
                    st.success("✅ File format recognized and data loaded successfully! You can now ask questions.")
                    # Initialize current_filtered_df with the full processed_df after successful load
                    st.session_state.current_filtered_df = st.session_state.processed_df.copy()
                    
                    summary_text = f"Loaded {len(st.session_state.processed_df)} rows of data from {st.session_state.processed_df['Month'].min().strftime('%B %Y')} to {st.session_state.processed_df['Month'].max().strftime('%B %Y')}."
                    if "Brand" in st.session_state.processed_df.columns:
                        summary_text += f" Found {st.session_state.processed_df['Brand'].nunique()} unique brands."
                    if "Sector" in st.session_state.processed_df.columns:
                        summary_text += f" Found {st.session_state.processed_df['Sector'].nunique()} unique sectors."
                    st.info(summary_text)
                    st.rerun()
                else:
                    st.session_state.data_loaded = False
                    st.session_state.data_mapped = False
        # --- Manual mapping fallback UI (borrowed from v12 style) ---
        if not is_already_correct_format and not st.session_state.data_mapped and not st.session_state.skip_mapping_ui:
            st.warning("We couldn’t recognize all required columns. Please map them manually.")
            df_columns = list(st.session_state.raw_df.columns)

            # Initialize mapping dict in session
            if "column_mapping" not in st.session_state or not isinstance(st.session_state.column_mapping, dict):
                st.session_state.column_mapping = {col: None for col in ALL_INTERNAL_COLUMNS}

            # Build options
            base_options = ["-- Select a column --"]
            options_optional = base_options + ["-- Do not map this column --"] + df_columns
            options_required = base_options + df_columns

            all_core_mapped_manual = True

            # Draw selectboxes for each internal column
            for internal_col in ALL_INTERNAL_COLUMNS:
                # prefer any prefilled mapping in session
                current_mapped_col = st.session_state.column_mapping.get(internal_col)
                if internal_col in CORE_REQUIRED_COLUMNS:
                    options = options_required
                else:
                    options = options_optional

                # compute default index
                if current_mapped_col in df_columns:
                    initial_index = options.index(current_mapped_col) if current_mapped_col in options else 0
                elif current_mapped_col is None and internal_col not in CORE_REQUIRED_COLUMNS and "-- Do not map this column --" in options:
                    initial_index = options.index("-- Do not map this column --")
                else:
                    initial_index = 0

                selected_column = st.selectbox(
                    f"Map '{internal_col}' to:",
                    options,
                    index=initial_index,
                    key=f"map_{internal_col}"
                )

                # Update mapping in session
                if selected_column == "-- Select a column --":
                    st.session_state.column_mapping[internal_col] = None
                    if internal_col in CORE_REQUIRED_COLUMNS:
                        all_core_mapped_manual = False
                elif selected_column == "-- Do not map this column --":
                    st.session_state.column_mapping[internal_col] = None
                else:
                    st.session_state.column_mapping[internal_col] = selected_column

            # Confirm and load
            if all_core_mapped_manual and st.button("Confirm Columns and Load Data"):
                with st.spinner("Applying mapping and processing data..."):
                    st.session_state.processed_df = load_and_map_data(uploaded_file, st.session_state.column_mapping)
                    if st.session_state.processed_df is not None:
                        st.session_state.data_loaded = True
                        st.session_state.data_mapped = True
                        st.session_state.current_filtered_df = st.session_state.processed_df.copy()
                        st.success("✅ Data loaded with your manual mapping. You can now ask questions.")
                        st.rerun()
                    else:
                        st.error("Failed to process the file with the provided mapping. Please verify your selections.")
            elif not all_core_mapped_manual:
                st.info("Please map all required fields before continuing: " + ", ".join(CORE_REQUIRED_COLUMNS))

    except Exception as e:
        st.error(f"❌ Error reading uploaded file to determine columns: {e}. Please ensure it's a valid Excel file.")
        st.session_state.data_loaded = False
        st.session_state.llm_mapping_attempted = False
        st.session_state.processed_df = None
        st.session_state.skip_mapping_ui = False
        st.session_state.current_filtered_df = None # Reset filtered data on error
else:
    st.info("Please upload an Excel file to get started.")
    st.session_state.data_loaded = False
    st.session_state.data_mapped = False
    st.session_state.llm_mapping_attempted = False
    st.session_state.processed_df = None
    st.session_state.skip_mapping_ui = False
    st.session_state.current_filtered_df = None # Ensure it's None if no file is uploaded


df = st.session_state.processed_df

if st.session_state.data_loaded and st.session_state.data_mapped and df is not None:
    
    if model is None:
        st.warning("NLP model not available. Chatbot functions requiring NLP are disabled.")
    else:
        if "conversation" not in st.session_state:
            st.session_state.conversation = []
            st.session_state.conversation.append({"role": "assistant", "content": "Hello! I'm your Kinesso Chatbot. I can help you analyze your marketing spend data. Ask me anything like 'What was the total spend by Sharjah Co-op?' or 'Compare online and offline spend'."})

        if "last_brand" not in st.session_state:
            st.session_state.last_brand = None

        for message in st.session_state.conversation:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        query = st.chat_input("Ask me something about your marketing data:")

        st.sidebar.markdown("---")
        if st.sidebar.button("Clear Chat History", help="Start a new conversation"):
            st.session_state.conversation = []
            st.session_state.last_brand = None
            st.session_state.data_loaded = False
            st.session_state.data_mapped = False
            st.session_state.raw_df = None
            st.session_state.column_mapping = {col: None for col in ALL_INTERNAL_COLUMNS}
            st.session_state.llm_mapping_attempted = False
            st.session_state.processed_df = None
            st.session_state.skip_mapping_ui = False
            st.session_state.sidebar_filters = {} # Clear sidebar filters too
            st.session_state.current_filtered_df = None # Clear filtered data
            logger.info("Chat history and data states cleared.")
            st.rerun()

        st.sidebar.markdown("### Filter Data by Columns:")
        # Dynamic Dropdowns for filtering
        # Use the full processed_df for filter options, not the currently filtered one
        # ADD 'Category' to the list of columns to be displayed in the sidebar
        available_filter_columns = [col for col in OPTIONAL_ANALYTICAL_COLUMNS if col in df.columns]
        
        # Initialize sidebar_filters in session state if not present
        if "sidebar_filters" not in st.session_state:
            st.session_state.sidebar_filters = {col: "All" for col in available_filter_columns}

        current_sidebar_selections = {}
        for col in available_filter_columns:
            # Ensure all values are strings before sorting to prevent TypeError
            unique_values = ["All"] + sorted([str(x) for x in df[col].dropna().unique().tolist()])
            selected_value = st.sidebar.selectbox(
                f"Select {col.replace('_', ' ').title()}:",
                unique_values,
                key=f"sidebar_filter_{col}",
                index=unique_values.index(st.session_state.sidebar_filters.get(col, "All"))
            )
            current_sidebar_selections[col] = selected_value
        
        # Update session state with current selections
        st.session_state.sidebar_filters = current_sidebar_selections

        if st.sidebar.button("Calculate Filtered Spend"):
            # Construct a query string or directly pass filters to handle_complex_query
            filters_to_apply = {k: v for k, v in st.session_state.sidebar_filters.items() if v != "All"}
            
            if not filters_to_apply:
                sidebar_query_text = "total spend (no filters selected)"
            else:
                filter_parts = [f"{k.replace('_', ' ').title()}: {v}" for k, v in filters_to_apply.items()]
                sidebar_query_text = f"total spend for {', '.join(filter_parts)}"

            st.session_state.conversation.append({"role": "user", "content": sidebar_query_text})
            with st.chat_message("user"):
                st.markdown(sidebar_query_text)

            with st.spinner("Calculating filtered spend..."):
                # Call handle_complex_query with the filters_dict
                response_text = handle_complex_query(filters_dict=filters_to_apply) 
            
            st.session_state.conversation.append({"role": "assistant", "content": response_text}) 
            with st.chat_message("assistant"):
                st.markdown(response_text) 
            st.rerun() 


        if query:
            st.session_state.conversation.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)

            with st.spinner("Processing your request..."):
                response_text = "" 
                
                if model is None:
                    logger.warning("SentenceTransformer model not loaded. Falling back to OpenAI for all queries.")
                    response_text = get_openai_response(query, st.session_state.conversation)
                    response_text += "\n\n*(Note: The internal data analysis features are unavailable because the NLP model could not be loaded.)*"
                else:
                    # Prepare unique entity lists for the LLM function call pre-check
                    # These should be based on the *full* dataset for the LLM to understand all possible entities
                    brands_in_data = df["Brand"].dropna().unique().tolist() if "Brand" in df.columns else []
                    sectors_in_data = df["Sector"].dropna().unique().tolist() if "Sector" in df.columns else []
                    sources_in_data = df["Source"].dropna().unique().tolist() if "Source" in df.columns else []
                    categories_in_data = df["Category"].dropna().unique().tolist() if "Category" in df.columns else []
                    products_in_data = df["Product"].dropna().unique().tolist() if "Product" in df.columns else []
                    media_in_data = df["Media"].dropna().unique().tolist() if "Media" in df.columns else []
                    agencies_in_data = df["Media Agency"].dropna().unique().tolist() if "Media Agency" in df.columns else []
                    producers_in_data = df["Producer"].dropna().unique().tolist() if "Producer" in df.columns else []
                    countries_in_data = df["Country"].dropna().unique().tolist() if "Country" in df.columns else []
                    channels_in_data = df["Channel"].dropna().unique().tolist() if "Channel" in df.columns else [] 

                    # Attempt LLM function calling first, with improved pre-check
                    tool_call_result = call_llm_for_function_call(
                        query, AVAILABLE_TOOLS, st.session_state.conversation, df.columns.tolist(),
                        brands_in_data, sectors_in_data, sources_in_data, categories_in_data,
                        products_in_data, media_in_data, agencies_in_data, producers_in_data,
                        countries_in_data, channels_in_data 
                    )
                    
                    if tool_call_result:
                        function_name_to_call = tool_call_result["function_name"]
                        function_args_to_pass = tool_call_result["args"]
                        logger.info(f"LLM decided to call: {function_name_to_call} with args: {function_args_to_pass}")
                        
                        # Reset current_filtered_df to the full processed_df for most direct, single-purpose queries
                        # handle_complex_query manages its own filtering context, so it's excluded.
                        if function_name_to_call not in ["handle_complex_query", "get_top_n_by_column"]: # get_top_n_by_column also uses full df
                            st.session_state.current_filtered_df = st.session_state.processed_df.copy()
                            logger.info(f"Resetting current_filtered_df for function: {function_name_to_call}")

                        function_mapping = {
                            "get_total_spend_by_column": get_total_spend_by_column, # Updated to generic
                            "get_top_n_by_column": get_top_n_by_column, # Updated to generic
                            "compare_online_offline_func": compare_online_offline_func, 
                            "get_daily_trend_data": get_daily_trend_data, # Renamed function here
                            "get_highest_spend_day": get_highest_spend_day,
                            "compare_months": compare_months, 
                            "forecast_next_7_days_func": forecast_next_7_days_func,
                            "compare_two_brands": compare_two_brands,
                            "list_all_brands": list_all_brands,
                            "list_all_sectors": list_all_sectors,
                            "calculate_column_statistic": calculate_column_statistic,
                            "plot_column_distribution": plot_column_distribution,
                            "list_unique_values": list_unique_values,
                            "handle_complex_query": handle_complex_query 
                        }

                        if function_name_to_call in function_mapping: # Ensure the suggested function exists
                            try:
                                # Special handling for functions needing the original query string
                                if function_name_to_call in ["compare_two_brands", "handle_complex_query", "handle_top_sectors", "handle_top_brands", "handle_compare_online_offline_query", "handle_trend", "handle_month_comparison"]:
                                    response_text = function_mapping[function_name_to_call](query=query)
                                # For functions that take no parameters
                                elif not function_args_to_pass: 
                                    response_text = function_mapping[function_name_to_call]()
                                # For all other functions that expect specific arguments
                                else: 
                                    response_text = function_mapping[function_name_to_call](**function_args_to_pass)
                                logger.info(f"Successfully executed LLM-suggested function: {function_name_to_call}")
                            except TypeError as te:
                                logger.error(f"TypeError when calling {function_name_to_call} with args {function_args_to_pass}: {te}", exc_info=True)
                                response_text = f"I'm sorry, I couldn't process that request due to incorrect arguments for the function. Please try rephrasing. (Error: {te})"
                                response_text += "\n\n" + get_openai_response(query, st.session_state.conversation) # Fallback
                            except Exception as e:
                                logger.error(f"Error executing LLM-suggested function {function_name_to_call}: {e}", exc_info=True)
                                response_text = f"An error occurred while trying to perform that analysis: {e}. I'll try to give a general answer."
                                response_text += "\n\n" + get_openai_response(query, st.session_state.conversation) # Fallback
                        else:
                            logger.warning(f"LLM suggested an unknown function: '{function_name_to_call}'. Falling back to general OpenAI.")
                            response_text = get_openai_response(query, st.session_state.conversation)
                    else:
                        # Fallback to SentenceTransformer if LLM didn't suggest a tool
                        query_emb = model.encode(query, convert_to_tensor=True)
                        scores = util.cos_sim(query_emb, base_embeddings)[0].cpu().numpy()
                        
                        max_score = np.max(scores)
                        best_intent_index = int(np.argmax(scores))
                        best_intent_st = config.BASE_INTENTS[best_intent_index]
                        
                        logger.info(f"User query: '{query}'")
                        logger.info(f"Falling back to S-T. Best intent: '{best_intent_st}', Score: {max_score:.2f}")
                        st.sidebar.write(f"DEBUG - S-T Fallback Intent: '{best_intent_st}', Score: {max_score:.2f}")

                        if max_score >= config.CONFIDENCE_THRESHOLD:
                            final_intent = best_intent_st
                            try:
                                if final_intent in ["general question about marketing", "what is marketing spend analysis", "tell me about AI in marketing."]:
                                    response_text = intent_funcs[final_intent](query, st.session_state.conversation) 
                                elif final_intent in ["highest spend day", "predict next week spend", "forecast spend next week", "list all brands", "list all sectors"]:
                                    response_text = intent_funcs[final_intent]()
                                else:
                                    response_text = intent_funcs[final_intent](query)
                                logger.info(f"Executed ST-matched intent '{final_intent}' successfully.")
                            except Exception as e:
                                logger.error(f"Error executing ST-matched intent '{final_intent}': {e}", exc_info=True)
                                response_text = get_openai_response(query, st.session_state.conversation)
                                response_text += f"\n\n*(An internal error occurred for this data query: {e}. I've tried to give a general answer.)*"
                        else:
                            logger.info("S-T confidence too low. Falling back to general OpenAI.")
                            response_text = get_openai_response(query, st.session_state.conversation)
                    

            st.session_state.conversation.append({"role": "assistant", "content": response_text})
            with st.chat_message("assistant"):
                st.markdown(response_text)

        st.markdown("""
            <hr>
            <h3>💡 Try asking:</h3>
            <ul>
                <li>"What was the total spend by Sharjah Co-op?"</li>
                <li>"Compare online and offline spend in 2024"</li>
                <li>"Show me the daily trend for Oasis Mall"</li>
                <li>"Compare spend of Dubai Duty Free vs Daiso Japan"</li>
                <li>"What are the top 3 sectors by spend??"</li>
                <li>"What are the top 5 brands by spend?"</li>
                <li>"Predict spend for next week"</li>
                <li>"Compare January 2023 and February 2024"</li>
                <li>"List all brands"</li>
                <li>"List all sectors"</li>
                <li>"What is the average of Amount?"</li>
                <li>"Show me a histogram of Amount"</li>
                <li>"List unique values in Source"</li>
                <li>"What is marketing spend analysis?" (AI fallback)</li>
                <li>"Tell me about AI in marketing." (AI fallback)</li>
            </ul>
        """, unsafe_allow_html=True)
else:
    pass
