import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

#  API keys 
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
FMP_API_KEY = os.getenv("FMP_API_KEY", "")
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")

#  SEC EDGAR
SEC_EDGAR_USER_AGENT = os.getenv("SEC_EDGAR_USER_AGENT", "YourName your@email.com")
SEC_BASE_URL = "https://data.sec.gov"
SEC_EFTS_URL = "https://efts.sec.gov/LATEST"
SEC_ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data"
SEC_RATE_LIMIT_DELAY = 0.12  # 10 requests/sec max → 100ms + margin

#  FMP
FMP_BASE_URL = "https://financialmodelingprep.com/stable"

#  NewsAPI
NEWS_API_BASE_URL = "https://newsapi.org/v2"

#  Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
CHROMA_DIR = DATA_DIR / "chroma"

#  Ingestion defaults 
FILING_YEARS_BACK = 2          
MAX_EARNINGS_TRANSCRIPTS = 4   
NEWS_DAYS_BACK = 30            

#  Models 
LLM_MODEL = "gpt-4o"                    
LLM_MODEL_MINI = "gpt-4o-mini"          
EVAL_MODEL = "gpt-4o"                   
EMBEDDING_MODEL = "text-embedding-3-small"
