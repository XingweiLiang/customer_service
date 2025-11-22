from langdetect import detect, LangDetectException
import dspy
from dspy_judge.logging_config import get_logger

logger = get_logger(__name__)


def detect_language(text):
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"


# functions to use with datasets.map()
def extract_llm_response_fields(example):
    # Assumes llm_response is already a dict, not a stringified JSON
    resp = example["llm_response"]
    return {
        "explanation": resp.get("explanation", None),
        "satisfied": resp.get("satisfied", None),
    }


def extract_llm_response_fields_dspy(example):
    # Assumes llm_response is already a dict, not a stringified JSON
    resp = example["dspy_response"]
    return {
        "explanation": resp.get("reasoning", None),
        "satisfied": resp.get("satisfied", None).lower(),
    }


def concat_company_and_conversation(example):
    return {
        "company_and_transcript": f"Company: {example['company']}\nTranscript so far: {example['truncated_conversation']}"
    }


def concat_latest_response(example):
    return {
        "output_transcript": f"{example['company_and_transcript']}\nSupport: {example['llm_response']}"
    }


def concat_latest_response_dspy(example):
    return {
        "output_transcript": f"{example['company_and_transcript']}\nSupport: {example['dspy_response']['llm_response']}"
    }


def build_company_and_conversation_cols(example):
    try:
        # Handle None or empty responses
        if not example.get("llm_response"):
            return {"company": "Unknown", "conversation": ""}
        
        response = example["llm_response"]
        lines = response.split("\n")
        
        # Extract company name from first line
        company_name = "Unknown"
        conversation_string = ""
        
        # Look for "Company:" line
        for i, line in enumerate(lines):
            if line.startswith("Company:"):
                company_name = line.split(":", 1)[1].strip()
                break
        
        # Find "Conversation:" line and extract everything after it
        for i, line in enumerate(lines):
            if line.strip() == "Conversation:":
                # Join all lines after "Conversation:" 
                conversation_lines = lines[i+1:]
                conversation_string = "\n".join(conversation_lines).strip()
                break
        
        return {"company": company_name, "conversation": conversation_string}
        
    except Exception as e:
        # Fallback for any parsing errors
        print(f"Error parsing response: {e}")
        return {"company": "Unknown", "conversation": str(example.get("llm_response", ""))}


def convert_dataset_to_dspy_examples(dataset, field_mapping, input_field):
    """
    Convert dataset to a list of dspy.Example objects.
    
    Parameters:
        dataset: The dataset object (e.g., Hugging Face Dataset)
        field_mapping: dict mapping from dspy attribute name to dataset column name
            e.g., {'transcript': 'transcript', 'satisfied': 'satisfied'}
        input_field: str
    Returns:
        List of dspy.Example objects
    """
    
    examples = []
    
    # Manual conversion to avoid PyArrow compatibility issues
    try:
        # Try to iterate directly through the dataset
        for idx in range(len(dataset)):
            row = dataset[idx]
            
            example_fields = {}
            for key, column_name in field_mapping.items():
                value = row[column_name]
                if isinstance(value, str):
                    example_fields[key] = value.strip()
                else:
                    example_fields[key] = value
            
            # Add an ID to help with hashing or caching
            example_fields["_id"] = f"example_{idx}"
            
            # Create the example with dynamic arguments
            example = dspy.Example(**example_fields).with_inputs(input_field)
            examples.append(example)
    
    except Exception as e:
        logger.error(f"Error converting dataset to DSPy examples: {e}")
        # Fallback: try to use to_pandas if available
        try:
            dataset_pd = dataset.to_pandas()
            for idx, row in dataset_pd.iterrows():
                example_fields = {
                    key: str(row[value]).strip() if isinstance(row[value], str) else row[value]
                    for key, value in field_mapping.items()
                }
                example_fields["_id"] = f"example_{idx}"
                example = dspy.Example(**example_fields).with_inputs(input_field)
                examples.append(example)
        except Exception as pandas_error:
            logger.error(f"Pandas fallback also failed: {pandas_error}")
            raise e
    
    logger.info(f"Processed {len(examples)} training examples")
    return examples
