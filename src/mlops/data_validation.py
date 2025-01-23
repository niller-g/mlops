import logging
from typing import Dict, Any
import great_expectations as ge
from datasets import Dataset
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataValidator:
    def __init__(self):
        self.context = ge.get_context()
    
    def validate_dataset(self, dataset: Dataset, is_test_mode: bool = False) -> Dict[str, Any]:
        """
        Validate a HuggingFace dataset using Great Expectations.
        
        Args:
            dataset: HuggingFace Dataset to validate
            is_test_mode: If True, applies relaxed validation rules for testing
            
        Returns:
            Dictionary containing validation results
        """
        # Convert to pandas for Great Expectations
        df = pd.DataFrame(dataset)
        ge_df = ge.from_pandas(df)
        
        # Core expectations that always apply
        expectations = [
            # Validate text field exists and is not null
            ge_df.expect_column_to_exist("clean_text"),
            ge_df.expect_column_values_to_not_be_null("clean_text"),
            
            # Validate text content
            ge_df.expect_column_values_to_be_of_type("clean_text", "str"),
            ge_df.expect_column_value_lengths_to_be_between(
                "clean_text", 
                min_value=1,  # At least 1 character
                max_value=2048  # Reasonable max length for transformer models
            ),
        ]
        
        # Add dataset size expectations based on mode
        if is_test_mode:
            expectations.append(
                ge_df.expect_table_row_count_to_be_between(
                    min_value=1,  # Allow small datasets in test mode
                    max_value=1000  # Reasonable test set size
                )
            )
        else:
            expectations.append(
                ge_df.expect_table_row_count_to_be_between(
                    min_value=100,  # Minimum rows needed for meaningful training
                    max_value=1000000  # Reasonable upper limit
                )
            )
        
        # Compile results
        validation_results = {
            "success": all(exp.success for exp in expectations),
            "results": [
                {
                    "expectation": exp.expectation_config["expectation_type"],
                    "success": exp.success,
                    "result": {
                        "observed_value": exp.result.get("observed_value"),
                        "details": exp.result
                    }
                }
                for exp in expectations
            ]
        }
        
        # Log results with more context
        logger.info(f"Data validation results ({'TEST MODE' if is_test_mode else 'PRODUCTION MODE'}):")
        for result in validation_results["results"]:
            status = "✓" if result["success"] else "✗"
            observed = result["result"]["observed_value"]
            logger.info(f"{status} {result['expectation']} - Value: {observed}")
        
        return validation_results 