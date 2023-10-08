import logging
import pandas as pd 
from zenml import step
from src.data_cleaning import DataCleaning, DivideDataStragety, datapreprocessing

from typing_extensions import Annotated
from typing import Tuple
from zenml import step


@step
def clean_data(
    data: pd.DataFrame,
) -> Tuple[
    Annotated[pd.DataFrame, "x_train"],
    Annotated[pd.DataFrame, "x_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
]:

    try:
        preprocess_strategy = datapreprocessing()
        data_cleaning = DataCleaning(data, preprocess_strategy)
        preprocessed_data = data_cleaning.handel_data()

        divide_strategy = DivideDataStragety()
        data_cleaning = DataCleaning(preprocessed_data, divide_strategy)
        x_train, x_test, y_train, y_test = data_cleaning.handel_data()
        return x_train, x_test, y_train, y_test
    except Exception as e:
        logging.error(e)
        raise e