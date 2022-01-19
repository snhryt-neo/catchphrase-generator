import dataclasses
import logging
from typing import Dict, List

import pandas as pd
from google.oauth2.service_account import Credentials


@dataclasses.dataclass
class GBQ:
    """Class for reading data in BigQuery and writing data to BigQuery."""

    project_id: str
    creds: dataclasses.InitVar[Dict[str, str]]
    logger: logging.Logger = None
    cred: Credentials = dataclasses.field(init=False)

    def __post_init__(self, creds) -> None:
        self.cred = Credentials.from_service_account_info(creds)

    def replace(
        self,
        df: pd.DataFrame,
        dataset: str,
        table: str,
        progress_bar: bool = True,
        table_schema: List[Dict[str, str]] = None,
        allows_overwrite: bool = True,
    ) -> None:
        """Create a new table in the BigQuery dataset.

        Parameters
        ----------
        df : pd.DataFrame

        dataset : str

        table : str

        progress_bar : bool, optional
            , by default False

        table_schema : List[Dict[str, str]], optional
            , by default None. E.g.,
        ```json
        [
            {
                "description": "description of column 1",
                "name": "column1",
                "type": "STRING",
                "mode": "REQUIRED",
            },
            {
                "description": "description of column2",
                "name": "column2",
                "type": "INTEGER",
                "mode": "NULLBALE",
            }
        ]
        ```
            See [official document](https://cloud.google.com/bigquery/docs/schemas)
            for more detail.

        allows_overwrite : bool, optional
            , by default True.
            When this is False and the table has already existed, loading fails.
        """
        load_mode = "replace" if allows_overwrite else "fail"
        df.to_gbq(
            f"{dataset}.{table}",
            project_id=self.project_id,
            if_exists=load_mode,
            table_schema=table_schema,
            progress_bar=progress_bar,
            credentials=self.cred,
        )

    def append(
        self,
        df: pd.DataFrame,
        dataset: str,
        table: str,
        progress_bar: bool = False,
        table_schema: List[Dict[str, str]] = None,
    ) -> None:
        """Create a new table in the BigQuery dataset.
        If `table` already exists, append `df` to it.
        Parameters
        ----------
        df : pd.DataFrame
        dataset : str
        table : str
        progress_bar : bool, optional
            , by default False
        table_schema : List[Dict[str, str]], optional
            , by default None. E.g.,
        ```json
        [
            {
                "description": "description of column 1",
                "name": "column1",
                "type": "STRING",
                "mode": "REQUIRED",
            },
            {
                "description": "description of column2",
                "name": "column2",
                "type": "INTEGER",
                "mode": "NULLBALE",
            }
        ]
        ```
            See [official document](https://cloud.google.com/bigquery/docs/schemas)
            for more detail.
        """
        df.to_gbq(
            f"{dataset}.{table}",
            project_id=self.project_id,
            if_exists="append",
            table_schema=table_schema,
            progress_bar=progress_bar,
            credentials=self.cred,
        )
