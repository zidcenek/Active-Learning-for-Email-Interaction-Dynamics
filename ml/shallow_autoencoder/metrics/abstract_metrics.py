import json
from abc import abstractmethod, ABC
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Self, TypeVar

import pandas as pd


TMetrics = TypeVar("TMetrics", bound="AbstractMetrics")

@dataclass
class AbstractMetrics(ABC):
    mailshot_id: int
    opened: bool
    prediction: float

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        pass

    @classmethod
    @abstractmethod
    def from_csv_gz(cls, folder: Path, filename: str = "results.csv.gz") -> List[Self]:
        """
        Read a gzipped CSV file and return a list of metrics.
        """
        pass

    @classmethod
    @abstractmethod
    def from_dataframe(cls, df: pd.DataFrame) -> List[Self]:
        """
        Convert a DataFrame to a list of metrics.
        """
        pass

    @classmethod
    def list_to_dict(cls, metrics: List[Self]) -> List[Dict[str, Any]]:
        return [m.to_dict() for m in metrics]

    @staticmethod
    def to_json_file(metrics: List[TMetrics], folder: Path, filename: str = "results.json") -> None:
        with open(folder / filename, 'w') as f:
            json.dump([m.to_dict() for m in metrics], f, indent=4)

    @staticmethod
    def to_csv_gz(metrics: List[TMetrics], folder: Path,
                  filename: str = "results.csv.gz") -> None:
        # Convert metrics to a list of dictionaries
        metrics_dicts = [m.to_dict() for m in metrics]

        # Create a DataFrame from the list of dictionaries
        df = pd.DataFrame(metrics_dicts)

        # Save to compressed CSV
        df.to_csv(folder / filename, compression='gzip', index=False)