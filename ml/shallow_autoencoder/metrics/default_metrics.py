from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Self, TypeVar

import pandas as pd

from ml.shallow_autoencoder.metrics.abstract_metrics import AbstractMetrics

TMetrics = TypeVar("TMetrics", bound="AbstractMetrics")

@dataclass
class DefaultMetrics(AbstractMetrics):
    mailshot_id: int
    opened: bool
    prediction: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mailshot_id": int(self.mailshot_id) if hasattr(self.mailshot_id, "item") else self.mailshot_id,
            "opened": bool(self.opened),
            "prediction": float(self.prediction) if hasattr(self.prediction, "item") else self.prediction
        }

    @classmethod
    def from_csv_gz(cls, folder: Path, filename: str = "results.csv.gz") -> List[Self]:
        """
        Read a gzipped CSV file and return a list of metrics.
        """
        # Read the gzipped CSV file into a DataFrame
        df = pd.read_csv(folder / filename, compression='gzip')

        # Convert the DataFrame to a list of metrics
        metrics_list = cls.from_dataframe(df)

        return metrics_list

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> List[Self]:
        """
        Convert a DataFrame to a list of metrics.
        """
        metrics_list = []
        for _, row in df.iterrows():
            metrics = cls(
                mailshot_id=row["mailshot_id"],
                opened=row["opened"],
                prediction=row["prediction"]
            )
            metrics_list.append(metrics)
        return metrics_list
