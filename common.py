from dataclasses import dataclass
from typing import List


@dataclass
class TmcSample:
    sample_id: int
    labels: List[int]
    present_attributes: List[int]
