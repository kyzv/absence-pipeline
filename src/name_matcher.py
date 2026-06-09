import pandas as pd
from typing import List


def match_names(absent_indices: List[int], student_db_path: str) -> List[str]:
    df = pd.read_csv(student_db_path)
    col = _find_name_column(df)
    names = df[col].tolist()
    return [names[i] for i in absent_indices if 0 <= i < len(names)]


def _find_name_column(df: pd.DataFrame) -> str:
    keywords = ['name', 'nom', 'prenom', 'prénom', 'student', 'étudiant']
    for col in df.columns:
        col_lower = col.lower().replace('&', '').replace('et ', '').strip()
        for kw in keywords:
            if kw in col_lower:
                return col
    return df.columns[0]
