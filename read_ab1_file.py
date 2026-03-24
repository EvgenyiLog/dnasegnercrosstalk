# from skbio.io import read, write
from typing import Tuple, List, Dict
from Bio.SeqRecord import SeqRecord

def read_ab1_file(file_path: str) -> Tuple[str, List[int], Dict[str, bytes], Dict[str, str]]:
    """
    Читает файл формата .ab1 (Applied Biosystems DNA chromatogram) и возвращает:
    
    Returns:
        sequence (str): Последовательность нуклеотидов (буквенная строка A, C, G, T).
        peak_indices (List[int]): Список индексов положения пиков, соответствующих буквам в последовательности.
        traces (Dict[str, bytes]): Сырые сигналы трассировки для нуклеотидов 'A', 'C', 'G', 'T' (по каналам DATA9–DATA12).
        meta (Dict[str, str]): Метаданные файла: ID, дата, модель прибора, версия basecaller и др.

    Args:
        file_path (str): Путь к .ab1 файлу.

    Raises:
        FileNotFoundError: Если файл не найден.
        ValueError: Если файл не является допустимым .ab1 файл
    """
    from Bio import SeqIO

    record: SeqRecord = SeqIO.read(file_path, "abi")
    abif = record.annotations["abif_raw"]

    sequence = str(record.seq)
    peak_indices = abif.get("PLOC2", [])

    traces = {
        "A": abif.get("DATA9"),
        "C": abif.get("DATA10"),
        "G": abif.get("DATA11"),
        "T": abif.get("DATA12"),
    }

    meta = {
        "id": record.id,
        "description": record.description,
        "sequence_length": len(sequence),
        "date": record.annotations.get("date", None),
        "instrument_model": abif.get("MCHN", b"").decode(errors="ignore"),
        "run_start": abif.get("RUND", b"").decode(errors="ignore"),
        "basecaller": abif.get("SPAC", b"").decode(errors="ignore"),
        "dye": abif.get("DYEP", b"").decode(errors="ignore"),
        "file_path": file_path
    }

    return sequence, peak_indices, traces, meta