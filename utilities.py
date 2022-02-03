import numpy as np
import os

def list_files(data_directory: str, form: str = None):
    """
    Zwraca nazwy plików znajdujących się w danym folderze data_directory jako listę stringów
    :param data_directory: scieżka do folderu z danymi
    :param form: format szukanych plików, jeśli None zwraca nazwy wszystkich plików w folderze
    """
    if form is None:
        files = os.listdir(data_directory)
    else:
        files = []
        all_files = os.listdir(data_directory)
        for i in all_files:
            if '.{}.'.format(form) in i:
                pass
            else:
                if '.{}'.format(form) in i:
                    files.append(i)

    return files

def normalize(data, a=0, b=1):
    """Normalizuje zestaw danych do zakresu od a do b. Przyjmuje tylko numpy array"""
    normalized = []
    for i in data:
        normalized.append(a + (i-i.min()) * (b-a) / (i.max()-i.min()))

    return np.array(normalized)