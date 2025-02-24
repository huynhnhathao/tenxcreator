import msgpack
import os

from typing_extensions import Dict


def save_dict_to_msgpack(dictionary: Dict, file_path: str) -> bool:
    """
    Save a dictionary to a file using MessagePack format.

    Args:
        dictionary (dict): The dictionary to be saved
        file_path (str): The full path where the file should be saved

    Returns:
        bool: True if successful, False if an error occurred
    """
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Open file in binary write mode and save the dictionary using msgpack
        with open(file_path, "wb") as f:
            msgpack.pack(dictionary, f)

        return True

    except Exception as e:
        print(f"Error saving dictionary to msgpack: {str(e)}")
        return False


def load_dict_from_msgpack(file_path: str) -> Dict:
    """
    Load a dictionary from a MessagePack file.

    Args:
        file_path (str): The full path to the msgpack file

    Returns:
        dict: The dictionary loaded from the file
        Raise exception if error
    """
    # Open file in binary read mode and load the dictionary using msgpack
    with open(file_path, "rb") as f:
        dictionary = msgpack.unpack(f, raw=False)

    return dictionary
