"""
Development utilities, including:
data and model directory path and lazy creation
the configuration of logger
"""
import os
import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# standard data directory names
POPPY_PATH = "/Users/hongtengxu/PycharmProjects/PopPy"
DATA_DIR = "data"
PREPROCESSED_DIR = "preprocess"
MODEL_DIR = "model"
OUTPUT_DIR = "output"
EXAMPLE_DIR = "example"


def navigate_parent_dirs(path: str, levels: int) -> str:
    """
    Navigate to a parent directory relative to a given file path.

    Args:
        path (str): path to navigate from (file or directory; note that the parent of a file is directory,
                    and the parent of a directory is its parent directory)
        levels (int): number of levels to navigate (must be >= 0)

    Returns:
        str: absolute path of the parent directory that sits `levels` above given path

    Raises:
        ValueError: if `levels` is negative
    """
    if levels < 0:
        raise ValueError("levels must be >= 0, not {}".format(levels))

    result = os.path.abspath(os.path.join(path, os.path.sep.join(".." for _ in range(levels))))

    if os.path.isfile(result):
        return os.path.dirname(result)

    return result


def makedirs(path: str) -> None:
    """
    Create a directory (and its parents) if it does not exist.

    Args:
        path (str): directory path to create, including any missing parents

    Raises:
        ValueError: if the resolved path already exists as a file
    """
    if os.path.isfile(path):
        raise ValueError("path '{}' is an existing file; cannot create as a directory".format(path))

    os.makedirs(path, exist_ok=True)


def find_repo_root() -> str:
    """
    Find the root path of this repository.

    Returns:
        str: absolute path of the root of this repository
    """
    return navigate_parent_dirs(os.path.dirname(__file__), 2)


def find_data_root() -> str:
    """
    Find the root "data" directory within this repository.

    Returns:
        str: absolute path of the "data" directory within this repository
    """
    return os.path.join(find_repo_root(), DATA_ROOT_DIR)


def find_data_dir(relative_path: str, create=True) -> str:
    """
    Find a custom data directory within this repository, and optionally create it if it does not exist.

    For example, to resolve the absolute path of the "work" data directory (and create it if needed)::

        work_dir = find_data_dir(WORK_DATA_DIR)
        work_dir_2 = find_data_dir(WORK_DATA_DIR + "2")
        # can proceed assuming these directories exist

    Args:
        relative_path (str): relative directory path to find, within this repository's "data" directory
        create (bool): flag to create the directory if it does not exist (default = True)

    Returns:
        str: absolute path of the "data/`relative_path`" directory

    Raises:
        ValueError: if `create` is True and the resolved path already exists as a file
    """
    custom_dir_absolute = os.path.join(find_data_root(), relative_path)

    if create:
        makedirs(custom_dir_absolute)

    return custom_dir_absolute
