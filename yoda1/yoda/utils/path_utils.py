import os
import tempfile
from typing import Optional


def is_dir(path: str) -> bool:
    """is a directory or not"""
    return os.path.isdir(path)


def is_file(path: str) -> bool:
    """is a file or not"""
    return os.path.isfile(path)


def create_tmp_dir(root: Optional[str] = None) -> str:
    """create temp directory under given dir"""
    dirname = tempfile.mkdtemp(dir=root)
    if not is_dir(dirname):
        raise OSError(f"Create tmp dir: {dirname} fails")
    return dirname


def create_if_not_exists(dirname: str):
    """create directory if not exists"""
    if not is_dir(dirname):
        os.makedirs(dirname)
        if not is_dir(dirname):
            raise OSError(f"Cannot create dir: {dirname}")


def get_file_dirname(path: str) -> str:
    """get directory name of given file path"""
    return os.path.dirname(os.path.abspath(path))


def get_parent_dirname(path: str) -> str:
    """get parent name of given dir path"""
    return os.path.dirname(os.path.abspath(path))


def create_parent_dir(path: str):
    """create parent directory given file path"""
    dirname = get_file_dirname(path)
    create_if_not_exists(dirname)
