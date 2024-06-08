import logging
import os


def create_directory_and_file(directory_path, file_name):
    if not os.path.exists(directory_path):
        # 递归创建目录
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created successfully.")
    else:
        print(f"Directory '{directory_path}' already exists.")

    file_path = os.path.join(directory_path, file_name)

    if not os.path.exists(file_path):
        with open(file_path, 'w') as file:
            print(f"File '{file_path}' created successfully.")
    else:
        print(f"File '{file_path}' already exists.")


def get_logger(logger_dir, logger_file, verbosity=1, name=None):
    create_directory_and_file(logger_dir, logger_file)
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(logger_dir+logger_file, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger
