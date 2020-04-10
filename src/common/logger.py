import datetime
import logging
import os


def get_log(log_dir, annotation):
    if not os.path.exists(log_dir):
        os.system("mkdir -p " + log_dir)
    logger = logging.getLogger(annotation)
    logger.setLevel(logging.INFO)
    dd = datetime.datetime.now()
    fh = logging.FileHandler(log_dir + "out_project_%s.log.%s" % (annotation, dd.isoformat()))
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # log format
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    # log part end
    return logger


def get_log_one_path(path):
    logger = logging.getLogger(path)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(path)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # log format
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    # log part end
    return logger
