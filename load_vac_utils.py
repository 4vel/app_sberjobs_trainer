import logging
from src.dbsrc import Vacancy

from sqlalchemy.exc import DBAPIError



def clean_table(dal):
    try:
        dal.connect()
        num_rows_deleted = dal.session.query(Vacancy).delete()
        logging.info(f'Удаляю данные Vacancy. Удалено {num_rows_deleted}')
        dal.session.commit()
    except DBAPIError as exc:
        logging.info(f'{exc}')
        dal.session.rollback()






