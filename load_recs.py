import logging
from src.dbsrc import TableRecommendation
from sqlalchemy.exc import DBAPIError


def clean_table(session):
    try:

        num_rows_deleted = session.query(TableRecommendation).delete()
        logging.info(f'Удаляю данные TableRecommendation. Удалено {num_rows_deleted}')
        session.commit()
    except DBAPIError as exc:
        logging.info(f'{exc}')
        session.rollback()


def add_recs(session, rec_objecsts):
    try:

        logging.info(f'Обновляю рекомендации')
        session.bulk_save_objects(rec_objecsts)
        session.commit()

    except DBAPIError as exc:
        logging.info(f'{exc}')
        session.rollback()


def update_recs(session, rec_objecsts):
    clean_table(session)
    add_recs(session, rec_objecsts)