import logging
from src.config import conn_string
from src.w2v import ft_pipeline
from load_recs import update_recs
from src.dbsrc import DataAccessLayer
from apscheduler.schedulers.blocking import BlockingScheduler
from dotenv import load_dotenv

logging.basicConfig(level = "INFO")


# todo:
# дописать функцию загрузки рекомендаций в БД +
# функцию выгрузки ключевых слов по юзерам
# функцию предобработки ключевых слов


def main_pipe(conn_string=conn_string):

    recs = ft_pipeline(conn_string)
    # print(recs)
    dal = DataAccessLayer(conn_string)
    session = dal.get_session()
    update_recs(session, recs)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    load_dotenv()
    main_pipe(conn_string)


    schedule = BlockingScheduler()
    schedule.add_job(main_pipe, 'interval', minutes = 10)
    schedule.start()