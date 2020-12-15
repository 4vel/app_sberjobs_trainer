import logging
from src.config import conn_string
from src.w2v import ft_pipeline
from load_recs import update_recs
from src.dbsrc import DataAccessLayer

logging.basicConfig(level = "INFO")


# todo:
# дописать функцию загрузки рекомендаций в БД +
# функцию выгрузки ключевых слов по юзерам
# функцию предобработки ключевых слов


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    recs = ft_pipeline(conn_string)
    dal = DataAccessLayer(conn_string)
    session = dal.get_session()
    update_recs(session, recs)
