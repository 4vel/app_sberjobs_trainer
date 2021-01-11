import datetime
import logging

from sqlalchemy import Column, Integer, VARCHAR, TIMESTAMP, BOOLEAN, NUMERIC
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()
#

class BaseModel(Base):
    __abstract__ = True

    id = Column(Integer, nullable = False, unique = True, primary_key = True, autoincrement = True)
    created_at = Column(TIMESTAMP, nullable = False, default = datetime.datetime.now())
    updated_at = Column(TIMESTAMP, nullable = False, default = datetime.datetime.now(),
                        onupdate = datetime.datetime.now())

    def __repr__(self):
        return "<{0.__class__.__name__}(id={0.id!r})>".format(self)


class Vacancy(BaseModel):
    """ Модель данных для таблицы вакансий """

    __tablename__ = 'vacancy'

    vacid = Column(VARCHAR())
    vactitle = Column(VARCHAR(255))
    vacdescription = Column(VARCHAR())
    vacdate = Column(VARCHAR(255))
    vacstatus = Column(VARCHAR(100))
    vaclink = Column(VARCHAR(100))

    def __init__(self, vacid, vactitle, vacdescription, vacdate, vacstatus, vaclink):
        self.vacid = vacid
        self.vactitle = vactitle
        self.vacdescription = vacdescription
        self.vacdate = vacdate
        self.vacstatus = vacstatus
        self.vaclink = vaclink

    def __repr__(self):
        return f'{self.vactitle} {self.vacdescription}'


class TableUserVacancyPreference(BaseModel):
    """ Модель данных для таблицы предпочтений пользователей """

    __tablename__ = 'user_vacancy_preference'

    user_id = Column(VARCHAR(), primary_key = True)
    vacid = Column(VARCHAR(), primary_key = True)
    liked = Column(BOOLEAN())
    disliked = Column(BOOLEAN())

    def __init__(self, user_id, vacid, liked, disliked):
        self.user_id = user_id
        self.vacid = vacid
        self.liked = liked
        self.disliked = disliked

    def __repr__(self):
        return f'{self.user_id}'


class TableUser(BaseModel):
    """ Модель данных для таблицы c данными пользователей """

    __tablename__ = 'user'

    user_id = Column(VARCHAR(), primary_key = True)
    user_name = Column(VARCHAR())
    user_email = Column(VARCHAR(255))
    user_keywords = Column(VARCHAR())

    def __init__(self, user_id, user_name, user_email, user_keywords):
        self.user_id = user_id
        self.user_name = user_name
        self.user_email = user_email
        self.user_keywords = user_keywords

    def __repr__(self):
        return f'{self.user_id} {self.user_email} {self.user_keywords}'


class TableRecommendation(BaseModel):
    """ Модель данных для таблицы рекомендованных вакансий """

    __tablename__ = 'recommendation'

    user_id = Column(VARCHAR(), primary_key = True)
    vacid = Column(VARCHAR())
    score = Column(NUMERIC())

    def __init__(self, user_id, vacid, score):
        self.user_id = user_id
        self.vacid = vacid
        self.score = score

    def __repr__(self):
        return f'{self.user_id}'


class DataAccessLayer:
    """Подключение к базе данных. Сессия"""

    def __init__(self, connection_string):
        self.engine = None
        self.session = None
        self.Session = None
        self.conn_string = connection_string

    def connect(self):
        logging.info(f"Подключаюсь к БД")

        self.engine = create_engine(self.conn_string)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind = self.engine)
        self.session = self.Session()

    def get_session(self):
        logging.info(f"Подключаюсь к БД")

        self.engine = create_engine(self.conn_string)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind = self.engine)
        self.session = self.Session()
        return self.session


class VacancyMessage:

    def __init__(self, vobj):
        self.vacdescription = ""
        self.title = ""
        self.vacdescription = vobj.vacdescription
        self.title = vobj.vactitle
        self.vac_id = vobj.vacid
        self.vaclink = vobj.vaclink

    def check_data(self):
        if self.vacdescription == "":
            self.vacdescription = "... (не добавлено описание вакансии)"
        elif self.title == "":
            self.title = "..."

        if not self.vacdescription:
            self.vacdescription = "...."
        elif not self.title:
            self.title = "...."

    def check_vacdescription(self):

        if len(self.vacdescription) > 1000:
            desc_len = len(self.vacdescription)
            desc_parts = desc_len // 1000
            new_desc = []
            start = 0
            pos = start + 1000
            for i in range(1, desc_parts):
                new_desc.append(self.vacdescription[start:pos])
                start = pos
                pos = start + 1000

            new_desc.append(self.vacdescription[pos:desc_len])
            self.vacdescription = new_desc

    def make_message(self):
        self.check_data()
        self.check_vacdescription()
        return (self.title,
                self.vacdescription,
                self.vac_id,
                self.vaclink)


class VacNavigator:

    def __init__(self, vactuple):
        self.current_id = vactuple[1]
        self.previous_id = vactuple[0]
        self.next_id = vactuple[2]
