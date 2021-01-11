# import sys
# sys.path.append("..")
import pandas as pd
import numpy as np
# import pickle
import tqdm
import pymorphy2
import logging
# import os
from string import punctuation
from nltk import TreebankWordTokenizer
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm
from sqlalchemy import create_engine
# from src.config import conn_string
from gensim.models import fasttext
from src.dbsrc import TableRecommendation
from src.config import fasttext_pth

logging.basicConfig(level = "INFO")
fast_text = fasttext.load_facebook_vectors(fasttext_pth)

# with open(fasttext_pth, 'rb') as f:
#     fast_text = pickle.load(f)



def get_users_texts(conn_string):
    """ Получает словарь из ключевых слов по пользователям """

    logging.info("Подгружаю данные из базы")
    engine = create_engine(conn_string)
    df = pd.read_sql_table('user', engine)
    # logging.info(df.head)
    users = df.user_id.tolist()
    texts = df.user_keywords.tolist()
    texts = txt_pipe(texts)
    user_txt_dict = dict(zip(users, texts))
    return user_txt_dict


def get_lines(conn_string):
    """ Подключается к БД и выкачивает вакансии """

    logging.info("Подгружаю данные из базы")
    engine = create_engine(conn_string)
    df = pd.read_sql_table('vacancy', engine)
    logging.info(df.head)
    lines = df.vacdescription.tolist()
    vacids = df.vacid.tolist()
    return lines, vacids


def txt_pipe(lines):
    """ Пайплайн для предобработки текста """

    logging.info("Готовлю корпус")
    morph = pymorphy2.MorphAnalyzer()
    ru_stop_words = stopwords.words('russian')
    lines_tok = [TreebankWordTokenizer().tokenize(x) for x in lines]
    lines_tok = [[x for x in el if x not in punctuation] for el in lines_tok]
    u_norm = [[morph.parse(x)[0][2] for x in el] for el in tqdm(lines_tok)]
    u_norm = [[x for x in el if x not in ru_stop_words] for el in tqdm(u_norm)]
    corpus = [' '.join(x) for x in u_norm]
    return corpus


def l2_norm(x):
    return np.sqrt(np.sum(x ** 2))


def div_norm(x):
    """ Нормализует вектор """

    norm_value = l2_norm(x)
    if norm_value > 0:
        return x * (1.0 / norm_value)
    else:
        return x


def get_vacancy_vectors(vacids, corpus):
    """ Формирует словарь векторов из предложений вакансий """

    vacancy_vectors = {}
    for x in tqdm(list(zip(vacids, corpus))):
        text = x[1].split()
        text.append('\n')
        matrix = np.zeros((300,), dtype = 'float32')
        for word in text:
            matrix += div_norm(fast_text.word_vec(word))
        vacancy_vectors[x[0]] = matrix
    return vacancy_vectors


def get_text_vector(user_text):
    """ Формирует вектор из ключевых слов пользователя """

    user_text = user_text.split()
    user_text.append('\n')
    matrix = np.zeros((300,), dtype = 'float32')

    for word in user_text:
        matrix += div_norm(fast_text.word_vec(word))

    return matrix


def get_user_text_vectors(user_txt_dict):
    """ Формирует векторы для всех пользователей и записывает в словарь """

    user_text_vectors = {}
    for user_id, keywords in user_txt_dict.items():
        logging.info(keywords)
        user_text_vectors[user_id] = get_text_vector(keywords)

    return user_text_vectors


def get_recs_for_all_users(vacids, vacancy_vectors, user_text_vectors):
    """ Находит сходства пользователей и вакансий, ранжирует вакансии и записывет их """

    logging.info("Нахожу сходства и формирую список рекомендаций")
    all_recommendations = []
    for user_id, user_text_matrix in tqdm(user_text_vectors.items()):
        user_sorted_vacids = find_similarity(vacids, vacancy_vectors, user_text_matrix)
        user_recommendation_objects = get_user_vacid_objects(user_sorted_vacids, user_id)
        all_recommendations.extend(user_recommendation_objects)

    return all_recommendations


def find_similarity(vacids, vacancy_vectors, user_text_matrix):
    """ Находит косинусное сходство из вектора пользователя и вакансий"""

    tu_sims = {}
    for vacid in tqdm(vacids):
        tu_sims[vacid] = cosine_similarity(vacancy_vectors[vacid].reshape(1, -1),
                                           user_text_matrix.reshape(1, -1))[0][0]

    tu_sorted = sorted(tu_sims.items(), key = lambda x: x[1], reverse = True)
    # tu_sorted = [x[0] for x in tu_sorted]
    return tu_sorted


def get_user_vacid_objects(sorted_vacancy_ids, user_id):
    """ Формирует список объектов рекомендаций по пользователю """

    list_of_recommendation_objects = []
    for vacid_score_tuple in sorted_vacancy_ids:
        rec = TableRecommendation(user_id = user_id, vacid = vacid_score_tuple[0], score = float(vacid_score_tuple[1]))
        list_of_recommendation_objects.append(rec)

    return list_of_recommendation_objects


def ft_pipeline(conn_string):
    """
    Поэтапно проходит по всем шагам рассчета рекомендаций
    1. получает список вакансий из базы
    2. препроцессит текст в корпус
    3. формирует из фасттекста векторы вакансий
    4. формирует векторы текстов пользователей
    5. рассчитывает сходство между векторами пользователей и ваканcий
    """

    lines, vacids = get_lines(conn_string)
    user_txt_dict = get_users_texts(conn_string)

    logging.info(user_txt_dict)
    corpus = txt_pipe(lines)

    user_text_vectors_ = get_user_text_vectors(user_txt_dict)
    vacancy_vectors_ = get_vacancy_vectors(vacids, corpus)

    all_recs_list = get_recs_for_all_users(vacids, vacancy_vectors_, user_text_vectors_)

    return all_recs_list
