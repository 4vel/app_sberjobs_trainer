import numpy as np
import pandas as pd
import pickle
import tqdm
import pymorphy2
import logging
from string import punctuation
from nltk import TreebankWordTokenizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm
from sqlalchemy import create_engine
# from sqlalchemy.ext.declarative import declarative_base
from src.config import tf_path, tf_vectorizer_path

morph = pymorphy2.MorphAnalyzer()

logging.basicConfig(level="INFO")

def txt_pipe(lines):
    logging.info("Готовлю корпус")
    ru_stop_words = stopwords.words('russian')
    lines_tok = [TreebankWordTokenizer().tokenize(x) for x in lines]
    lines_tok = [[x for x in el if x not in punctuation] for el in lines_tok]
    u_norm = [[morph.parse(x)[0][2] for x in el] for el in tqdm(lines_tok)]
    u_norm = [[x for x in el if x not in ru_stop_words] for el in tqdm(u_norm)]
    corpus = [' '.join(x) for x in u_norm]
    return corpus


def get_tf_idf(corpus):
    logging.info("Готовлю tf_idf")
    tfvectorizer = TfidfVectorizer(ngram_range=(2,3))
    matrix = tfvectorizer.fit_transform(corpus)
    tfmat = matrix.toarray()
    return tfmat, tfvectorizer


def get_new_tf_idf(new_corpus, tfvectorizer):

    logging.info("Рассчитываю новый tf_idf")

    new_corpus = [','.join(new_corpus)]
    tf = tfvectorizer.transform(new_corpus)
    tf = tf.toarray()
    return tf

def get_vstacked_mats(tf, tfmat):
    return np.vstack((tf, tfmat))


def get_sims(new_tfmat, vacids):

    logging.info("Рассчитываю сходства")
    cossim = cosine_similarity(new_tfmat)
    cdf = pd.DataFrame(cossim)
    cdf['vacids'] = ['target'] + vacids
    cdf = cdf.set_index('vacids')
    cdf.columns = ['target'] + vacids
    scdf = cdf.loc['target'].copy()
    sorted_vacids = scdf.sort_values(ascending = False).index.tolist()[1:]
    if len(sorted_vacids) > 1:
        return sorted_vacids[1:]
    else:
        return None

new_corpus = ['python', 'git', 'machine learning',
              'senior', 'анализировать', 'данные',
              'руководить','москва','кутузов']

def get_lines(conn_string):
    """
    Подключается к БД, выгружает датафрейм и из него создает списки описаний вакансий и их id
    """
    logging.info("Подгружаю данные из базы")
    engine = create_engine(conn_string)

    df = pd.read_sql_table('vacancy', engine)
    logging.info(df.head)
    lines = df.vacdescription.tolist()
    vacids = df.vacid.tolist()
    return lines, vacids

def tf_idf_piplene(conn_string, to_pkl=True):
    """
    Создает матрицу tf-idf и векторайзер для того, чтобы в дальнейшем получать вектор пользователя или вакансии
     и сравнивать его с tf-idf матрицей
    """

    lines, vacids = get_lines(conn_string)

    corpus = txt_pipe(lines)
    tf, tfvectorizer = get_tf_idf(corpus)

    if to_pkl:
        with open(tf_path, "wb") as f:
            pickle.dump(tf, f)

        with open(tf_vectorizer_path, "wb") as f:
            pickle.dump(tfvectorizer, f)

    return tf, tfvectorizer

