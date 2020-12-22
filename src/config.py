import os

from dotenv import load_dotenv

load_dotenv()
BOT_TOKEN = str(os.getenv("BOT_TOKEN"))
DATABASE_USER = str(os.getenv("DATABASE_USER"))
DATABASE_HOST = str(os.getenv("DATABASE_HOST"))
DATABASE_PORT = str(os.getenv("DATABASE_PORT"))
DATABASE_NAME = str(os.getenv("DATABASE_NAME"))
DATABASE_PASS = str(os.getenv("DATABASE_PASS"))
conn_string = f'postgresql://'
conn_string += f'{DATABASE_USER}:{DATABASE_PASS}'
conn_string += f'@{DATABASE_HOST}:{DATABASE_PORT}/{DATABASE_NAME}'


tf_path = os.path.join('data', 'tf.pkl')
tf_vectorizer_path = os.path.join('data', 'vectorizer.pkl')
fasttext_pth = os.path.join('wvmodel', 'cc.ru.300.bin')
# fasttext_pth = os.path.join('wvmodel', 'fast_text.pkl')
# fasttext_pth = os.path.join('wvmodel', 'ft_native_300_ru_wiki_lenta_lemmatize.bin')

admins = [
    os.getenv("ADMIN_ID"),
]


