import logging
from src.config import conn_string
from src.w2v import ft_pipeline

logging.basicConfig(level = "INFO")


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    ft_pipeline(conn_string)
