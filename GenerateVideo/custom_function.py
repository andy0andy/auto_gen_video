import requests
import openai
from PIL import Image
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from retrying import retry

from settings import *

@retry(stop_max_attempt_number=3, wait_fixed=1000 * 2)
def chat_answer_for_3(gpt_key: str, question: str) -> str:
    """
    gpt3 问答
    """

    openai.api_base = "https://d1.xiamoai.top/v1"
    openai.proxy = "http://127.0.0.1:7890"
    openai.api_key = gpt_key

    chat_completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": question}
        ],
    )
    return chat_completion.choices[0].message.content


def chat_answer_for_4(gpt_key: str, question: str) -> str:
    """
    gpt3 问答
    """

    openai.api_base = "https://api.kwwai.top/v1"
    openai.proxy = "http://127.0.0.1:7890"
    openai.api_key = gpt_key

    chat_completion = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": question}
        ],
    )
    return chat_completion.choices[0].message.content


def trans_pn(pn: str) -> str:
    """
    转换产品名格式
    :param pn:
    :return:
    """

    mark = list(map(chr, range(ord("a"), ord("z") + 1))) + list(map(chr, range(ord("A"), ord("Z") + 1))) + list(
        map(chr, range(ord("0"), ord("9") + 1))) + ["_", "-"]
    n_pn = ""

    for w in pn:
        if w in mark:
            n_pn += w
        else:
            n_pn += "_"

    return n_pn.lower()

def check_image(filepath):
    """
    检查图片是否异常
    :param filepath:
    :return: 正常 True，反之
    """
    try:
        with Image.open(filepath) as img:
            img.verify()
            return True
    except:
        return False


def download_image(link, file_path):
    headers = {
        # "authority": f"www.{row[3].lower()}.com",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
    }
    resp = requests.get(link, headers=headers, verify=False, timeout=10)
    with open(file_path, "wb") as f:
        f.write(resp.content)






