import shutil
import time
import openai
import os

import pandas
from loguru import logger
import multiprocessing
import pandas as pd

from genAV import generateAV

gpt_api_key = [
    ""
]


root_path = r"GenerateVideo\assets"

def chat_answer(key: str, question: str):
    openai.api_base = "https://d1.xiamoai.top/v1"
    openai.proxy = "http://127.0.0.1:7890"
    openai.api_key = key

    chat_completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": question}
        ],
    )
    return chat_completion.choices[0].message.content


def trans_pn(pn):
    mark = list(map(chr, range(ord("a"), ord("z") + 1))) + list(map(chr, range(ord("A"), ord("Z") + 1))) + list(
        map(chr, range(ord("0"), ord("9") + 1))) + ["_", "-"]
    n_pn = ""

    for w in pn:
        if w in mark:
            n_pn += w
        else:
            n_pn += "_"

    return n_pn


def pn_gen_desc():
    """
    根据型号生成描述
    :return:
    """

    imgs_dir = os.path.join(root_path, "imgs")
    descs_dir = os.path.join(root_path, "descs")
    gened_descs = [desc.replace(".txt", "") for desc in os.listdir(descs_dir)]  # 生成过的产品

    pns = []
    with open(os.path.join(root_path, "pns.txt"), "r", encoding="utf-8") as f:
        for pn in f:
            pns.append(pn.strip())

    for pn in pns:

        n_pn = trans_pn(pn).lower()
        desc_path = os.path.join(descs_dir, f"{n_pn}.txt")
        if n_pn in gened_descs:
            logger.info(f"[]>> 生成过描述 {n_pn}")
            continue

        for _ in range(3):
            try:
                question = f"根据{pn}的参数和应用，请帮我使用英文生成一段不超过25个单词，不少于15个单词的简短介绍信息。"
                desc = chat_answer(gpt_api_key[0], question)
                logger.info(f"[生成描述]>> {pn} -> {desc}")

                with open(desc_path, "w+", encoding="utf-8") as f:
                    f.write(desc)
                logger.success(f"[生成描述]>> 成功 {pn} -> {desc_path}")
                gened_descs.append(n_pn)
                break
            except Exception as e:
                logger.error(f"[异常]>> {pn} 重试 - {e}")
                continue
            finally:
                time.sleep(4)

        # break


def work_gen_whole_video(q):
    while not q.empty():
        [i, web_site, pn] = q.get()

        try:
            logger.info(f"[开始]>> ({i}) {web_site} {pn}")

            generateAV(web_site, pn)

            logger.success(f"[成功]>> ({i}) {web_site} {pn}")
        except Exception as e:
            logger.error(f"[异常]>> ({i}) {web_site} {pn} - {e}")


# 生成全部完整视频
def task_gen_whole_video():
    q = multiprocessing.Queue()
    pool_size = 2

    # 获取所有型号
    videos_dir = os.path.join(root_path, "videos")
    grab_names = []
    for file_name in os.listdir(videos_dir):
        file_path = os.path.join(videos_dir, file_name)
        if not os.path.isdir(file_path):
            continue

        # video_path = os.path.join(file_path, f"final-{file_name}.mp4")  # *
        video_path = os.path.join(file_path, f"av-{'_'.join(file_name.split('_')[1:])}.mp4")  # *
        if os.path.isfile(video_path):
            grab_names.append(file_name)

    imgs_dir = os.path.join(root_path, "imgs")
    pn_names = os.listdir(imgs_dir)

    for i, pn_name in enumerate(pn_names):
        if pn_name in grab_names:
            logger.warning(f"[重复]>> 视频已制作 ({i}) {pn_name}")
            continue

        pn_name = pn_name.split("_")
        web_site = pn_name[0]
        pn = "_".join(pn_name[1:])
        q.put([i, web_site, pn])
        logger.info(f"({i}) >> {web_site} {pn}")

        # break

    p_list = []
    for _ in range(pool_size):
        p = multiprocessing.Process(target=work_gen_whole_video, args=(q,))
        p.start()
        p_list.append(p)

    for p in p_list:
        p.join()


# 提取视频
def extract_vedio():
    final_video_dir = os.path.join(root_path, "final_videos")
    videos_dir = os.path.join(root_path, "videos")

    for pn in os.listdir(videos_dir):
        video_dir = os.path.join(videos_dir, pn)
        if not os.path.isdir(video_dir):
            continue

        old_path = os.path.join(video_dir, f"av-{pn}.mp4")
        final_video_path = os.path.join(final_video_dir, f"av-{pn}.mp4")

        shutil.copy(old_path, final_video_path)
        print(f"{old_path} --> {final_video_path}")
        # break


if __name__ == '__main__':
    # pn_gen_desc()

    task_gen_whole_video()

    # extract_vedio()
    ...
