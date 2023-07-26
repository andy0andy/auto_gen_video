import shutil
import time
import openai
import os
from loguru import logger
import multiprocessing

from genAV import generateAV

gpt_api_key = [
    "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",
    "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",
    "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",
    "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",
]

def chat_answer(key: str, question: str):


    # openai.proxy = "http://127.0.0.1:7890"
    openai.api_key = key
    chat_completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": question}
        ],
    )
    return chat_completion.choices[0].message.content


def pn_gen_desc():
    """
    根据型号生成描述
    :return:
    """

    imgs_dir = r"D:\py-project\GenerateVideo\assets\imgs"
    descs_dir = r"D:\py-project\GenerateVideo\assets\descs"

    key_idx = 0

    imgs_path_list = os.listdir(imgs_dir)
    i = 0
    while i < len(imgs_path_list):
        if i <= 811:
            i += 1
            continue

        pn = imgs_path_list[i]
        try:
            question = f"根据{pn}的参数和应用，请帮我使用英文生成一段不超过30个单词，不少于20个单词的介绍信息。"
            desc = chat_answer(gpt_api_key[key_idx], question)
            logger.info(f"[生成描述]>> ({i}) {pn} -> {desc}")

            desc_path = os.path.join(descs_dir, f"{pn}.txt")
            with open(desc_path, "w+", encoding="utf-8") as f:
                f.write(desc)
            logger.info(f"[生成描述]>> ({i}) 成功 {pn}")
            i += 1
        except Exception as e:
            logger.error(f"[异常]>> ({i}) {pn} 重试 - {e}")
        finally:
            key_idx += 1
            if key_idx == 4:
                key_idx = 0

            time.sleep(5)
        # break


def work_gen_whole_video(q):
    while not q.empty():
        [i, pn] = q.get()

        try:
            logger.info(f"[开始]>> ({i}) {pn}")

            generateAV(pn)

            logger.success(f"[成功]>> ({i}) {pn}")
        except Exception as e:
            logger.error(f"[异常]>> ({i}) {pn} - {e}")


# 生成全部完整视频
def task_gen_whole_video():
    q = multiprocessing.Queue()
    pool_size = 2

    # 获取所有型号
    videos_dir = r"D:\py-project\GenerateVideo\assets\videos"
    grab_pns = []
    for file_name in os.listdir(videos_dir):
        file_path = os.path.join(videos_dir, file_name)
        if not os.path.isdir(file_path):
            continue

        video_path = os.path.join(file_path, f"final-{file_name}.mp4")
        if os.path.isfile(video_path):
            grab_pns.append(file_name)

    imgs_dir = r"D:\py-project\GenerateVideo\assets\imgs"
    pns = os.listdir(imgs_dir)
    for i, pn in enumerate(pns):
        if pn in grab_pns:
            logger.warning(f"[重复]>> 视频已制作 ({i}) {pn}")
            continue

        q.put([i, pn])
        logger.info(f"({i}) >> {pn}")

        # break

    p_list = []
    for _ in range(pool_size):
        p = multiprocessing.Process(target=work_gen_whole_video, args=(q, ))
        p.start()
        p_list.append(p)

    for p in p_list:
        p.join()



# 提取视频
def extract_vedio():
    final_video_dir = r"D:\py-project\GenerateVideo\assets\final_videos"
    videos_dir = r"D:\py-project\GenerateVideo\assets\videos"

    for pn in os.listdir(videos_dir):
        video_dir = os.path.join(videos_dir, pn)
        if not os.path.isdir(video_dir):
            continue

        old_path = os.path.join(video_dir, f"final-{pn}.mp4")
        final_video_path = os.path.join(final_video_dir, f"final-{pn}.mp4")

        shutil.copy(old_path, final_video_path)
        print(f"{old_path} --> {final_video_path}")
        # break







if __name__ == '__main__':

    # pn_gen_desc()

    # task_gen_whole_video()


    extract_vedio()




    ...

