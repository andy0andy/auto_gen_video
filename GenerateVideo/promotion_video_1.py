import random
import shutil
import pandas as pd
import pyodbc
import pymongo
import json
from loguru import logger
from concurrent.futures import ThreadPoolExecutor, wait
import queue
import threading
import os
import time
import math
import librosa
from copy import deepcopy
from retrying import retry

from custom_function import *
from genAV import Video, Audio, HtAudio

"""
视频样式：上下模糊，中间插入5s短视频
上模糊 标注指定文字
下模糊 标注产品型号
整体视频（长*高） 900*1600，内部小视频 900*1600
无片头片尾，无字幕


步骤：
1. 同步数据库
2. 下载图片等静态资源
3. 生成短描述
4. 生成音频
5. 生成视频
6. 合并音视频
7. 生成对应表格
"""

conn = pyodbc.connect(driver=driver, server=server, user=user, password=password, database=database)
cursor = conn.cursor()

# mongo链接
mcli = pymongo.MongoClient(host=MONGO_IP, port=MONGO_PORT, username=MONGO_USER_NAME, password=MONGO_USER_PASS)
mGenVideo = mcli["Dwight"]["GenVideo"]

# 分类对应分类图
CATEGORYMAP = {
    "Motor Drivers": "IC",
    "Drivers, Receivers, Transceivers": "IC",
    "Low Dropout (LDO) Regulators": "IC",
    "ADCs/DACs - Special Purpose": "IC",
    "Multiplexer Switch ICs": "IC",
    "Drivers": "IC",
    "PMIC - RMS to DC Converters": "IC",
    "Discrete Semiconductor Modules": "IC",
    "16-bit Microcontrollers - MCU": "IC",
    "Voltage Regulators - Linear": "IC",
    "Data Conversion ICs": "IC",
    "Clock & Timer ICs": "IC",
    "MCU MIC (Microcontroller)": "IC",
    "ARM Microcontrollers - MCU": "IC",
    "Controllers": "IC",
    "Switches": "IC",
    "MCU MIC": "IC",
    "Driver ICs": "IC",
    "AC/DC Switching Converters": "IC",
    "Phase Locked Loops - PLL": "IC",
    "Analog to Digital Converters (ADC)": "IC",
    "Digital Potentiometer ICs": "IC",
    "Linear Regulators - Standard": "IC",
    "Amplifiers": "Amplifier ICs",
    "Amplifier ICs": "Amplifier ICs",
    "Embedded Processors & Controllers": "Embedded Processors & Controllers",
    "Interface ICs": "Interface ICs",
    "Interface - Encoders, Decoders, Converters": "Interface ICs",
    "Logic ICs": "Logic ICs",
    "Programmable Logic ICs": "Logic ICs",
    "Memory": "Memory",
    "Memory E(E)PROM/FLASH": "Memory",
    "Microcontrollers": "Microcontrollers",
    "Battery Management": "Power Management ICs",
    "Power Management ICs": "Power Management ICs"
}


def migrate_to_mongo():
    """
    slqserver 同步mongo
    :return:
    """

    item_list = []

    sql = "SELECT * FROM [Vae].[dbo].[Product_weik];"
    cursor.execute(sql)
    for row in cursor.fetchall():

        # 处理图片链接
        pn_imgs = []
        for img_link in row[1].split(","):
            img_link = f"https://www.{row[3].lower()}.com" + img_link.strip()
            pn_imgs.append(img_link)

        item = {
            "ProductName": row[4],
            "NickName": trans_pn(row[4]),
            "ProductCategoryName": row[5],
            "Target": row[3],
            "RelatedProduct": json.loads(row[2]),
            "ProductImage": pn_imgs
        }
        item_list.append(item)

        if len(item_list) == 1000:
            mGenVideo.insert_many(item_list)
            item_list = []
    else:
        mGenVideo.insert_many(item_list)
        item_list = []

    logger.success(f"[]>> 数据已同步至mongo")


def download_Product_images():
    """
    下载库中所有图片
    :return:
    """
    pool_size = 30
    pool = ThreadPoolExecutor(max_workers=pool_size)
    q = queue.Queue()

    # ==================
    def work(q):

        # thread_name = threading.current_thread().name

        while not q.empty():
            row = q.get()

            nick_name = row["NickName"]
            target = row["Target"]
            img_links = row["ProductImage"]

            img_dir = os.path.join(root_path, os.path.join("imgs", f"{target}_{nick_name}"))
            if not os.path.exists(img_dir):
                os.mkdir(img_dir)

            # 下载
            for i, im in enumerate(img_links):

                img_path = os.path.join(img_dir, f"{nick_name}_{i}.{im.split('.')[-1]}")
                if os.path.exists(img_path):
                    continue

                try:
                    download_image(im, img_path)

                    # 验证图片
                    if not check_image(img_path):
                        os.remove(img_path)
                        logger.warning(f"[图片异常]>> 已删除 ({nick_name}, {target}) {im}")
                    else:
                        logger.success(f"[下载图片]>> {im} -> {img_path}")
                except Exception as e:
                    logger.error(f"[下载图片异常]>> ({nick_name}, {target}) {im} | {e}")

    # ==================

    for row in mGenVideo.find({}, {"NickName": 1, "Target": 1, "ProductImage": 1}):
        q.put(row)

    all_tasks = []
    for _ in range(pool_size):
        all_tasks.append(pool.submit(work, q))
    wait(all_tasks)


def pn_gen_desc():
    """
    根据型号生成描述
    :return:
    """

    q = queue.Queue()
    pool_size = 2
    pool = ThreadPoolExecutor(max_workers=pool_size)

    # =============
    def work(q):
        while not q.empty():
            row = q.get()
            pn = row["ProductName"]

            try:
                question = f"Please summarize the features and applications of {pn} for me, answer in 40 words or less."
                desc = chat_answer_for_3(gpt3_key[0], question)

                item = {
                    "GptDesc": desc
                }

                # 入库
                mGenVideo.update_one({"_id": row["_id"]}, {"$set": item})
                logger.success(f"[生成描述]>> 成功 {pn} -> {desc}")

            except Exception as e:
                logger.error(f"[异常]>> {pn} | {e}")

    # =============

    filter_data = {"GptDesc": {"$exists": False}, "GptDesc": ""}
    for row in mGenVideo.find(filter_data, {"ProductName": 1}):
        q.put(row)

        # break

    all_tasks = []
    for _ in range(pool_size):
        all_tasks.append(pool.submit(work, q))
    wait(all_tasks)


@retry(stop_max_attempt_number=3, wait_fixed=1000 * 2)
def generate_audio(web_site: str, product: str, gpt_desc: str):
    """
    生成音频
    :return:
    """

    pn = trans_pn(product)

    voice_list = [
        "s3://mockingbird-prod/olivia_vo_training_4376204f-a411-4e5d-a5c0-ce6cc3908052/voices/speaker/manifest.json",
        "s3://peregrine-voices/nolan saad parrot/manifest.json",
        "s3://mockingbird-prod/larry_vo_narrative_4bd5c1bd-f662-4a38-b5b9-76563f7b92ec/voices/speaker/manifest.json",
        "s3://voice-cloning-zero-shot/801a663f-efd0-4254-98d0-5c175514c3e8/jennifer/manifest.json",
        "s3://peregrine-voices/hudson saad parrot/manifest.json",
        "s3://voice-cloning-zero-shot/418a94fa-2395-4487-81d8-22daf107781f/george/manifest.json",
    ]

    hta = HtAudio(ht_api_key, ht_secret_key)

    audios_dir = os.path.join(root_path, "audios")
    pn_audios_dir = os.path.join(audios_dir, f"{web_site}_{pn}")
    if not os.path.exists(pn_audios_dir):
        os.mkdir(pn_audios_dir)
    audio_path = os.path.join(pn_audios_dir, f"{pn}.mp3")
    if os.path.exists(audio_path):
        logger.warning(f"[]>> ({web_site}, {product}) 已生成音频")
        return

    voice = random.choice(voice_list)  # 随机音色
    hta.gen_audio(gpt_desc, voice, audio_path)


def batch_gen_audio():
    """
    批量生成音频
    :return:
    """

    pool_size = 2
    pool = ThreadPoolExecutor(max_workers=pool_size)
    q = queue.Queue()

    # =========================
    def work(q):
        while not q.empty():
            [web_site, product, gpt_desc] = q.get()
            logger.info(f"[]>> 准备生成音频 ({web_site}, {product}) 单词数：{len(gpt_desc.split())}")

            try:
                generate_audio(web_site, product, gpt_desc)
            except Exception as e:
                logger.error(f"[生成音频异常]>> ({web_site}, {product}) | {e}")

    # =========================

    for row in mGenVideo.find({"GptDesc": {"$exists": True}}):
        product = row["ProductName"]
        web_site = row["Target"]
        gpt_desc = row["GptDesc"]

        q.put([web_site, product, gpt_desc])

    all_tasks = []
    for _ in range(pool_size):
        all_tasks.append(pool.submit(work, q))
    wait(all_tasks)


def generate_video(web_site: str, product: str, category: str):
    """
    生成视频，视频上下模糊
    :param web_site:
    :param pn:
    :return:
    """

    pn = trans_pn(product)

    # 文件路径
    imgs_dir = os.path.join(root_path, "imgs")
    videos_dir = os.path.join(root_path, "videos")
    audios_dir = os.path.join(root_path, "audios")
    videos_mid_dir = os.path.join(root_path, "videos_mid")
    cate_img_dir = os.path.join(root_path, "category_images")

    pn_imgs_dir = os.path.join(imgs_dir, f"{web_site}_{pn}")
    pn_videos_dir = os.path.join(videos_dir, f"{web_site}_{pn}")
    if not os.path.exists(pn_videos_dir):
        os.mkdir(pn_videos_dir)

    pn_audios_dir = os.path.join(audios_dir, f"{web_site}_{pn}")
    audio_path = os.path.join(pn_audios_dir, f"{pn}.mp3")

    # 插入小视频路径, 以及时长
    mid_file_path = os.path.join(videos_mid_dir, random.choice(os.listdir(videos_mid_dir)))
    mid_duration = 5
    logger.info(f"[插入视频及时长]>> ({mid_duration}s) - {mid_file_path}")

    # 分类图片列表
    cate_img_path_list = [
        os.path.join(cate_img_dir, f"{CATEGORYMAP[category]}-01.jpg"),
        os.path.join(cate_img_dir, f"{CATEGORYMAP[category]}-02.jpg"),
    ]
    for cate_img_path in cate_img_path_list:
        if not os.path.exists(cate_img_path):
            logger.warning(f"[无分类图片]>> 跳过 ({web_site}, {pn})")
            return None

    # 音频时长
    audio_duration = math.ceil(librosa.get_duration(filename=audio_path))  # 不准
    video_duration = audio_duration - mid_duration
    logger.info(f"音频时长 ({audio_duration}s)")

    # 视频
    img_path_list = [os.path.join(pn_imgs_dir, p) for p in os.listdir(pn_imgs_dir)]
    if len(img_path_list) == 0:
        logger.warning(f"[无图片]>> 跳过 ({web_site}, {pn})")
        return None
    elif len(img_path_list) > 4:  # 视频要求 15s， 所以最多取4张图
        use_img_path_list = deepcopy(img_path_list[:4])
    else:
        use_img_path_list = deepcopy(img_path_list)
    video_path = os.path.join(pn_videos_dir, f"{pn}.mp4")  # 视频路径
    if os.path.exists(video_path):
        logger.warning(f"[]>> 已生成视频，跳过 ({web_site}, {product})")
        return

    # 初始视频
    max_show = 3  # 图片最长展示时间 单位s
    video_size = (900, 800)
    video = Video()
    video.imgs_to_video(use_img_path_list, video_path, video_duration, video_size, cate_img_path_list, max_show)  # 图片转视频
    video.trans_ecfect(video_path, len(use_img_path_list), video_duration)  # 添加随机转场动画

    # 完整无音频视频
    no_audio_path = os.path.join(pn_videos_dir, f"{pn}_no_audio.mp4")  # 无音频视频路径
    video.merge_videos_at_time(no_audio_path, [video_path, mid_file_path], video_duration/len(use_img_path_list)*3)

    # 高斯模糊
    blow_up_size = (900, 1600)
    video_blur_path = os.path.join(pn_videos_dir, f"{pn}_blur.mp4")  # 蒙版视频路径
    video.up_down_gaussian(no_audio_path, video_blur_path, blow_up_size)

    # 添加文字
    txt = "Popular Electronic Components"
    video_txtup_path = os.path.join(pn_videos_dir, f"{pn}_txt_up.mp4")  # 文字视频路径
    video.add_fixed_text(video_blur_path, video_txtup_path, txt, ("center", -600))

    video_txtdown_path = os.path.join(pn_videos_dir, f"{pn}_txt_down.mp4")  # 文字视频路径
    video.add_fixed_text(video_txtup_path, video_txtdown_path, product, ("center", 500))

    os.remove(video_path)
    os.remove(no_audio_path)
    os.remove(video_blur_path)
    os.remove(video_txtup_path)
    shutil.move(video_txtdown_path, video_path)

    logger.success(f"[]>> 生成无音频完整视频 -> {video_path}")


def batch_gen_video():
    """
    批量生成视频
    :return:
    """

    pool_size = 2
    pool = ThreadPoolExecutor(max_workers=pool_size)
    q = queue.Queue()

    # =========================
    def work(q):
        while not q.empty():
            [web_site, product, category] = q.get()
            logger.info(f"[]>> 准备生成视频 ({web_site}, {product})")

            try:
                generate_video(web_site, product, category)
            except Exception as e:
                logger.error(f"[生成视频异常]>> ({web_site}, {product}) | {e}")

    # =========================

    for row in mGenVideo.find():
        product = row["ProductName"]
        web_site = row["Target"]
        category = row["ProductCategoryName"]

        q.put([web_site, product, category])

    all_tasks = []
    for _ in range(pool_size):
        all_tasks.append(pool.submit(work, q))
    wait(all_tasks)


def merge_whole_video():
    """
    合并完整视频
    :return:
    """

    pool_size = 2
    pool = ThreadPoolExecutor(max_workers=pool_size)
    q = queue.Queue()

    # =========================
    def work(q):
        video = Video()

        while not q.empty():
            [web_site, product] = q.get()
            logger.info(f"[]>> 准备合并音视频 ({web_site}, {product})")

            try:

                pn = trans_pn(product)

                # 文件路径
                videos_dir = os.path.join(root_path, "videos")
                audios_dir = os.path.join(root_path, "audios")
                bgm_dir = os.path.join(root_path, "videos_bgm")

                pn_videos_dir = os.path.join(videos_dir, f"{web_site}_{pn}")
                pn_audios_dir = os.path.join(audios_dir, f"{web_site}_{pn}")

                video_path = os.path.join(pn_videos_dir, f"{pn}.mp4")
                audio_path = os.path.join(pn_audios_dir, f"{pn}.mp3")
                if not os.path.exists(video_path) or not os.path.exists(audio_path):
                    logger.warning(f"[]>> 缺少合并音视频文件 ({web_site}, {product})")
                    continue

                # 合并音视频
                av_path = os.path.join(pn_videos_dir, f"{pn}_av.mp4")
                video.merge_av(av_path, audio_path, video_path)

                # 合并bgm
                bgm_list = os.listdir(bgm_dir)
                bgm_path = os.path.join(bgm_dir, random.choice(bgm_list))
                target_path = os.path.join(pn_videos_dir, f"{pn}_bgm.mp4")
                video.merge_bgm(av_path, bgm_path, target_path)

                # 最终视频路径
                final_path = os.path.join(pn_videos_dir, f"{pn}_final.mp4")
                os.remove(av_path)
                os.rename(target_path, final_path)


            except Exception as e:
                logger.error(f"[合并音视频异常]>> ({web_site}, {product}) | {e}")

    # =========================

    for row in mGenVideo.find():
        product = row["ProductName"]
        web_site = row["Target"]

        q.put([web_site, product])

    all_tasks = []
    for _ in range(pool_size):
        all_tasks.append(pool.submit(work, q))
    wait(all_tasks)


def gen_report():
    """
    生成最终报表
    :return:
    """

    item_list = []
    report_path = os.path.join(root_path, "report.csv")

    videos_dir = os.path.join(root_path, "videos")
    videos_final_dir = os.path.join(root_path, "videos_final")

    fields_data = {"ProductName": 1, "NickName": 1, "ProductCategoryName": 1, "Target": 1, "RelatedProduct": 1}
    for row in mGenVideo.find({}, fields_data):
        row.pop("_id")

        item = dict(row)
        item["RelatedProduct"] = ", ".join(row["RelatedProduct"])

        pn_videos_dir = os.path.join(videos_dir, f"{row['Target']}_{row['NickName']}")
        video_path = os.path.join(pn_videos_dir, f"{row['NickName']}_final.mp4")
        if not os.path.exists(video_path):
            logger.warning(f"[]>> 不存在视频 {video_path}")
            continue
        else:
            pn_videos_final_dir = os.path.join(videos_final_dir, f"{row['Target']}")
            if not os.path.isdir(pn_videos_final_dir):
                os.mkdir(pn_videos_final_dir)
            video_final_path = os.path.join(pn_videos_final_dir, f"{row['NickName']}.mp4")
            shutil.copy(video_path, video_final_path)
            logger.info(f"[]>> 转移视频 {video_path} -> {video_final_path}")

        item_list.append(item)
        # break

    # 保存报告
    df = pd.DataFrame(item_list)
    df.sort_values(by=["Target", "ProductName"], inplace=True, ascending=False)
    df.to_csv(report_path, index=False)
    logger.success(f"[]>> 生成报告 {report_path}")







if __name__ == '__main__':
    # 1. 同步数据库
    # migrate_to_mongo()

    # 2. 下载图片等静态资源
    # download_Product_images()

    # 3. 生成短描述
    # pn_gen_desc()

    # 4. 生成音频
    # batch_gen_audio()
    # generate_audio("Avaq", "LM1972M", "LM1972M is a digitally controlled potentiometer with a built-in stereo audio processor. It allows volume, balance, tone control, and loudness adjustment. It is used in audio applications where precise and convenient control of audio parameters is required, such as audio mixers, amplifiers, and AV receivers.")  # *

    # 5. 生成视频
    # batch_gen_video()
    # generate_video("Avaq", "ADM3202ARUZ", "Drivers")  # *

    # 6. 合并音视频
    # merge_whole_video()

    # 7. 生成对应表格
    # gen_report()




    ...
