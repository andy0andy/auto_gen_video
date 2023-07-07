import random
import time
import datetime
from typing import Optional, List, Tuple
import cv2
from tqdm import tqdm  # 进度条展示
from moviepy.editor import *
from loguru import logger
import pyttsx4
import librosa
import math
import os
import requests
from jsonpointer import resolve_pointer
from faster_whisper import WhisperModel
from pycaption import SRTReader


"""
图片 + 文字 => 视频
"""

# 信息
app_id = "***"
api_key = "***"
secret_key = "***"


class Video(object):

    def imgs_to_video(self, img_path_list: List[str], video_path: str, duration: int,
                      img_size: Optional[Tuple] = (1920, 1080)) -> None:
        """
        把多张图片 合并成 视频
        :param img_path_list: 图片列表
        :param video_path: 视频保存位置
        :param duration: 视频时长 秒
        :param img_size: 图片尺寸
        :return:
        """

        max_show = 5  # 一张图最长展示5s
        if duration / len(img_path_list) > max_show:  # 每张图片展示时长超过最大展示时长
            new_len = math.ceil(duration / max_show) - len(img_path_list)  # 还需 new_len 张图平摊时长
            img_path_list.extend(random.choices(img_path_list, k=new_len))

        fps = len(img_path_list) / duration  # 帧率， fps = 1 1s播放1张图

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 编码参数
        videoWriter = cv2.VideoWriter(video_path, fourcc, fps, img_size)

        for img_path in tqdm(img_path_list, desc="合成中: "):
            frame = cv2.imread(img_path)
            frame = cv2.resize(frame, img_size)  # 生成视频   图片尺寸和设定尺寸相同
            videoWriter.write(frame)  # 将图片写进视频里

        videoWriter.release()  # 释放资源

        logger.info(f"图片已合成视频 {video_path}")

    def trans_ecfect(self, video_path: str, range_count: int, duration: int) -> None:
        """
        视频添加转场特效
        fadein： 淡入
        fadeout： 淡出
        :param video_path: 视频保存位置
        :param range_count: 视频合并所用图片数量
        :param duration: 视频时长 秒
        :return:
        """

        effect_list = ["fadein", "fadeout"]
        video = VideoFileClip(video_path)

        step_time = duration / range_count  # 每一张图片展示时间
        st = 0

        clip_list = []
        for i in range(range_count):
            effect = random.choice(effect_list)

            if effect == "fadein":
                newclip = vfx.fadein(video.subclip(st, st + step_time), duration=1, initial_color=0.5)
            elif effect == "fadeout":
                newclip = vfx.fadeout(video.subclip(st, st + step_time), duration=1, final_color=0.5)

            clip_list.append(newclip)

            st += step_time

        final_clip = concatenate_videoclips(clip_list)
        final_clip.to_videofile(video_path, fps=32, remove_temp=True)  # fps越大，越耗时间

        logger.info(f"视频已添加转场 {video_path}")

    def merge_av(self, av_path: str, audio_path: str, video_path: str) -> None:
        """
        合并音频，视频
        :param audio_path:
        :param video_path:
        :return:
        """

        audio = AudioFileClip(audio_path)
        video = VideoFileClip(video_path)

        newvideo = video.set_audio(audio)  # 音频文件
        newvideo.write_videofile(av_path, audio_codec='libmp3lame',
                                 threads=os.cpu_count())  # 保存合成视频，注意加上参数audio_codec='aac'，否则音频无声音

        logger.info(f"合并音视频 {av_path}")

    def merge_videos(self, video_path: str, video_paths: List[str]):
        """
        拼接视频
        :param video_path:  拼接后视频
        :param video_paths: 待拼接视频列表
        :return:
        """

        videos = [VideoFileClip(vp) for vp in video_paths]
        final_clip = concatenate_videoclips(videos)
        final_clip.to_videofile(video_path, fps=32, remove_temp=True)

    def merge_srt_video(self, video_path: str, srt_path: str, avs_path: str):
        """
        添加字幕
        :param video_path:  视频路径
        :param srt_path: 字幕路径
        :param avs_path: 字幕视频音频路径
        :return:
        """

        video = VideoFileClip(video_path)
        # 获取视频的宽度和高度
        w, h = video.w, video.h

        reader = SRTReader()
        with open(srt_path, "r+", encoding="utf-8") as f:
            captions = reader.read(f.read())

        txt_clip_list = [video]
        for caption in captions.get_captions("en-US"):
            start = caption.start
            end = caption.end
            text = caption.get_text()

            start = start // 1000 / 1000
            end = end // 1000 / 1000

            txt_clip = TextClip(text, fontsize=60, size=(w - 20, 140), font='微软雅黑',
                                method="caption", align='South', color='#FFF',
                                transparent=True, stroke_color="black", stroke_width=2)
            txt_clip = txt_clip.set_position((10, h - 150)).set_duration(end-start).set_start(start)
            txt_clip_list.append(txt_clip)

        # 合成视频，写入文件
        newvideo = CompositeVideoClip(txt_clip_list)
        newvideo.write_videofile(avs_path, fps=32)

        logger.info(f"添加字幕 {srt_path} -> {avs_path}")


class Audio(object):
    """
    纯离线生成音频，但过于机械
    """

    # 创建对象
    engine = pyttsx4.init()

    def txt_to_audio(self, txt: str, audio_path: str) -> None:
        """
        文本转音频
        :param txt: 文本
        :param audio_path: 音频位置
        :return:
        """

        # 获取当前语音声音的详细信息  # 这里我也是找到的实例代码感觉写的很矛盾，最后发出的还是女声
        voices = self.engine.getProperty('voices')
        # 设置当前语音声音为女性，当前声音不能读中文
        self.engine.setProperty('voice', voices[1].id)

        # 保存音频
        self.engine.save_to_file(txt, filename=audio_path)
        self.engine.runAndWait()

        logger.info(f"文本已生成语音 {audio_path}")


class BaiDuAudio(object):

    def __init__(self, app_id: str, api_key: str, secret_key: str):
        self._app_id = app_id
        self._api_key = api_key
        self._secret_key = secret_key

        self._access_token = self.get_access_token

    @property
    def get_access_token(self) -> str:
        url = "https://aip.baidubce.com/oauth/2.0/token"

        params = {
            "grant_type": "client_credentials",
            "client_id": self._api_key,
            "client_secret": self._secret_key
        }

        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }

        resp = requests.post(url, params=params, headers=headers)
        resp = resp.json()

        return resp["access_token"]

    def create_long_txt(self, txt_list: List[str]) -> str:
        """
        根据文本内容、音频格式、音库等参数创建语音合成任务
        :param txt_list: 文本内容
        :return: 任务id
        """

        url = "https://aip.baidubce.com/rpc/2.0/tts/v1/create"

        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }

        params = {
            "access_token": self._access_token
        }

        form_data = {
            "text": txt_list,
            "format": "mp3-16k",  # 音频格式: "mp3-16k"，"mp3-48k"，"wav"，"pcm-8k"，"pcm-16k"，默认为mp3-16k
            "voice": 5,
            # 基础音库：度小宇=1，度小美=0，度逍遥（基础）=3，度丫丫=4；精品音库：度逍遥（精品）=5003，度小鹿=5118，度博文=106，度小童=110，度小萌=111，度米朵=103，度小娇=5。默认为度小美
            "lang": "zh",
            "speed": 5,  # 取值0-15，默认为5中语速
            "pitch": 5,  # 取值0-15，默认为5中语调
            "enable_subtitle": 0,  # 取值范围0, 1, 2，默认为0。0表示不开启字幕时间戳，1表示开启句级别字幕时间戳，2表示开启词级别字幕时间戳
            "break": 0  # 取值 0-5000 ，单位ms，用于合成文本分段传入时设置段落间间隔。
        }

        resp = requests.post(url, params=params, json=form_data, headers=headers)
        resp_json = resp.json()

        task_id = resp_json.get("task_id")
        if task_id:
            return task_id
        else:
            raise Exception(f"[异常]>> 创建语音合成任务失败 - {resp.text}")

    def query_long_txt(self, task_id: str) -> str:
        """
        根据task_id的数组批量查询语音合成任务结果
        :param task_id:
        :return: 语音文件url
        """

        url = "https://aip.baidubce.com/rpc/2.0/tts/v1/query"

        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }

        params = {
            "access_token": self._access_token
        }

        form_data = {
            "task_ids": [task_id]
        }

        resp = requests.post(url, params=params, json=form_data, headers=headers)
        resp_json = resp.json()

        task_status = resolve_pointer(resp_json, "/tasks_info/0/task_status")
        if task_status == "Success":
            speech_url = resolve_pointer(resp_json, "/tasks_info/0/task_result/speech_url")
            # sentences = resolve_pointer(resp_json, "/tasks_info/0/task_result/speech_timestamp/sentences")
            return speech_url
        else:
            logger.warning(f"[异常]>> 查询合成任务结果异常 - {resp.text}")

    def gen_mp3_srt(self, speech_url: str, audio_path: str, srt_path: str):
        """
        下载音频 生成字幕
        :param txt_data: query_long_txt() 返回结构
        :param audio_path: 音频路径
        :param srt_path: 字幕文件路径
        :return:
        """

        resp = requests.get(speech_url)
        with open(audio_path, "wb+") as f:
            f.write(resp.content)
        logger.info(f"[音频]>> 下载音频 {audio_path}")

        # 字幕
        model_size = "large-v2"
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        segments, info = model.transcribe(audio_path, language="en", beam_size=5)

        with open(srt_path, "w+", encoding="utf-8") as f:
            for i, segment in enumerate(segments):
                s = (datetime.datetime.fromtimestamp(segment.start) - datetime.timedelta(hours=8)).strftime(
                    "%H:%M:%S,%f")[:-3]
                e = (datetime.datetime.fromtimestamp(segment.end) - datetime.timedelta(hours=8)).strftime(
                    "%H:%M:%S,%f")[:-3]
                line = f"{i + 1}\n{s} --> {e}\n{segment.text.strip()}\n\n"

                f.write(line)
                f.flush()
                print(line)

        logger.info(f"[字幕]>> 生成成功 {srt_path}")


def run():
    imgs_dir = r"D:\py-project\GenerateVideo\assets\imgs"
    desc_dir = r"D:\py-project\GenerateVideo\assets\descs"
    videos_dir = r"D:\py-project\GenerateVideo\assets\videos"
    audios_dir = r"D:\py-project\GenerateVideo\assets\audios"

    pn = "LM20"

    pn_imgs_dir = os.path.join(imgs_dir, pn)
    pn_videos_dir = os.path.join(videos_dir, pn)
    pn_audios_dir = os.path.join(audios_dir, pn)

    # 音频
    with open(os.path.join(desc_dir, f"{pn}.txt"), "r", encoding="utf-8") as f:
        txt = f.read()
    txt = txt.strip().replace("\n", "").replace("\t", "")

    audio_path = os.path.join(pn_audios_dir, f"{pn}.mp3")
    srt_path = os.path.join(pn_audios_dir, f"{pn}.srt")

    baiDuAudio = BaiDuAudio(app_id, api_key, secret_key)

    task_id = baiDuAudio.create_long_txt([txt])  # 创建音频生成任务
    speech_url = None
    for _ in range(12):
        speech_url = baiDuAudio.query_long_txt(task_id)  # 查询生成任务

        if speech_url is None:
            time.sleep(5)
        else:
            break
    else:
        raise Exception(f"[异常]>> 音频生成异常 - {task_id}")

    # speech_url = 'http://aipe-speech.bj.bcebos.com/text_to_speech/2023-07-05/64a524ec1134240001adac67/speech/0.mp3?authorization=bce-auth-v1%2F8a6ca9b78c124d89bb6bca18c6fc5944%2F2023-07-05T08%3A08%3A24Z%2F259200%2F%2Ffd436c1e8dfcf57e73f135c8ba42505bc4cef716f76874d3bd1d05cc45aff417'
    baiDuAudio.gen_mp3_srt(speech_url, audio_path, srt_path)

    audio_duration = math.ceil(librosa.get_duration(filename=audio_path))  # 不准
    logger.info(f"音频时长 ({audio_duration}s) {audio_path}")

    # 视频
    img_path_list = [os.path.join(pn_imgs_dir, p) for p in os.listdir(pn_imgs_dir)]
    video_path = os.path.join(pn_videos_dir, f"{pn}.mp4")  # 视频路径
    video_duration = audio_duration

    video = Video()
    video.imgs_to_video(img_path_list, video_path, video_duration)
    video.trans_ecfect(video_path, len(img_path_list), video_duration)

    # 合并音视频
    split_video_path = os.path.split(video_path)  # 完整音视频路径
    av_path = os.path.join(split_video_path[0], "av-" + split_video_path[-1])
    video.merge_av(av_path, audio_path, video_path)

    # 添加字幕
    split_av_path = os.path.split(av_path)  # 完整音视频+字幕路径
    avs_path = os.path.join(split_av_path[0], split_av_path[-1].replace("av", "avs"))
    video.merge_srt_video(av_path, srt_path, avs_path)

    # 拼接片头
    logo_video_path = os.path.join(videos_dir, "logo.mp4")
    final_video_path = avs_path.replace("avs", "final")   # 最终视频路径
    video.merge_videos(final_video_path, [logo_video_path, avs_path])
    logger.success(f"[完成]>> 视频制作 - {final_video_path}")


if __name__ == '__main__':
    run()


    ...
