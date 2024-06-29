import random
import datetime
import shutil
import sys
from typing import Optional, List, Tuple
import cv2
from tqdm import tqdm  # 进度条展示
from moviepy.editor import *
from loguru import logger
import gtts
import librosa
import math
import os
import requests
from jsonpointer import resolve_pointer
from faster_whisper import WhisperModel
from pycaption import SRTReader
from copy import deepcopy
from skimage.filters import gaussian
from pyht import Client, TTSOptions, Format

from settings import *


class Video(object):

    def imgs_to_video(self, img_path_list: List[str], video_path: str, duration: int,
                      img_size: Optional[Tuple] = (1920, 1080)
                      , cate_img_path_list: Optional[List[str]] = []
                      , max_show: Optional[int] = 3) -> None:
        """
        把多张图片 合并成 视频
        :param img_path_list: 图片列表
        :param video_path: 视频保存位置
        :param duration: 视频时长 秒
        :param img_size: 图片尺寸 (宽，高)
        :param cate_img_path_list: 分类图片列表，分类图顺序： 分类图-实物图-分类图-视频-实物图
        :param max_show: 一张图最长展示时间 单位s
        :return:
        """

        cate_img_len = 0
        if cate_img_path_list and len(cate_img_path_list) >= 2:
            cate_img_len = 2
            cate_img_path_list = cate_img_path_list[:2]

        if duration / (len(img_path_list) + cate_img_len) > max_show:  # 每张图片展示时长超过最大展示时长
            new_len = math.ceil(duration / max_show) - (len(img_path_list) + cate_img_len)  # 还需 new_len 张图平摊时长
            img_path_list.extend(random.choices(img_path_list, k=new_len))

        # 插入分类图
        if cate_img_len == 2:
            img_path_list.insert(0, cate_img_path_list[0])
            img_path_list.insert(2, cate_img_path_list[1])

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

    def merge_bgm(self, video_path: str, bgm_path: str, target_path: str):
        """
        为视频添加一个背景音乐
        多轨音频合成
        """

        # 需添加背景音乐的视频
        video_clip = VideoFileClip(video_path)
        # 提取视频对应的音频
        video_audio_clip = video_clip.audio

        # 背景音乐 并调节音量
        audio_clip = AudioFileClip(bgm_path).volumex(0.2)
        # 设置背景音乐循环，时间与视频时间一致
        audio = afx.audio_loop(audio_clip, duration=video_clip.duration)
        # 视频声音和背景音乐，音频叠加
        double_audio_clip = CompositeAudioClip([video_audio_clip, audio])

        # 视频写入背景音
        final_video = video_clip.set_audio(double_audio_clip)

        # 将处理完成的视频保存
        final_video.write_videofile(target_path)
        logger.success(f"[添加bgm]>> {target_path}")

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

        logger.info(f"合并视频 {video_path}")

    def merge_videos_at_time(self, video_path: str, video_paths: List[str], duration: float):
        """
        在原视频的第N秒拼接视频
        :param video_path:  拼接后视频
        :param video_paths: 待拼接视频列表, 列表只有2个元素 【原视频，拼接视频】
        :param duration: 在原视频的第几秒拼接
        :return:
        """

        if len(video_paths) != 2:
            logger.warning(f"[]>> 缺少拼接视频")
            return

        videos = [VideoFileClip(vp) for vp in video_paths]
        clip_prev = videos[0].subclip(0, duration)
        clip_next = videos[0].subclip(duration, videos[0].duration)

        final_clip = concatenate_videoclips([clip_prev, videos[1], clip_next])
        final_clip.to_videofile(video_path, fps=32, remove_temp=True)

        logger.info(f"合并视频 {video_path}")

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
            txt_clip = txt_clip.set_position((10, h - 150)).set_duration(end - start).set_start(start)
            txt_clip_list.append(txt_clip)

        # 合成视频，写入文件
        newvideo = CompositeVideoClip(txt_clip_list)
        newvideo.write_videofile(avs_path, fps=32)

        logger.info(f"添加字幕 {srt_path} -> {avs_path}")

    def up_down_gaussian(self, input_path, output_path, blow_up: tuple):
        """
        给视频添加上下模糊，并居中展示
        :param input_path:
        :param output_path:
        :param blow_up: 放大后视频大小（宽，高）
        :return:
        """

        def blur(image):
            """ Returns a blurred (radius=30 pixels) version of the image """
            return gaussian(image.astype(float), sigma=30)

        # 加载视频文件
        clip1 = VideoFileClip(input_path)

        # 将视频放大并加蒙版遮罩
        clip2 = VideoFileClip(input_path, audio=False, has_mask=True).resize(blow_up)
        clip2 = clip2.fl_image(blur)

        # 将小的视频叠在大视频的居中位置
        clip2 = CompositeVideoClip([clip2, clip1.set_pos("center")])

        # 输出编辑完成的视频
        clip2.write_videofile(output_path)

        logger.info(f"[生成蒙版视频]>> ({blow_up}) {output_path}")

    def add_fixed_text(self, input_path: str, output_path: str, text: str, pos: tuple):
        """
        添加固定文字在固定位置
        :return:
        """

        clip = VideoFileClip(input_path)

        text_clip = TextClip(text, fontsize=70, font='微软雅黑', size=clip.size,
                             method="caption", align='center', color='black',
                             transparent=True, stroke_color="black", stroke_width=2)
        text_clip = text_clip.set_position(pos).set_duration(clip.duration)

        # 合成视频，写入文件
        newclip = CompositeVideoClip([clip, text_clip])
        newclip.write_videofile(output_path, fps=clip.fps)

        logger.info(f"[添加文字]>> {text}")


class Audio(object):
    """
    纯离线生成音频
    """

    def txt_to_audio(self, txt: str, audio_path: str) -> None:
        """
        文本转音频
        :param txt: 文本
        :param audio_path: 音频位置
        :return:
        """

        tts = gtts.gTTS(text=txt, lang='en', tld='com')
        tts.save(audio_path)

        logger.info(f"文本已生成语音 {audio_path}")

    def audio_gen_srt(self, audio_path: str, srt_path: str):
        """
        音频生成字幕
        :param audio_path: 音频路劲
        :param srt_path: 字幕路径
        :return:
        """

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

        logger.info(f"音频已生成字幕 {srt_path}")


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


class HtAudio(object):
    def __init__(self, api_key: str, secret_key: str):
        self.api_key = api_key
        self.secret_key = secret_key

    def voice_list(self):
        """
        获取声音列表
        :return:
        """

        url = "https://api.play.ht/api/v2/voices"

        headers = {
            "accept": "application/json",
            "AUTHORIZATION": self.secret_key,
            "X-USER-ID": self.api_key
        }

        response = requests.get(url, headers=headers)

        return response.json()

    def gen_audio(self, text: str, voice: str, file_path: str):

        # Initialize PlayHT API with your credentials
        client = Client(self.api_key, self.secret_key)

        # configure your stream
        options = TTSOptions(
            voice=voice,
            format=Format.FORMAT_MP3,
            speed=1.3
        )

        with open(file_path, "ab+") as af:
            for chunk in client.tts(text=text, voice_engine="PlayHT2.0-turbo", options=options):
                if chunk == b'':
                    client.close()
                    break

                af.write(chunk)

        logger.info(f"[生成音频]>> {file_path}")


def generateAV(web_site: str, pn: str):
    imgs_dir = os.path.join(root_path, "imgs")
    desc_dir = os.path.join(root_path, "descs")
    videos_dir = os.path.join(root_path, "videos")
    audios_dir = os.path.join(root_path, "audios")
    end_video_dir = os.path.join(root_path, "videos_end")

    # 有结尾视频站点
    end_site_list = [ws.replace(".mp4", "") for ws in os.listdir(end_video_dir)]

    pn_imgs_dir = os.path.join(imgs_dir, f"{web_site}_{pn}")
    pn_videos_dir = os.path.join(videos_dir, f"{web_site}_{pn}")
    pn_audios_dir = os.path.join(audios_dir, pn)

    if not os.path.exists(pn_videos_dir):
        os.mkdir(pn_videos_dir)
    if not os.path.exists(pn_audios_dir):
        os.mkdir(pn_audios_dir)

    # 音频
    with open(os.path.join(desc_dir, f"{pn}.txt"), "r", encoding="utf-8") as f:
        txt = f.read()
    txt = txt.strip().replace("\n", "").replace("\t", "")

    audio_path = os.path.join(pn_audios_dir, f"{pn}.mp3")
    srt_path = os.path.join(pn_audios_dir, f"{pn}.srt")

    # 百度接口生成音频字幕
    # baiDuAudio = BaiDuAudio(app_id, api_key, secret_key)
    #
    # task_id = baiDuAudio.create_long_txt([txt])  # 创建音频生成任务
    # speech_url = None
    # for _ in range(12):
    #     speech_url = baiDuAudio.query_long_txt(task_id)  # 查询生成任务
    #
    #     if speech_url is None:
    #         time.sleep(5)
    #     else:
    #         break
    # else:
    #     raise Exception(f"[异常]>> 音频生成异常 - {task_id}")
    #
    # baiDuAudio.gen_mp3_srt(speech_url, audio_path, srt_path)

    # 本地生成语音字幕
    audio = Audio()
    audio.txt_to_audio(txt, audio_path)
    # audio.audio_gen_srt(audio_path, srt_path)

    audio_duration = math.ceil(librosa.get_duration(filename=audio_path))  # 不准
    logger.info(f"音频时长 ({audio_duration}s) {audio_path}")

    # 视频
    img_path_list = [os.path.join(pn_imgs_dir, p) for p in os.listdir(pn_imgs_dir)]
    if len(img_path_list) > 4:  # 视频要求 15s， 所以最多取4张图
        use_img_path_list = deepcopy(img_path_list[:4])
    else:
        use_img_path_list = deepcopy(img_path_list)
    video_path = os.path.join(pn_videos_dir, f"{pn}.mp4")  # 视频路径
    video_duration = audio_duration
    if len(use_img_path_list) <= 2 and web_site in end_site_list:
        video_duration = video_duration - 3

    # 视频
    video = Video()
    video.imgs_to_video(use_img_path_list, video_path, video_duration, (2160, 3840))  # 图片转视频
    video.trans_ecfect(video_path, len(use_img_path_list), video_duration)  # 添加随机转场动画

    # 拼接片尾
    if len(img_path_list) <= 2 and web_site in end_site_list:
        end_video_site_path = os.path.join(end_video_dir, f"{web_site}.mp4")
        # 加上片尾视频路径
        split_video_path = os.path.split(video_path)  # 完整音视频路径
        end_video_path = os.path.join(split_video_path[0], "end-" + split_video_path[-1])

        video.merge_videos(end_video_path, [video_path, end_video_site_path])
        logger.success(f"[]>> 视频添加片尾 {web_site} - {end_video_path}")

        os.remove(video_path)
        os.rename(end_video_path, video_path)

    # 合并音视频
    split_video_path = os.path.split(video_path)  # 完整音视频路径
    av_path = os.path.join(split_video_path[0], "av-" + split_video_path[-1])
    video.merge_av(av_path, audio_path, video_path)

    # 添加字幕
    # split_av_path = os.path.split(av_path)  # 完整音视频+字幕路径
    # avs_path = os.path.join(split_av_path[0], "avs" + split_av_path[-1][2:])
    # video.merge_srt_video(av_path, srt_path, avs_path)

    # 拼接片头
    # logo_video_path = os.path.join(videos_dir, "logo.mp4")
    # final_video_path = "final" + avs_path[3:]  # 最终视频路径
    # video.merge_videos(final_video_path, [logo_video_path, avs_path])
    # logger.success(f"[完成]>> 视频制作 - {final_video_path}")


if __name__ == '__main__':
    # generateAV("veswin", "1n4752a")

    hta = HtAudio(ht_api_key, ht_secret_key)
    text = "The AD620ANZ is a precision instrumentation amplifier IC commonly used in weigh scales, medical devices, and industrial control systems for accurate and reliable measurement applications."
    file_path = r"GenerateVideo\assets\audios\Avaq_ad620anz\ad620anz.mp3"
    hta.gen_audio(text, file_path)

    ...
