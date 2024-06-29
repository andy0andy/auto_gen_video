import queue
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
from PIL import Image
import pyodbc
import pandas as pd
import os
import requests
from loguru import logger
import shutil
import json
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

"""
任务前的先觉准备
例如图片下载等
"""


server = "192.168.1.255"
user = "sa"
password = "123"
database = "DBName"
driver = "ODBC Driver 17 for SQL Server"
vaeConn = pyodbc.connect(driver=driver, server=server, user=user, password=password, database=database)
vaeCursor = vaeConn.cursor()

root_path = r"GenerateVideo\assets"

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

# ==============================

def download_image(link, file_path):
    headers = {
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
    }
    resp = requests.get(link, headers=headers, verify=False, timeout=10)
    with open(file_path, "wb") as f:
        f.write(resp.content)

def familylist():
    familylist = {
        "FamilyName": [],
        "ProductName": [],
        "WebSite": [],
        "VideoName": [],
    }

    sql = "SELECT * FROM [Vae].[dbo].[FamilyList];"
    vaeCursor.execute(sql)
    rows = vaeCursor.fetchall()
    for row in rows:
        if not row[-1]:
            continue
        # print(row)

        n_pn = trans_pn(row[2]).lower()
        # if os.listdir(os.path.join(root_path, os.path.join("imgs", n_pn))):
        #     continue

        # 下载图片
        img_list = row[-1].split(",")
        if len(img_list) < 2:
            # shutil.rmtree(os.path.join(root_path, os.path.join("imgs", n_pn)))
            continue
        logger.info(f"[]>> {row[3]} ({len(img_list)}) - {row[2]}")

        if not os.path.exists(os.path.join(root_path, os.path.join("imgs", n_pn))):
            os.mkdir(os.path.join(root_path, os.path.join("imgs", n_pn)))

        familylist["FamilyName"].append(row[1])
        familylist["ProductName"].append(row[2])
        familylist["WebSite"].append(row[3])
        familylist["VideoName"].append("av-" + n_pn + ".mp4")

        for idx, img_link in enumerate(img_list):
            if not img_link.strip().startswith("/"):
                continue

            img_link = f"https://www.{row[3].lower()}.com" + img_link.strip()

            file_path = os.path.join(root_path, os.path.join("imgs", os.path.join(n_pn, n_pn + f"_{idx}." + img_link.split(".")[-1])))
            download_image(img_link, file_path)
            logger.info(f"[]>> ({idx}) {n_pn} 下载图片 {img_link} -> {file_path}")

        # break

    df = pd.DataFrame(familylist)
    df.to_csv(os.path.join(root_path, "upload_video.csv"), index=False)

def familylist_handle():
    """
    处理完善表格

    :return:
    """

    file_path = os.path.join(root_path, "upload_video.csv")
    new_file_path = os.path.join(root_path, "new_upload_video.csv")

    df = pd.read_csv(file_path)

    seriesMap = {}

    family_names = list(set(list(df["FamilyName"])))
    for family_name in family_names:
        sql = f"SELECT * FROM [Vae].[dbo].[FamilyList] where FamilyName ='{family_name}';"
        vaeCursor.execute(sql)
        rows = vaeCursor.fetchall()

        for row in rows:
            seriesMap[row[1]] = seriesMap.get(row[1], [])
            seriesMap[row[1]].append(row[2])

    relate_product = {}
    df_dict = df.to_dict()
    for i in range(len(df_dict['ProductName'])):
        family_name = df_dict["FamilyName"][i]
        product_name = df_dict["ProductName"][i]

        rp_str = ", ".join([series for series in seriesMap[family_name] if series != product_name])
        rp_str = rp_str.replace("[", "").replace("]", "")
        relate_product[str(i)] = rp_str

    df_dict["RelateProduct"] = relate_product

    for k, v in df_dict.items():

        lst = []
        for _, _v in v.items():
            lst.append(_v)
        df_dict[k] = lst

    ndf = pd.DataFrame(df_dict)
    ndf.sort_values(by=["WebSite"], inplace=True, ascending=False)
    ndf.to_csv(new_file_path, index=False)

def handle_video_split():
    file_path = os.path.join(root_path, "upload_video.csv")
    df = pd.read_csv(file_path)

    for row in df.iterrows():
        web_iste = row[1]["WebSite"]
        video_name = row[1]["VideoName"]

        old_path = os.path.join(root_path, f"final_videos/{video_name}")
        new_path = os.path.join(root_path, f"final_videos/{web_iste}/{video_name}")
        shutil.move(old_path, new_path)
        logger.info(f"[]>> {old_path} -> {new_path}")

        # break


# ===========================
def handle_product_weik():

    def work(q):

        while not q.empty():
            [product, target, Image] = q.get()

            # 下载
            n_pn = trans_pn(product).lower()
            img_dir = os.path.join(root_path, os.path.join("imgs", f"{target.lower()}_" + n_pn))
            if not os.path.exists(img_dir):
                os.mkdir(img_dir)

            for i, im in enumerate(Image):
                im = f"https://www.{target.lower()}.com{im}"
                img_path = os.path.join(img_dir, f"{n_pn}_{i}.{im.split('.')[-1]}")
                if os.path.exists(img_path):
                    continue

                logger.info(f"[]>> download pic: {im} -> {img_path}")
                download_image(im, img_path)


    q = queue.Queue()
    pool_size = 10
    pool = ThreadPoolExecutor(max_workers=pool_size)

    datas = {
        "WebSite": [],
        "Product": [],
        "VideoName": [],
        "RelateProducts": [],
    }

    sql = "SELECT * FROM [Vae].[dbo].[Product_weik] where Target not in ('Veswin', 'Veswin1', 'Veswin2', 'Censtry') and RelatedProduct <> '';"
    vaeCursor.execute(sql)
    rows = vaeCursor.fetchall()
    for row in rows:
        product_param = json.loads(row[1])
        if len(product_param["Image"]) == 0:
            continue

        relate_product = json.loads(row[2])
        target = row[3]
        if target == "Veswin3":
            target = "Veswin"
        elif target == "Censtry_1":
            target = "Censtry"

        product = product_param["Product"].strip()
        relate_product = ", ".join(relate_product)

        datas["WebSite"].append(target)
        datas["Product"].append(product)
        datas["VideoName"].append(trans_pn(product).lower() + ".mp4")
        datas["RelateProducts"].append(relate_product)
        logger.info(f"[]>> {target} {product}")
        # break

        image = product_param["Image"]
        if isinstance(image, str):
            image = [image]

        q.put([product, target, image])

    all_tasks = []
    for i in range(pool_size):
        all_tasks.append(pool.submit(work, q))
    wait(all_tasks, return_when=ALL_COMPLETED)

    # csv_file_path = os.path.join(root_path, "product.csv")
    # df = pd.DataFrame(datas)
    # df.sort_values(by=["WebSite"], inplace=True)
    # df.to_csv(csv_file_path, index=False)
    # logger.success(f"[]>> 生成文件 {csv_file_path}")

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

def handle_imgs():
    """
    处理异常图片,或空图片文件夹
    :return:
    """

    imgs_dir = os.path.join(root_path, "imgs")
    for file_name in os.listdir(imgs_dir):
        file_path = os.path.join(imgs_dir, file_name)
        pic_list = os.listdir(file_path)

        # 删除空文件夹
        if len(pic_list) == 0:
            shutil.rmtree(file_path)
            logger.info(f"[]>> 删除空文件夹 {file_path}")

        # 删除异常图片文件夹
        for pic_name in pic_list:
            pic_path = os.path.join(file_path, pic_name)
            if not check_image(pic_path):
                shutil.rmtree(file_path)
                logger.info(f"[]>> 删除异常图片文件夹 {file_path}")
                break

def extract_videos():

    final_videos_path = os.path.join(root_path, "final_videos")
    videos_path = os.path.join(root_path, "videos")

    for video_dir in os.listdir(videos_path):
        web_site = video_dir.split('_')[0]
        if not os.path.isdir(os.path.join(final_videos_path, web_site)):
            os.mkdir(os.path.join(final_videos_path, web_site))

        file_name = '_'.join(video_dir.split('_')[1:]) + ".mp4"
        src_file_path = os.path.join(videos_path, os.path.join(video_dir, "av-" + file_name))
        tar_file_path = os.path.join(final_videos_path, os.path.join(web_site, file_name))

        shutil.copy(src_file_path, tar_file_path)
        logger.info(f"[]>> {src_file_path} -> {tar_file_path}")






if __name__ == '__main__':

    # familylist()
    # familylist_handle()
    # handle_video_split()
    # ==============

    # handle_product_weik()
    # handle_imgs()
    # extract_videos()





    ...

