import argparse
import torch
from pytube. innertube import _default_clients


_default_clients[ "ANDROID"][ "context"]["client"]["clientVersion"] = "19.08.35"
_default_clients["IOS"]["context"]["client"]["clientVersion"] = "19.08.35"
_default_clients[ "ANDROID_EMBED"][ "context"][ "client"]["clientVersion"] = "19.08.35"
_default_clients[ "IOS_EMBED"][ "context"]["client"]["clientVersion"] = "19.08.35"
_default_clients["IOS_MUSIC"][ "context"]["client"]["clientVersion"] = "6.41"
_default_clients[ "ANDROID_MUSIC"] = _default_clients[ "ANDROID_CREATOR" ]

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)


from PIL import Image

import requests
from PIL import Image
from io import BytesIO
import re

import os
import time
import datetime
import cv2 as cv
import pandas as pd
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from pytube import YouTube
from pytube.exceptions import VideoUnavailable
import base64
import requests

def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_image(image_file):
    try:
        if image_file.startswith("http") or image_file.startswith("https"):
            response = requests.get(image_file)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            image = Image.open(image_file).convert("RGB")
        return image
    except FileNotFoundError:
        print(f"File not found: {image_file}")
        return None
    except Exception as e:
        print(f"Error loading image {image_file}: {e}")
        return None


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        if image is not None:
            out.append(image)
    return out

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def crawl_youtube_videos(search_query, max_results=5):
    # 크롬 드라이버 초기화
    service = Service(ChromeDriverManager().install())
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    
    driver = webdriver.Chrome(service=service, options=options)
    # driver = webdriver.Chrome(service=service)
    
    try:
        # YouTube에 접속
        driver.get("https://www.youtube.com")
        time.sleep(2) # 페이지 로딩 대기

        # 검색어 입력 및 검색 실행
        search_query_with_movie = str(search_query) + ' movie'
        search_box = driver.find_element("name", "search_query")
        search_box.send_keys(search_query_with_movie)
        search_box.send_keys(Keys.RETURN)
        time.sleep(2) # 검색 결과 로딩 대기

        # 검색 결과에서 비디오 타이틀과 URL 수집
        videos = driver.find_elements("id", "video-title")
        results = []
        for video in videos[:max_results]:
            title = video.text
            url = video.get_attribute('href')
            results.append({"title": title, "url": url})

        print(results)
        
        return results

    finally:
        driver.quit()

def download_youtube_video(videourl, path):
    try:
        path = str(path)
        print("다운로드 경로:", path)
        
        yt = YouTube(videourl)
        print("유튜브 객체 생성 완료")
        # None인 경우 (재생목록인 경우)
        
        # 화질 가장 낮은 것 선택
        yt_stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').asc().first()
        print("스트림 선택 완료:", yt_stream)
        
        if not os.path.exists(path):
            os.makedirs(path)
            print("경로 생성 완료:", path)
        
        yt_stream.download(path)
        print("다운로드 완료")
        
    except VideoUnavailable:
        print("비디오 접근할 수 없음")
    except Exception as e:
        print("오류 발생:", e)

def save_frames(video_path, target_width, target_height, output_dir, max_frames=20):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    vidcap = cv.VideoCapture(video_path)
    fps = vidcap.get(cv.CAP_PROP_FPS)
    total_frames = int(vidcap.get(cv.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    # 프레임 간격 설정 (기본적으로 10초 간격, 너무 긴 경우 프레임 수를 제한)
    # 영상이 너무 긴 경우 max_frames 수 만큼 일정하게 frame 나눔
    frame_interval = max(10, duration / max_frames)

    count = 0
    second = 0
    saved_frames = 0

    while True:
        success, image = vidcap.read()
        if not success or saved_frames >= max_frames:
            break

        current_time = count / fps

        if current_time >= second:
            # 풀링된 이미지 생성
            pooled_image = resize_and_pool(image, target_width, target_height)

            # 이미지 파일 이름 생성
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
            frame_filename = f"frame_at_{int(current_time)}_seconds_{timestamp}.jpg"
            # 파일 경로 생성
            image_path = os.path.join(output_dir, frame_filename)
            cv.imwrite(image_path, pooled_image)  # 현재 프레임 저장
            print(f"Saved pooled frame at {int(current_time)} seconds at resolution {target_width}x{target_height}")
            second += frame_interval
            saved_frames += 1

        count += 1

    vidcap.release()
    
    # 영상 파일 삭제
    os.remove(video_path)
    print(f"Deleted video file: {video_path}")


def resize_and_pool(image, target_width, target_height, num_layers=2):
    # 원본 이미지의 비율 계산
    original_height, original_width = image.shape[:2]
    ratio = original_height / original_width

    # 새로운 크기 계산
    if ratio < target_height / target_width:
        new_width = target_width
        new_height = int(new_width * ratio)
    else:
        new_height = target_height
        new_width = int(new_height / ratio)

    # 리사이징
    resized_image = cv.resize(image, (new_width, new_height), interpolation=cv.INTER_AREA)

    # 여러 단계의 풀링 적용
    pooled_image = multi_pooling(resized_image, target_width, target_height, num_layers)

    return pooled_image

def multi_pooling(image, target_width, target_height, num_layers=2):
    # 원본 이미지의 비율 계산
    original_height, original_width = image.shape[:2]
    ratio = original_height / original_width

    # 새로운 크기 계산
    if ratio < target_height / target_width:
        new_width = target_width
        new_height = int(new_width * ratio)
    else:
        new_height = target_height
        new_width = int(new_height / ratio)

    # 초기 이미지 설정
    pooled_image = image

    # 여러 단계의 풀링 적용
    for _ in range(num_layers):
        # 풀링 크기 계산
        pool_height = max(1, new_height // target_height)
        pool_width = max(1, new_width // target_width)

        # 새로운 크기 계산
        new_height //= pool_height
        new_width //= pool_width

        # 풀링 적용
        pooled_image = np.zeros((new_height, new_width, image.shape[2]), dtype=np.uint8)
        for y in range(new_height):
            for x in range(new_width):
                region = image[y * pool_height:(y + 1) * pool_height, x * pool_width:(x + 1) * pool_width]
                pooled_image[y, x] = np.mean(region, axis=(0, 1)).astype(np.uint8)

    return pooled_image

# 'mpst_full_data.csv' 파일에서 상위 100개 항목의 'title' 피처를 읽어옴
df = pd.read_csv('mpst_full_data.csv')
# search_queries = df['title'].head(100).tolist()

# 파인튜닝용 임시 list
search_queries = [
    "This Means War", 
    "Star Wars: Episode V - The Empire Strikes Back", 
    "Four Christmases", 
    "17 Again", 
    "Santa Fe Trail", 
    "Natural Born Killers", 
    "Crimson Tide", 
    "Reservoir Dogs", 
    "In a Lonely Place", 
    "Teenage Mutant Ninja Turtles: Out of the Shadows", 
    "Miller's Crossing", 
    "Mission: Impossible III", 
    "Paul Blart: Mall Cop", 
    "Confessions of a Shopaholic", 
    "Along Came a Spider", 
    "Brokeback Mountain", 
    "Big Fat Liar", 
    "Clash of the Titans", 
    "Broken Blossoms or The Yellow Man and the Girl", 
    "Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb"
]

# # 101~200개 항목
# search_queries = df['title'].iloc[100:200].tolist()

base_directory = 'videos'

# 검색어를 통한 유튜브 영상 데이터 크롤링 및 다운로드
crawl_start_time = time.time()

for query in search_queries:
    movie_dir = os.path.join(base_directory, str(query))
    results = crawl_youtube_videos(str(query), max_results=5)
    for result in results:
        download_youtube_video(result['url'], movie_dir)

crawl_end_time = time.time()

# 각 영화 디렉토리에 있는 비디오 프레임 추출
for query in search_queries:
    movie_dir = os.path.join(base_directory, str(query))
    
    # 디렉토리 존재하지 않는 경우 (상위 5개 영상 모두 사용 불가능한 경우)
    if not os.path.exists(movie_dir):
        print(f"디렉토리가 존재하지 않습니다: {movie_dir}")
        continue
    
    video_files = [f for f in os.listdir(movie_dir) if f.endswith('.mp4')]
    for video_file in video_files:
        video_path = os.path.join(movie_dir, video_file)
        save_frames(video_path, 640, 360, movie_dir)

# # 이미지 분석 및 결과 저장
# analyze_start_time = time.time()
# analyze_images_and_save_results(base_directory, 'movie_tags_results2.csv')
# analyze_end_time = time.time()


# print(f"크롤링 시간: {crawl_end_time - crawl_start_time}초")
# print(f"이미지 분석 및 저장 시간: {analyze_end_time - analyze_start_time}초")
