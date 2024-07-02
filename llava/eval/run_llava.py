import argparse
import torch

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


# 모델 경로
model_path = "liuhaotian/llava-v1.5-7b"

# 모델 초기화 (한 번만 호출)
disable_torch_init()
model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path, None, model_name
)

prompt = "What tags can we add to this movie?"

def eval_model(args):
    # # Model
    # disable_torch_init()

    
    # model_name = get_model_name_from_path(args.model_path)
    # tokenizer, model, image_processor, context_len = load_pretrained_model(
    #     args.model_path, args.model_base, model_name
    # )
    
    
    qs = args.query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    image_files = image_parser(args)
    images = load_images(image_files)
    image_sizes = [x.size for x in images]
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )
    
    torch.cuda.empty_cache()

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    # print(outputs)
    return(outputs)


# if __name__ == "__main__":
    
#     model_path = "liuhaotian/llava-v1.5-7b"
    
#     # 여기에 프롬프트 작성
#     prompt = "What tags can be guessed with the movie scene?"
#     # 이미지 파일 경로
#     image_file = "https://llava-vl.github.io/static/images/view.jpg"
    
#     # 모델 파라미터 수정
#     args = type('Args', (), {
#         "model_path": model_path,
#         "model_base": None,
#         "model_name": get_model_name_from_path(model_path),
#         "query": prompt,
#         "conv_mode": None,
#         "image_file": image_file,
#         "sep": ",",
#         "temperature": 0,
#         "top_p": None,
#         "num_beams": 1,
#         "max_new_tokens": 512
#     })()
    
#     #
    
#     eval_model(args)


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


model_path = "liuhaotian/llava-v1.5-7b"

# 여기에 프롬프트 작성
prompt = "What tags can we add to this movie?"

def analyze_image_with_LLaVA(image_path):
    
    # 이미지 파일 경로
    # image_file = "https://llava-vl.github.io/static/images/view.jpg"
    
    # 모델 파라미터 수정
    args = type('Args', (), {
        "model_path": model_path,
        "model_base": None,
        "model_name": get_model_name_from_path(model_path),
        "query": prompt,
        "conv_mode": None,
        "image_file": image_path,
        "sep": ",",
        "temperature": 0,
        "top_p": None,
        "num_beams": 1,
        "max_new_tokens": 512
    })()
    
    #
    
    content = eval_model(args)

    
    print(content)
    
    return(content)

def load_unique_tags(csv_path):
    # CSV 파일에서 unique_tags 열을 읽어 소문자로 변환하여 집합으로 반환
    df = pd.read_csv(csv_path)
    unique_tags = set(df['unique_tags'].str.lower().str.split(',').sum())
    return unique_tags

def find_matching_tags(generated_tags, unique_tags):
    # 결과에서 태그를 추출하여 소문자로 변환
    generated_tags_lower = generated_tags.lower()
    
    # 일치하는 태그 찾기
    matching_tags = []
    for tag in unique_tags:
        if tag.lower() in generated_tags_lower:
            matching_tags.append(tag)
    
    # print(unique_tags)
    # print(generated_tags)
    # print(list(matching_tags))
    
    return matching_tags

def analyze_images_and_save_results(base_directory, output_csv):
    unique_tags = load_unique_tags('unique_tags.csv')
    results = []

    for movie_dir in os.listdir(base_directory):
        movie_path = os.path.join(base_directory, str(movie_dir))
        if os.path.isdir(movie_path):
            tag_counts = {}
            image_count = 0
            for filename in os.listdir(movie_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(movie_path, filename)
                    image_count += 1
                    
                    # analysis_output = analyze_image_with_LLaVA(image_path)
                    # analysis_output = 'Action, Thriller, Crime, Mystery, Drama, Romance, Comedy, Horror, Sci-fi, Fantasy, Adventure, Western, War'
                    
                    try:
                        analysis_output = analyze_image_with_LLaVA(image_path)
                    except Exception as e:
                        print(f"Error analyzing image {image_path}: {e}")
                        continue
                    
                    matching_tags = find_matching_tags(analysis_output, unique_tags)
                    
                    # 태그 카운트 업데이트
                    for tag in matching_tags:
                        if tag in tag_counts:
                            tag_counts[tag] += 1
                        else:
                            tag_counts[tag] = 1
            
            # 태그 비율 계산 및 요약 생성
            tag_summary = ', '.join([f"{tag}({count}, {count/image_count*100:.2f}%)" for tag, count in tag_counts.items()])
            results.append({"movie": str(movie_dir), "tags": tag_summary, "image_count": image_count})
 
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")
    
    # 태그 등장 횟수 추가
    tag_counts_df = pd.DataFrame(list(tag_counts.items()), columns=['tag', 'count'])
    
    # # 결과를 하나의 CSV 파일로 저장
    # results_df.to_csv(output_csv, index=False)
    # tag_counts_df.to_csv(output_csv.replace('.csv', '_tag_counts.csv'), index=False)
    
    # print(f"Results saved to {output_csv}")
    # print(f"Tag counts saved to {output_csv.replace('.csv', '_tag_counts.csv')}")

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

# # 'mpst_full_data.csv' 파일에서 상위 100개 항목의 'title' 피처를 읽어옴
# df = pd.read_csv('mpst_full_data.csv')
# search_queries = df['title'].head(100).tolist()

# # 101~200개 항목
# search_queries = df['title'].iloc[100:200].tolist()

# 'ml-25m/movies.csv' 파일에서 상위 100개 항목의 'title' 피처를 읽어옴
df = pd.read_csv('./ml-25m/movies.csv')
search_queries = df['title'].head(100).tolist()

base_directory = 'videos'

# # 검색어를 통한 유튜브 영상 데이터 크롤링 및 다운로드
# crawl_start_time = time.time()

# for query in search_queries:
#     movie_dir = os.path.join(base_directory, str(query))
#     results = crawl_youtube_videos(str(query), max_results=5)
#     for result in results:
#         download_youtube_video(result['url'], movie_dir)

# crawl_end_time = time.time()

# # 각 영화 디렉토리에 있는 비디오 프레임 추출
# for query in search_queries:
#     movie_dir = os.path.join(base_directory, str(query))
    
#     # 디렉토리 존재하지 않는 경우 (상위 5개 영상 모두 사용 불가능한 경우)
#     if not os.path.exists(movie_dir):
#         print(f"디렉토리가 존재하지 않습니다: {movie_dir}")
#         continue
    
#     video_files = [f for f in os.listdir(movie_dir) if f.endswith('.mp4')]
#     for video_file in video_files:
#         video_path = os.path.join(movie_dir, video_file)
#         save_frames(video_path, 640, 360, movie_dir)

# 이미지 분석 및 결과 저장
analyze_start_time = time.time()
analyze_images_and_save_results(base_directory, 'movie_tags_results.csv')
analyze_end_time = time.time()



# print(f"크롤링 시간: {crawl_end_time - crawl_start_time}초")
print(f"이미지 분석 및 저장 시간: {analyze_end_time - analyze_start_time}초")
