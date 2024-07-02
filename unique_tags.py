import pandas as pd

# 데이터 불러오기
data = pd.read_csv("mpst_full_data.csv")

# "tags" 열에서 모든 태그 추출
all_tags = data['tags'].str.split(',').explode().str.strip()

# 중복 제거하여 고유한 태그 리스트 생성
unique_tags = all_tags.unique().tolist()

# 고유한 태그를 DataFrame으로 변환
unique_tags_df = pd.DataFrame({'unique_tags': unique_tags})

# CSV 파일로 저장
unique_tags_df.to_csv("unique_tags.csv", index=False)
