import pandas as pd

# CSV 파일 경로
file_path = 'mpst_full_data.csv'  # 실제 파일 경로로 변경하세요.

# CSV 파일 읽기
data = pd.read_csv(file_path)

# 태그 개수 계산
data['tag_count'] = data['tags'].apply(lambda x: len(x.split(',')))

# 전체 tag_count의 합과 평균 계산
total_tag_count = data['tag_count'].sum()
average_tag_count = data['tag_count'].mean()

# 결과 출력
print(f"Total tag count: {total_tag_count}")
print(f"Average tag count: {average_tag_count:.2f}")

# 각 영화의 tag_count 출력
print(data[['imdb_id', 'title', 'tag_count']])
