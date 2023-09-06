import os
import pandas as pd

# 폴더 경로와 CSV 파일 경로를 설정합니다.
folder_path = 'test'  # 폴더 경로를 적절히 수정하세요.
csv_file_path = 'ground_truth_Vacuum_section_00_test.csv'  # CSV 파일 경로를 적절히 수정하세요.

# CSV 파일을 읽어옵니다.
df = pd.read_csv(csv_file_path)

# 파일 이름 변경 작업을 수행합니다.
for index, row in df.iterrows():
    origin_name = row['origin']
    new_name = row['new']
    
    # 파일의 현재 경로와 새로운 경로를 생성합니다.
    old_file_path = os.path.join(folder_path, origin_name)
    new_file_path = os.path.join(folder_path, new_name+'.wav')
    # 파일 이름을 변경합니다.
    try:
        os.rename(old_file_path, new_file_path)
        print(f'파일 이름 변경 완료: {origin_name} -> {new_name}.wav')
    except FileNotFoundError:
        print(f'파일을 찾을 수 없습니다: {origin_name}')
    except FileExistsError:
        print(f'새로운 파일 이름이 이미 존재합니다: {new_name}')

print('작업 완료')