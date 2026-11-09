BUSAN CRUISE PORTFOLIO DEMO / GitHub 적용 안내
===============================================

기존 Streamlit Cloud의 Main file path가 dash2.py인 경우:
1. 이 압축파일을 해제합니다.
2. dash2.py를 기존 프로젝트의 dash2.py에 덮어씁니다.
3. assets/demo_raw.csv 와 assets/demo_schedule.csv 를 기존 프로젝트의 assets/ 폴더에 복사합니다.
4. .gitignore 도 교체하여 위 두 데모 CSV가 GitHub에 올라가도록 합니다.
5. (선택) assets/workflow_demo.mp4, assets/workflow_preview.png, assets/make_video.py도 교체합니다.
6. (선택) 원본 프로젝트 진입파일이 app.py/dash1.py라면 압축파일의 해당 파일도 교체합니다.

PowerShell 터미널 (프로젝트 폴더 위치에서):
  git status
  git add dash2.py app.py dash1.py .gitignore assets/demo_raw.csv assets/demo_schedule.csv assets/workflow_demo.mp4 assets/workflow_preview.png assets/make_video.py
  git commit -m "Add portfolio demo mode with synthetic data"
  git push origin main

GitHub에서 수정한 내용과 충돌하는 경우 git push --force를 사용하지 마세요.

작동 방식:
- 기본 모드: 예시 데이터로 체험. CSV를 따로 업로드하지 않아도 2027년 결과 표시.
- 직접 CSV 분석: 기존의 파일 업로드·완료월 확인·검증·ETS 재학습 기능.
- 데모의 선박명·입출항 시각·승객수·예측치는 완전한 가상 시뮬레이션 수치입니다.
  실제 부산항 데이터나 2027년 실제 예정 스케줄을 공유한 것이 아닙니다.
- 데모의 예측 기록 JSON 및 실적 평가 기능은 오인 방지를 위해 숨깁니다.
- 2023.01~2026.09 학습, 2027년 전망, 입항수·승객수 모두 ETS(A,Ad,A).

주의: 실제 원자료(i.csv 등)는 ZIP에 포함되어 있지 않습니다.
