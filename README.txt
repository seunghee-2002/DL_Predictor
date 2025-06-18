프로젝트 디렉토리/
│
├── config.py                 # 설정값을 dataclass로 관리
├── main.py                  # 진입점, config 로딩 + 모델/데이터 로딩 + 학습 호출
├── trainer.py               # train_model, evaluate_model 함수
├── utils.py                 # save_model, load_model, metrics 계산 등 유틸 모음
│
├── model_cnn.py             # CNNPredictorModel 클래스
├── model_attention.py       # AttentionPredictorModel 클래스
│
├── custom_dataset.py        # OrderSequenceDataset + collate_fn
│
├── DLResource/              # 데이터 CSV, 임베딩 numpy 저장 위치
└── models/                  # 학습된 모델 저장 디렉토리
└── xgb/ 			# XGBoost baseline ML model 관련 디렉토리

