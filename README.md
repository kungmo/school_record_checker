# 📑 생기부 검토 도우미 (School Record Checker)

AI(Google Gemini)를 활용하여 고등학교 학교생활기록부(세특, 행특, 창체, 동아리 등)의 오타, 맞춤법, 그리고 기재 요령 위반 사항을 자동으로 검토해주는 도구입니다.

> **⚠️ 개인정보 보호 안내 (Privacy Notice)**
> 1. 본 프로그램은 사용자가 입력한 **Google API Key**와 **생기부 데이터**를 서버에 별도로 저장하지 않습니다.
> 2. 모든 데이터 처리는 메모리상에서 일시적으로 이루어지며, 브라우저 종료 시 세션이 만료됩니다.
> 3. 다만, 외부 AI 모델(Google Gemini)을 사용하므로, 데이터가 모델 학습에 활용되지 않도록 하는 설정이나 API 이용 약관은 Google의 정책을 따릅니다. 
> 4. **민감한 정보가 포함된 경우 성명 등은 가명 처리 후 업로드하는 것을 권장**하지만, 이 웹 앱 자체에서는 이러한 정보를 저장하는 기능을 구현하지 않았습니다.

---

## ✨ 주요 기능

- **NEIS 엑셀 파일 호환**: NEIS에서 다운로드한 다양한 양식의 엑셀 파일을 그대로 업로드하여 일괄 점검 가능 (2025년 12월 기준)
- **영역별 정밀 검토**: 
  - 과목별 세부능력 및 특기사항 (과세특)
  - 행동특성 및 종합의견 (행특)
  - 창의적 체험활동 (자율, 진로)
  - 동아리 활동 특기사항
- **기재 요령 준수 체크**: '대회', '강사명', '기관명', '논문' 등 생기부 기재 금지어 및 부적절한 표현 감지
- **AI 기반 피드백**: 동료 교사가 검토해주는 듯한 자연스럽고 정확한 수정 의견 제공

## 🛠 기술 스택

- **Language**: Python 3.12+
- **Framework**: Chainlit, LangChain
- **AI Model**: Google Gemini 2.5 Flash
- **Data Analysis**: Pandas, Openpyxl

## 🚀 설치 및 실행 방법

### 1. 저장소 복제 및 의존성 설치
```bash
git clone [https://github.com/your-username/school-record-checker.git](https://github.com/your-username/school-record-checker.git)
cd school-record-checker
uv pip install -r requirements.txt
```

### 2. 웹 앱 실행
```
chainlit run school_record_checker.py --host 0.0.0.0 --port 8501
```

### 3. 환경 설정

#### .env 파일 관련 설정:

`.env` 파일 생성 후 설정:
```
# Google Gemini API
GOOGLE_API_KEY=your_api_key

# MariaDB 설정
# Google Gemini API
GOOGLE_API_KEY=your_api_key

# MariaDB 설정 (코드와 일치시킴)
DB_HOST=localhost
DB_USER=your_username
DB_PASSWORD=your_password
DB_NAME=your_DB_name
DB_PORT=3306
```

#### MariaDB 설정 SQL 스크립트:
```
CREATE TABLE IF NOT EXISTS school_record_logs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(255) NOT NULL,
    subject VARCHAR(100) NOT NULL,
    success TINYINT(1) NOT NULL,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP -- 자동으로 시간 입력됨
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

## 실제 구동 웹 앱 주소
http://kungmo2.mooo.com:8501

## 라이선스

듀얼 라이선스 (MIT for non-commercial / Commercial License)

LICENSE 파일의 내용을 참고하십시오.

### 📚 비상업적 사용 (무료)
다음 사용자는 이 프로젝트를 자유롭게 사용하실 수 있습니다:
- 개인 사용자
- 초중고 및 대학 등 교육기관
- 교육청, 정부기관 등 공공기관
- 비영리 단체

**MIT 라이선스** 조건으로 사용, 수정, 배포가 가능합니다.

### 🏢 상업적 사용 (허가 필요)
영리 목적의 기업이나 조직에서 사용하시려면 별도 계약이 필요합니다.

📧 문의: kungmo@snu.ac.kr

자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 기여하기
버그 리포트, 기능 제안, Pull Request 환영합니다!