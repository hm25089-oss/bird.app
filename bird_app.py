from pathlib import Path
import io
import json

import joblib
import librosa
import numpy as np
import streamlit as st

from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
from streamlit.runtime.scriptrunner import get_script_run_ctx


# =========================================================
# 1. 설정
# =========================================================

# 바탕화면의 "새 음향파일" 폴더를 자동으로 찾습니다.
# 찾지 못하면 None 대신 실제 경로를 입력하세요.
#
# 예:
# AUDIO_DIR = Path(r"C:\Users\ted56\Desktop\새 음향파일")
#
AUDIO_DIR = None

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "bird_model"

MODEL_PATH = MODEL_DIR / "bird_model.keras"
SCALER_PATH = MODEL_DIR / "scaler.joblib"
ENCODER_PATH = MODEL_DIR / "label_encoder.joblib"
METADATA_PATH = MODEL_DIR / "metadata.json"

CATEGORIES = [
    "큰부리까마귀+송장까마귀",
    "참새",
    "뻐꾸기",
    "배경소리",
    "물까치",
    "멧비둘기",
    "까치",
]

BACKGROUND = "배경소리"

# 호환성이 좋은 음원 형식을 사용합니다.
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg"}

SAMPLE_RATE = 44100
N_MFCC = 40
EPOCHS = 50
BATCH_SIZE = 4
RANDOM_STATE = 42


# =========================================================
# 2. 학습 폴더 찾기
# =========================================================

def find_audio_dir():
    if AUDIO_DIR is not None:
        folder = Path(AUDIO_DIR)

        if not folder.is_dir():
            raise FileNotFoundError(
                f"지정한 폴더가 없습니다:\n{folder}"
            )

        return folder

    home = Path.home()

    candidates = [
        home / "Desktop" / "새 음향파일",
        home / "바탕 화면" / "새 음향파일",
        home / "OneDrive" / "Desktop" / "새 음향파일",
        home / "OneDrive" / "바탕 화면" / "새 음향파일",
    ]

    for folder in candidates:
        if folder.is_dir():
            return folder

    raise FileNotFoundError(
        "바탕화면의 '새 음향파일' 폴더를 찾지 못했습니다.\n"
        "코드 위쪽 AUDIO_DIR에 실제 경로를 입력하세요."
    )


# =========================================================
# 3. MFCC 평균값 40개 추출
# =========================================================

def extract_mfcc(source):
    audio, _ = librosa.load(
        source,
        sr=SAMPLE_RATE,
        mono=True,
        duration=None,
    )

    if audio.size == 0:
        raise ValueError("소리 데이터가 없는 파일입니다.")

    if not np.all(np.isfinite(audio)):
        raise ValueError("음원에 올바르지 않은 값이 있습니다.")

    if np.max(np.abs(audio)) <= 1e-8:
        raise ValueError("무음 파일입니다.")

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
    )

    feature = np.mean(mfcc, axis=1).astype(np.float32)

    if not np.all(np.isfinite(feature)):
        raise ValueError("MFCC 계산에 실패했습니다.")

    return feature


# =========================================================
# 4. 학습 자료 읽기
# =========================================================

def load_dataset(audio_dir):
    features = []
    labels = []
    counts = {}
    failures = []

    print(f"\n학습 폴더: {audio_dir}\n")

    for folder in audio_dir.iterdir():
        if folder.is_dir() and folder.name not in CATEGORIES:
            print(f"[제외] 등록되지 않은 폴더: {folder.name}")

    for category in CATEGORIES:
        folder = audio_dir / category

        if not folder.is_dir():
            print(f"[건너뜀] 폴더 없음: {category}")
            continue

        files = sorted(
            path for path in folder.iterdir()
            if path.is_file()
            and path.suffix.lower() in AUDIO_EXTENSIONS
        )

        count = 0

        for file_path in files:
            try:
                feature = extract_mfcc(file_path)

                features.append(feature)
                labels.append(category)
                count += 1

                print(f"[읽음] {category} / {file_path.name}")

            except Exception as exc:
                message = f"{category}/{file_path.name}: {exc}"
                failures.append(message)
                print(f"[실패] {message}")

        if count > 0:
            counts[category] = count
        else:
            print(f"[제외] 읽을 수 있는 음원이 없음: {category}")

    if BACKGROUND not in counts:
        raise RuntimeError(
            "'배경소리' 폴더에 읽을 수 있는 음원이 필요합니다."
        )

    if len(counts) < 2:
        raise RuntimeError(
            "배경소리와 최소 한 종류의 새 음원이 필요합니다."
        )

    x = np.vstack(features).astype(np.float32)
    labels = np.asarray(labels)

    return x, labels, counts, failures


# =========================================================
# 5. 신경망
# =========================================================

def build_model(class_count):
    model = keras.Sequential([
        layers.Input(shape=(N_MFCC,)),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(class_count, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# =========================================================
# 6. 내 컴퓨터에서 학습·저장
# =========================================================

def train():
    keras.utils.set_random_seed(RANDOM_STATE)

    audio_dir = find_audio_dir()
    x, labels, counts, failures = load_dataset(audio_dir)

    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x).astype(np.float32)

    model = build_model(len(encoder.classes_))

    print("\n========== 학습 자료 ==========")

    for category, count in counts.items():
        print(f"{category}: {count}개")

    missing = [
        category for category in CATEGORIES
        if category not in counts
    ]

    if missing:
        print("\n음원이 없어 학습하지 않는 카테고리:")
        for category in missing:
            print(f"- {category}")

    print(f"\n전체 음원: {len(y)}개")
    print(f"학습 카테고리: {len(encoder.classes_)}개")
    print("\n50회 학습을 시작합니다.\n")

    model.fit(
        x_scaled,
        y,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        verbose=1,
    )

    loss, accuracy = model.evaluate(
        x_scaled,
        y,
        verbose=0,
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(encoder, ENCODER_PATH)

    # 배포 환경에서 버전을 맞출 수 있도록 기록합니다.
    from importlib.metadata import version

    package_names = [
        "streamlit",
        "tensorflow",
        "librosa",
        "soundfile",
        "scikit-learn",
        "joblib",
        "numpy",
    ]

    versions = {
        name: version(name)
        for name in package_names
    }

    metadata = {
        "classes": encoder.classes_.tolist(),
        "sample_counts": counts,
        "training_accuracy": float(accuracy),
        "training_loss": float(loss),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "sample_rate": SAMPLE_RATE,
        "n_mfcc": N_MFCC,
        "failed_files": failures,
        "package_versions": versions,
    }

    METADATA_PATH.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 学習環境に合わせた依存関係ファイルを別名で保存します。
    requirements = "\n".join(
        f"{name}=={versions[name]}"
        for name in package_names
    ) + "\n"

    requirements_path = BASE_DIR / "requirements_trained.txt"

    requirements_path.write_text(
        requirements,
        encoding="utf-8",
    )

    print("\n========== 学習完了 ==========")
    print(f"学習資料に対する正解率: {accuracy * 100:.1f}%")
    print("この数値は新しい音源に対する正解率ではありません。")
    print(f"\nモデル保存先:\n{MODEL_DIR}")
    print(
        "\nrequirements_trained.txt も作成しました。\n"
        "GitHubでは、この内容を requirements.txt として使ってください。"
    )


# =========================================================
# 7. Streamlit：保存済みモデルで分析
# =========================================================

def run_web():
    st.set_page_config(
        page_title="새소리 종 분류",
        page_icon="🐦",
        layout="centered",
    )

    st.title("🐦 새소리 종 분류")
    st.write("음원을 올리면 카테고리별 예측 점수를 표시합니다.")

    required_files = [
        MODEL_PATH,
        SCALER_PATH,
        ENCODER_PATH,
        METADATA_PATH,
    ]

    missing = [
        str(path.relative_to(BASE_DIR))
        for path in required_files
        if not path.is_file()
    ]

    if missing:
        st.error("학습된 모델 파일이 없습니다.")

        for name in missing:
            st.write(f"- {name}")

        st.info(
            "PyCharm에서 bird_app.py를 일반 실행해 학습한 뒤, "
            "생성된 bird_model 폴더 전체를 GitHub에 올려 주세요."
        )

        st.stop()

    @st.cache_resource
    def load_artifacts(model_time, scaler_time, encoder_time):
        model = keras.models.load_model(
            MODEL_PATH,
            compile=False,
        )

        scaler = joblib.load(SCALER_PATH)
        encoder = joblib.load(ENCODER_PATH)

        return model, scaler, encoder

    try:
        model, scaler, encoder = load_artifacts(
            MODEL_PATH.stat().st_mtime_ns,
            SCALER_PATH.stat().st_mtime_ns,
            ENCODER_PATH.stat().st_mtime_ns,
        )

        metadata = json.loads(
            METADATA_PATH.read_text(encoding="utf-8")
        )

    except Exception as exc:
        st.error(f"모델을 불러오지 못했습니다: {exc}")
        st.stop()

    with st.expander("학습 정보"):
        st.write("학습된 카테고리:", list(encoder.classes_))

        for category, count in metadata["sample_counts"].items():
            st.write(f"• {category}: {count}개")

        st.write(
            "학습 자료 정확도: "
            f"{metadata['training_accuracy'] * 100:.1f}%"
        )

        st.warning(
            "별도 검증·테스트 데이터를 나누지 않았습니다. "
            "이 정확도는 새로운 음원의 성능을 뜻하지 않습니다."
        )

    st.sidebar.header("배경소리 처리")

    use_background_rule = st.sidebar.checkbox(
        "최고 점수가 낮으면 배경소리로 처리",
        value=True,
    )

    threshold = st.sidebar.slider(
        "판단 기준",
        min_value=0.0,
        max_value=1.0,
        value=0.60,
        step=0.05,
    )

    st.sidebar.caption(
        "0.60은 검증되지 않은 임시 기준입니다. "
        "높일수록 실제 새소리도 배경소리로 처리될 수 있습니다."
    )

    uploaded = st.file_uploader(
        "분석할 음원을 선택하세요",
        type=["wav", "mp3", "flac", "ogg"],
    )

    if uploaded is None:
        st.info("학습용 ZIP이 아니라 분석할 음원 하나를 올려 주세요.")
        st.stop()

    audio_bytes = uploaded.getvalue()
    st.audio(audio_bytes)

    if st.button("음원 분석하기", type="primary"):
        try:
            with st.spinner("분석하고 있습니다..."):
                feature = extract_mfcc(io.BytesIO(audio_bytes))

                x_new = scaler.transform(
                    feature.reshape(1, -1)
                ).astype(np.float32)

                scores = model.predict(
                    x_new,
                    verbose=0,
                )[0]

            classes = encoder.classes_

            best_index = int(np.argmax(scores))
            raw_label = str(classes[best_index])
            best_score = float(scores[best_index])

            rule_applied = (
                use_background_rule
                and BACKGROUND in classes
                and raw_label != BACKGROUND
                and best_score < threshold
            )

            final_label = (
                BACKGROUND if rule_applied else raw_label
            )

            st.subheader(f"최종 분류: {final_label}")

            if rule_applied:
                st.warning(
                    f"원래 최고 점수는 {raw_label} "
                    f"{best_score * 100:.1f}%였지만, "
                    "설정 기준보다 낮아 배경소리로 처리했습니다."
                )
            else:
                st.write(
                    f"최고 예측 점수: **{best_score * 100:.1f}%**"
                )

            st.subheader("카테고리별 예측 점수")

            ranked = sorted(
                zip(classes, scores),
                key=lambda pair: float(pair[1]),
                reverse=True,
            )

            for category, score in ranked:
                score = float(score)

                st.write(
                    f"**{category}**: {score * 100:.2f}%"
                )

                st.progress(
                    min(max(score, 0.0), 1.0)
                )

            st.caption(
                "퍼센트는 Softmax 예측 점수입니다. "
                "실제 정답일 확률이나 검증 정확도가 아닙니다. "
                "배경소리 처리 규칙은 위 점수를 변경하지 않습니다."
            )

            st.info(
                "여러 종이 함께 울어도 파일 전체를 하나로 분류합니다. "
                "학습하지 않은 소리도 높은 점수로 오분류할 수 있습니다."
            )

        except Exception as exc:
            st.error(f"분석하지 못했습니다: {exc}")


# =========================================================
# 8. 실행 환경에 따라 자동 선택
# =========================================================

if __name__ == "__main__":
    if get_script_run_ctx(suppress_warning=True) is None:
        # PyCharm 초록색 실행 버튼 → 학습
        train()
    else:
        # Streamlit에서 실행 → 웹사이트
        run_web()
