import asyncio
import sys

# Asyncio event loop policy beállítása Windows-ra (Python 3.8+)
if sys.platform == "win32" and sys.version_info >= (3, 8, 0):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import streamlit as st
import yaml
import os
import torch  # A torch.cuda.is_available() ellenőrzéshez
import shutil  # Almappák törléséhez (opcionális)
import datetime  # Egyedi mappa nevekhez

# Importáljuk a meglévő moduljaink fő funkcióit
try:
    from extract_audio import extract_audio_from_video, check_ffmpeg
    from isolate_vocals import isolate_vocals_with_demucs
    from transcribe_segment import transcribe_and_segment
except ImportError as e:
    st.error(
        f"Hiba a modulok importálásakor: {e}. Győződj meg róla, hogy az `extract_audio.py`, `isolate_vocals.py`, és `transcribe_segment.py` fájlok ugyanabban a mappában vannak, mint az `app.py`, és a virtuális környezet aktív a szükséges csomagokkal.")
    st.stop()

# === Alap Streamlit Beállítások ===
st.set_page_config(page_title="AI Dataset Creator", layout="wide", initial_sidebar_state="expanded")
st.title("🎙️ AI Audio Dataset Creator 🎞️")
st.markdown(
    "Készíts adathalmazt videókból AI modellek (pl. TTS, RVC) tanításához. A folyamat kinyeri a hangot, opcionálisan eltávolítja a háttérzajt, szétválasztja a beszélőket, majd rövid szegmensekre bontja és átírja a beszédet.")

# === Konfiguráció Kezelése ===
CONFIG_PATH = 'config.yaml'


def load_default_config():
    default_conf = {
        'input_video': 'data/', 'output_base_dir': 'output',
        'output_raw_audio_filename': 'audio_raw.wav', 'output_clean_audio_filename': 'audio_clean.wav',
        'output_segments_dirname': 'segments', 'output_metadata_filename': 'metadata.csv',
        'target_sample_rate': 24000, 'language': 'hu', 'whisper_model_size': 'large-v3-turbo',
        # Itt javítottam large-v3-turbo-ról
        'min_segment_duration_ms': 3000, 'max_segment_duration_ms': 13000,
        'use_demucs': True, 'use_diarization': True,
        'huggingface_token': 'YOUR_HUGGINGFACE_TOKEN_HERE',
        'confidence_threshold': 0.7, 'device': 'auto',
        'output_dir_structure': 'speaker_separated'
    }
    if not os.path.exists(CONFIG_PATH):
        st.error(
            f"Alapértelmezett konfigurációs fájl ({CONFIG_PATH}) nem található! Hozz létre egyet a `config.example.yaml` alapján, vagy mentsd el a beállításokat az oldalsávon. Addig is alapértelmezett értékekkel dolgozunk.")
        return default_conf
    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            loaded_conf = yaml.safe_load(f)
            # Biztosítjuk, hogy minden kulcs létezzen, ha a fájl hiányos
            for key, value in default_conf.items():
                if key not in loaded_conf:
                    loaded_conf[key] = value
            return loaded_conf
    except Exception as e:
        st.error(f"Hiba az alapértelmezett konfiguráció olvasása közben: {e}. Alapértelmezett értékekkel dolgozunk.")
        return default_conf


def save_current_config_to_yaml(current_config, path=CONFIG_PATH):
    try:
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(current_config, f, sort_keys=False, allow_unicode=True, Dumper=yaml.SafeDumper)
        st.sidebar.success(f"Konfiguráció mentve ide: {os.path.abspath(path)}")
    except Exception as e:
        st.sidebar.error(f"Hiba a konfiguráció mentése közben: {e}")


config_defaults = load_default_config()

# === Globális Változók a UI Állapothoz ===
base_output_for_datasets = config_defaults.get('output_base_dir', 'output')
os.makedirs(base_output_for_datasets, exist_ok=True)
temp_upload_main_dir = os.path.join(base_output_for_datasets, "temp_uploads")
os.makedirs(temp_upload_main_dir, exist_ok=True)

# === Oldalsáv a Konfigurációhoz ===
with st.sidebar:
    st.header("⚙️ Feldolgozási Beállítások")

    # --- Adathalmaz Mód ---
    st.subheader("0. Adathalmaz Cél")
    dataset_mode_options = ("Új adathalmaz létrehozása", "Meglévő adathalmazhoz adás")
    if 'dataset_mode_radio_state' not in st.session_state:
        st.session_state.dataset_mode_radio_state = dataset_mode_options[0]

    existing_datasets_for_radio = [d for d in os.listdir(base_output_for_datasets) if
                                   os.path.isdir(os.path.join(base_output_for_datasets, d)) and d != "temp_uploads"]

    # Ha nincs meglévő adathalmaz, és a "Meglévőhöz adás" van kiválasztva, automatikusan váltson "Új"-ra
    # Ezt a radio widget `index` paraméterével kezeljük
    default_radio_idx = 0
    if st.session_state.dataset_mode_radio_state == dataset_mode_options[1] and not existing_datasets_for_radio:
        st.session_state.dataset_mode_radio_state = dataset_mode_options[0]  # Frissítjük a state-et
        # st.rerun() # Nem szükséges itt, a widget indexe gondoskodik róla
    elif st.session_state.dataset_mode_radio_state == dataset_mode_options[1] and existing_datasets_for_radio:
        default_radio_idx = 1

    dataset_mode = st.radio(
        "Válassz feldolgozási módot:",
        dataset_mode_options,
        index=default_radio_idx,
        key="dataset_mode_radio_widget"  # Új kulcs a widgetnek
    )
    # Frissítjük a session state-et a widget aktuális értékével, ha a felhasználó változtat
    if dataset_mode != st.session_state.dataset_mode_radio_state:
        st.session_state.dataset_mode_radio_state = dataset_mode
        st.rerun()  # Újrafuttatás a UI frissítéséhez a módváltás után

    target_dataset_name_ui = ""
    new_dataset_name_suggestion = f"dataset_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    if st.session_state.dataset_mode_radio_state == "Új adathalmaz létrehozása":
        target_dataset_name_ui = st.text_input("Új adathalmaz neve:", value=new_dataset_name_suggestion,
                                               key="new_dataset_name_input")
    else:  # Meglévő adathalmazhoz adás
        # A listát itt újra lekérdezzük, hogy friss legyen
        current_existing_datasets_for_selectbox = [d for d in os.listdir(base_output_for_datasets) if os.path.isdir(
            os.path.join(base_output_for_datasets, d)) and d != "temp_uploads"]
        if not current_existing_datasets_for_selectbox:
            # Ez az ág elvileg nem futhat le, ha a radio gomb logikája helyes
            st.warning(
                f"Hiba: Nincsenek meglévő adathalmazok, de a \"Meglévőhöz adás\" mód aktív. Kérlek, válts \"Új adathalmaz létrehozása\" módra.")
            target_dataset_name_ui = ""
        else:
            selected_existing_dataset = st.selectbox(
                "Válassz meglévő adathalmazt:",
                options=current_existing_datasets_for_selectbox,
                index=0 if current_existing_datasets_for_selectbox else None,
                key="existing_dataset_select_box"
            )
            target_dataset_name_ui = selected_existing_dataset if selected_existing_dataset else ""

    # --- Bemenet ---
    st.subheader("1. Bemenet")
    # Session state a feltöltött fájlokhoz és az elérési úthoz
    if 'uploaded_files_list_state' not in st.session_state:  # Lista a feltöltött fájloknak
        st.session_state.uploaded_files_list_state = []
    if 'input_path_text_state' not in st.session_state:
        st.session_state.input_path_text_state = config_defaults.get('input_video', 'data/')


    def clear_path_if_upload():
        if st.session_state.file_uploader_widget:  # Ha van új feltöltés
            st.session_state.input_path_text_state = ""  # Töröljük az útvonalat


    def clear_upload_if_path():
        if st.session_state.path_input_widget_for_disable_check:  # Ha az útvonalba írtak
            st.session_state.uploaded_files_list_state = []  # Töröljük a feltöltött fájlokat
            # A file_uploader widgetet nem tudjuk közvetlenül "kiüríteni", de a logikánk a session state-re épül


    uploaded_files_ui = st.file_uploader(
        "Videófájl(ok) feltöltése (max. 200MB/fájl böngészőn keresztül):",
        type=['mp4', 'mkv', 'mov', 'avi', 'webm'],
        accept_multiple_files=True,
        key="file_uploader_widget",
        disabled=bool(
            st.session_state.input_path_text_state and st.session_state.input_path_text_state.strip() and st.session_state.input_path_text_state != 'data/'),
        on_change=clear_path_if_upload
    )
    # Frissítjük a session state-et, ha a widget értéke változik
    if uploaded_files_ui is not None:  # Fontos, hogy ne None legyen, mert az is valid érték, ha nincs feltöltés
        st.session_state.uploaded_files_list_state = uploaded_files_ui

    input_video_path_ui_val = st.text_input(
        "Vagy add meg a videófájl/mappa TELJES elérési útját:",
        value=st.session_state.input_path_text_state,
        key="path_input_widget_for_disable_check",  # Másik kulcs az on_change-hez
        disabled=bool(st.session_state.uploaded_files_list_state),  # Ha van feltöltött fájl, ez inaktív
        on_change=clear_upload_if_path
    )
    # Frissítjük a session state-et, ha a widget értéke változik
    if input_video_path_ui_val != st.session_state.input_path_text_state:
        st.session_state.input_path_text_state = input_video_path_ui_val

    # --- Audio Paraméterek ---
    st.subheader("3. Audio Paraméterek")
    target_sr_ui = st.selectbox(
        "Cél mintavételezési frekvencia (Hz):", options=[16000, 22050, 24000, 44100, 48000],
        index=[16000, 22050, 24000, 44100, 48000].index(config_defaults.get('target_sample_rate', 24000)),
        key="sample_rate_select"
    )
    language_ui = st.text_input("Nyelv (ISO kód, pl. 'hu', 'en'):", value=config_defaults.get('language', 'hu'),
                                key="language_input")

    # --- Whisper Modell ---
    st.subheader("4. Whisper Modell")
    whisper_model_size_ui = st.selectbox(
        "Whisper modell mérete:", options=["tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3-turbo"],
        # large-v3-turbo helyett large-v3
        index=["tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3-turbo"].index(
            config_defaults.get('whisper_model_size', 'large-v3-turbo')), key="whisper_model_select"  # Javítva a defaultra
    )

    # --- Szegmentálás ---
    st.subheader("5. Szegmentálás")
    min_dur_ms_ui = st.number_input("Minimum szegmenshossz (ms):", min_value=500, max_value=30000,
                                    value=config_defaults.get('min_segment_duration_ms', 3000), step=100,
                                    key="min_duration_input")
    max_dur_ms_ui = st.number_input("Maximum szegmenshossz (ms):", min_value=1000, max_value=60000,
                                    value=config_defaults.get('max_segment_duration_ms', 13000), step=100,
                                    key="max_duration_input")

    # --- Opcionális Funkciók ---
    st.subheader("6. Opcionális Funkciók")
    use_demucs_ui = st.checkbox("Vokál izoláció (Demucs)", value=config_defaults.get('use_demucs', True),
                                key="use_demucs_checkbox")
    use_diarization_ui = st.checkbox("Beszélők szétválasztása (Diarizáció)",
                                     value=config_defaults.get('use_diarization', True), key="use_diarization_checkbox")
    hf_token_ui = st.text_input("Hugging Face Token (diarizációhoz):",
                                value=config_defaults.get('huggingface_token', 'YOUR_HUGGINGFACE_TOKEN_HERE'),
                                type="password", key="hf_token_input")

    # --- Minőség és Eszköz ---
    st.subheader("7. Minőség és Eszköz")
    confidence_threshold_ui = st.slider("Konfidencia küszöb (átirat ellenőrzéshez):", min_value=0.0, max_value=1.0,
                                        value=config_defaults.get('confidence_threshold', 0.7), step=0.05,
                                        key="confidence_slider")

    device_options = ["auto", "cpu"]
    default_device_index = 0
    if torch.cuda.is_available():
        device_options.append("cuda")
        saved_device = config_defaults.get('device', 'auto')
        if saved_device in device_options: default_device_index = device_options.index(saved_device)
    elif config_defaults.get('device', 'auto') == "cpu":
        default_device_index = device_options.index("cpu")
    device_ui = st.selectbox("Eszköz (CPU/GPU):", options=device_options, index=default_device_index,
                             key="device_select_box")

    if st.button("Jelenlegi beállítások mentése config.yaml-ba", key="save_config_sidebar_button"):
        current_config_for_save = {
            'input_video': st.session_state.input_path_text_state,
            'output_base_dir': base_output_for_datasets,
            'output_raw_audio_filename': config_defaults.get('output_raw_audio_filename'),
            'output_clean_audio_filename': config_defaults.get('output_clean_audio_filename'),
            'output_segments_dirname': config_defaults.get('output_segments_dirname'),
            'output_metadata_filename': config_defaults.get('output_metadata_filename'),
            'target_sample_rate': target_sr_ui, 'language': language_ui,
            'whisper_model_size': whisper_model_size_ui,
            'min_segment_duration_ms': min_dur_ms_ui, 'max_segment_duration_ms': max_dur_ms_ui,
            'use_demucs': use_demucs_ui, 'use_diarization': use_diarization_ui,
            'huggingface_token': hf_token_ui, 'confidence_threshold': confidence_threshold_ui,
            'device': device_ui,
            'output_dir_structure': 'speaker_separated' if use_diarization_ui else 'flat'
        }
        save_current_config_to_yaml(current_config_for_save)

# === Fő Tartalom - Feldolgozás Indítása ===
st.markdown("---")

# A végleges adathalmaz nevének meghatározása a widgetek aktuális állapotából
# Ezt minden rendereléskor újra kell számolni
# Először lekérdezzük a meglévő adathalmazokat, hogy a logika helyes legyen
current_existing_datasets_for_button_logic = [
    d for d in os.listdir(base_output_for_datasets)
    if os.path.isdir(os.path.join(base_output_for_datasets, d)) and d != "temp_uploads"
]

final_target_dataset_name_for_button_logic = ""
if st.session_state.dataset_mode_radio_state == "Új adathalmaz létrehozása":
    final_target_dataset_name_for_button_logic = st.session_state.get("new_dataset_name_input", new_dataset_name_suggestion)
    # Ha a fallback input mező volt aktív (mert nem volt meglévő dataset, és a "Meglévő" mód volt a radio gombon)
    if not current_existing_datasets_for_button_logic and "new_dataset_name_if_none_exist_for_existing_mode" in st.session_state:
        final_target_dataset_name_for_button_logic = st.session_state.new_dataset_name_if_none_exist_for_existing_mode

elif current_existing_datasets_for_button_logic and "existing_dataset_select_box" in st.session_state:
    # Csak akkor olvassuk ki a selectbox értékét, ha tényleg vannak meglévő adathalmazok
    final_target_dataset_name_for_button_logic = st.session_state.existing_dataset_select_box
# Ha "Meglévő" van kiválasztva, de valójában nincs meglévő (és a fallback inputot használjuk)
elif not current_existing_datasets_for_button_logic and "new_dataset_name_if_none_exist_for_existing_mode" in st.session_state:
    final_target_dataset_name_for_button_logic = st.session_state.new_dataset_name_if_none_exist_for_existing_mode


process_button_disabled_final = not final_target_dataset_name_for_button_logic or not final_target_dataset_name_for_button_logic.strip()
button_tooltip_final = "Kérlek, adj meg/válassz egy adathalmaz nevet az oldalsávon!" if process_button_disabled_final else "Indítsd el az összes videó feldolgozását a beállított adathalmazhoz."

if st.button("🚀 Teljes Feldolgozás Indítása", type="primary", use_container_width=True, key="start_processing_main_button", disabled=process_button_disabled_final, help=button_tooltip_final):
    actual_output_dataset_dir = os.path.join(base_output_for_datasets, final_target_dataset_name_for_button_logic)
    os.makedirs(actual_output_dataset_dir, exist_ok=True)
    st.info(f"Kimenetek mentése a következő adathalmaz mappába: **{os.path.abspath(actual_output_dataset_dir)}**")

    runtime_config = {
        'output_base_dir': actual_output_dataset_dir,
        'output_raw_audio_filename': config_defaults.get('output_raw_audio_filename', 'audio_raw.wav'),
        'output_clean_audio_filename': config_defaults.get('output_clean_audio_filename', 'audio_clean.wav'),
        'output_segments_dirname': config_defaults.get('output_segments_dirname', 'segments'),
        'output_metadata_filename': config_defaults.get('output_metadata_filename', 'metadata.csv'),
        'output_metadata_filename': config_defaults.get('output_metadata_filename', 'metadata.csv'),
        'target_sample_rate': target_sr_ui, 'language': language_ui,
        'whisper_model_size': whisper_model_size_ui,
        'min_segment_duration_ms': min_dur_ms_ui, 'max_segment_duration_ms': max_dur_ms_ui,
        'use_demucs': use_demucs_ui, 'use_diarization': use_diarization_ui,
        'huggingface_token': hf_token_ui if hf_token_ui and hf_token_ui.strip() and hf_token_ui != 'YOUR_HUGGINGFACE_TOKEN_HERE' else None,
        'confidence_threshold': confidence_threshold_ui, 'device': device_ui,
        'output_dir_structure': 'speaker_separated' if use_diarization_ui else 'flat'
    }

    videos_to_process = []
    temp_uploaded_paths = []

    current_uploaded_files_from_state = st.session_state.get('uploaded_files_list_state', [])
    current_input_path_from_state = st.session_state.get('input_path_text_state', '')

    if current_uploaded_files_from_state:
        for uploaded_file_item in current_uploaded_files_from_state:
            temp_video_path = os.path.join(temp_upload_main_dir, uploaded_file_item.name)
            with open(temp_video_path, "wb") as f: f.write(uploaded_file_item.getbuffer())
            videos_to_process.append(temp_video_path)
            temp_uploaded_paths.append(temp_video_path)
        st.info(f"{len(current_uploaded_files_from_state)} videó feltöltve feldolgozásra.")
    elif current_input_path_from_state and current_input_path_from_state.strip() and current_input_path_from_state != 'data/':
        path_to_check = current_input_path_from_state
        if not os.path.isabs(path_to_check): path_to_check = os.path.abspath(path_to_check)
        if os.path.isdir(path_to_check):
            st.info(f"Videók keresése a '{path_to_check}' mappában...")
            for filename in os.listdir(path_to_check):
                if filename.lower().endswith(('.mp4', '.mkv', '.mov', '.avi', '.webm')):
                    videos_to_process.append(os.path.join(path_to_check, filename))
            if not videos_to_process:
                st.warning(f"Nem találhatóak videófájlok a '{path_to_check}' mappában.")
            else:
                st.info(f"{len(videos_to_process)} videó található a mappában.")
        elif os.path.isfile(path_to_check):
            videos_to_process.append(path_to_check)
        else:
            st.error(f"A megadott elérési út nem létezik vagy nem támogatott: {path_to_check}"); st.stop()
    else:
        st.error("Nincs videó kiválasztva vagy érvényes elérési út megadva."); st.stop()

    if not videos_to_process: st.error("Nincsenek feldolgozandó videók."); st.stop()

    log_placeholder = st.empty()
    progress_bar_overall = st.progress(0, text="Teljes feldolgozás...")
    all_videos_processed_successfully = True

    for video_index, current_video_path_original in enumerate(videos_to_process):
        video_basename = os.path.basename(current_video_path_original)
        log_placeholder.info(f"Feldolgozás alatt: {video_basename} ({video_index + 1}/{len(videos_to_process)})")
        safe_video_name = "".join(c if c.isalnum() else "_" for c in os.path.splitext(video_basename)[0])
        video_processing_output_dir = os.path.join(actual_output_dataset_dir, safe_video_name)
        os.makedirs(video_processing_output_dir, exist_ok=True)

        video_config = runtime_config.copy()
        video_config['input_video'] = current_video_path_original
        video_config['output_base_dir'] = video_processing_output_dir
        video_config['output_metadata_file_absolute_path'] = os.path.join(actual_output_dataset_dir,
                                                                          runtime_config['output_metadata_filename'])
        video_config['segments_relative_path_prefix'] = safe_video_name

        st.info(f"Indul: {video_basename}")
        st.caption(f"Kimenetek (nyers, tiszta, szegmensek) ide: {os.path.abspath(video_config['output_base_dir'])}")

        raw_audio_file = None;
        clean_audio_file_for_this_video = None;
        input_for_transcription_this_video = None
        current_video_success = True

        with st.expander(f"[{safe_video_name}] 1. Audio kinyerése...", expanded=True):
            if not check_ffmpeg():
                st.error("FFmpeg nem található.");
                current_video_success = False
            if current_video_success:
                raw_audio_file = extract_audio_from_video(video_config)
                if raw_audio_file:
                    st.success(f"Nyers audio: {os.path.basename(raw_audio_file)}")
                else:
                    st.error("Audio kinyerés sikertelen."); current_video_success = False

        if current_video_success and video_config['use_demucs']:
            with st.expander(f"[{safe_video_name}] 2. Vokál izoláció (Demucs)...", expanded=True):
                clean_audio_file_for_this_video = isolate_vocals_with_demucs(video_config, raw_audio_file)
                if clean_audio_file_for_this_video:
                    st.success(f"Tiszta vokál: {os.path.basename(clean_audio_file_for_this_video)}")
                    input_for_transcription_this_video = clean_audio_file_for_this_video
                else:
                    st.warning("Demucs hiba. Nyers audió használata.")
                    input_for_transcription_this_video = raw_audio_file
        elif current_video_success:
            st.info(f"[{safe_video_name}] Demucs kihagyva.")
            input_for_transcription_this_video = raw_audio_file

        if current_video_success and input_for_transcription_this_video:
            with st.expander(f"[{safe_video_name}] 3. Transzkripció és szegmentálás...", expanded=True):
                st.write(f"WhisperX futtatása: {os.path.basename(input_for_transcription_this_video)}")
                transcribe_and_segment(video_config)
                st.success("Transzkripció és szegmentálás befejezve ehhez a videóhoz!")
        elif current_video_success:
            st.error(f"[{safe_video_name}] Nincs audio fájl a transzkripcióhoz.");
            current_video_success = False

        if not current_video_success:
            all_videos_processed_successfully = False

        progress_bar_overall.progress((video_index + 1) / len(videos_to_process),
                                      text=f"Videó {video_index + 1}/{len(videos_to_process)} feldolgozva.")

    if temp_uploaded_paths:
        st.markdown("---");
        st.write("Ideiglenes feltöltött fájlok törlése...")
        for temp_path in temp_uploaded_paths:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path); st.caption(f"Törölve: {temp_path}")
                except Exception as e:
                    st.warning(f"Hiba törléskor ({temp_path}): {e}")
        if os.path.exists(temp_upload_main_dir) and not os.listdir(temp_upload_main_dir):
            try:
                shutil.rmtree(temp_upload_main_dir); st.caption(f"Ideiglenes mappa törölve: {temp_upload_main_dir}")
            except Exception as e:
                st.warning(f"Hiba az ideiglenes mappa törlésekor: {e}")

    st.markdown("---")
    if videos_to_process:
        if all_videos_processed_successfully:
            st.balloons()
            st.header("🎉 Feldolgozás Befejezve! 🎉")
            st.success(f"Az összes videó feldolgozása sikeresen megtörtént.")
        else:
            st.warning(
                "Egy vagy több videó feldolgozása során hiba történt. Ellenőrizd a fenti üzeneteket és a terminál logjait.")
        st.write(
            f"A közös metaadatok (átiratok) a `{os.path.abspath(os.path.join(actual_output_dataset_dir, runtime_config['output_metadata_filename']))}` fájlban találhatók.")
        st.write(
            f"A szegmentált audio fájlok a `{os.path.abspath(actual_output_dataset_dir)}` mappán belüli, videó nevű almappákban (`segments` almappákon belül) vannak.")
    else:
        st.info("Nem volt feldolgozandó videó.")

with st.sidebar:
    st.markdown("---")
    st.info("Fejlesztette: AI Asszisztens")
