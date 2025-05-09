import os
import yaml
import whisperx
from whisperx.diarize import DiarizationPipeline
import torch
import pandas as pd
from pydub import AudioSegment
import logging
import sys
import gc
import numpy as np
import shutil

# === Logger beállítása ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])
log = logging.getLogger(__name__)


# === Konfiguráció betöltése (Ezt az app.py hívja meg a config objektummal) ===
# def load_config(config_path='config.yaml'): # Erre itt már nincs szükség, a configot kívülről kapja

# === Eszköz (CPU/GPU) meghatározása ===
def get_device(config_device):
    if config_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    elif config_device == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


# === Fő funkció ===
def transcribe_and_segment(config):  # Most a configot paraméterként kapja
    """
    Elvégzi a transzkripciót, igazítást, (opcionálisan) diarizációt és szegmentálást.
    A szegmenseket a config['output_base_dir'] / config['output_segments_dirname']-be menti.
    A metaadatokat a config['output_metadata_file_absolute_path']-ba menti/bővíti.
    A CSV-ben lévő file_path-ok a config['segments_relative_path_prefix'] előtagot kapják.
    """
    # Konfigurációs értékek kiolvasása
    # A 'output_base_dir' most a videóspecifikus almappára mutat (pl. output/dataset_neve/video_neve)
    video_specific_output_dir = config.get('output_base_dir')
    raw_audio_filename = config.get('output_raw_audio_filename')
    clean_audio_filename = config.get('output_clean_audio_filename')
    segments_dirname_in_video_folder = config.get('output_segments_dirname')  # Pl. "segments"

    # A közös metadata.csv abszolút elérési útja
    common_metadata_path = config.get('output_metadata_file_absolute_path')
    # Előtag a CSV-ben lévő file_path-okhoz (a videó neve, pl. "video_neve")
    csv_path_prefix = config.get('segments_relative_path_prefix', '')

    lang = config.get('language')
    whisper_model_size = config.get('whisper_model_size')
    min_dur_ms = config.get('min_segment_duration_ms')
    max_dur_ms = config.get('max_segment_duration_ms')
    use_demucs = config.get('use_demucs')
    use_diarization = config.get('use_diarization')
    hf_token = config.get('huggingface_token')
    conf_threshold = config.get('confidence_threshold')
    output_structure = config.get('output_dir_structure')
    config_device_setting = config.get('device')

    device = get_device(config_device_setting)
    compute_type = "float16" if device == "cuda" else "int8"
    log.info(f"[{csv_path_prefix}] Használt eszköz: {device}, Whisper compute type: {compute_type}")

    # Bemeneti audio fájl (a videóspecifikus mappán belül)
    input_audio_path = os.path.join(video_specific_output_dir,
                                    clean_audio_filename if use_demucs else raw_audio_filename)
    if not os.path.exists(input_audio_path):
        log.error(f"[{csv_path_prefix}] Hiba: Bemeneti audio fájl nem található: {input_audio_path}")
        return

    # Kimeneti szegmens mappa (a videóspecifikus mappán belül)
    # Pl. output/dataset_neve/video_neve/segments/
    segments_output_path_for_this_video = os.path.join(video_specific_output_dir, segments_dirname_in_video_folder)
    try:
        os.makedirs(segments_output_path_for_this_video, exist_ok=True)
        log.info(f"[{csv_path_prefix}] Szegmens kimeneti mappa: {os.path.abspath(segments_output_path_for_this_video)}")
    except OSError as e:
        log.error(
            f"[{csv_path_prefix}] Hiba a szegmens mappa létrehozásakor ({segments_output_path_for_this_video}): {e}")
        return

    model = None;
    align_model = None;
    metadata_align = None;
    diarization_pipeline = None;
    audio = None;
    full_audio_pydub = None;
    result = None;
    whisper_segments_list = None;

    try:
        log.info(f"[{csv_path_prefix}] Audio betöltése whisperx számára: {input_audio_path}")
        audio = whisperx.load_audio(input_audio_path)
        log.info(f"[{csv_path_prefix}] Audio betöltve whisperx számára.")

        log.info(f"[{csv_path_prefix}] Whisper modell betöltése: {whisper_model_size} ({device})")
        batch_size = 16 if device == "cuda" else 4
        model = whisperx.load_model(whisper_model_size, device, compute_type=compute_type, language=lang)
        log.info(f"[{csv_path_prefix}] Whisper modell betöltve.")

        log.info(f"[{csv_path_prefix}] Transzkripció indítása (batch_size={batch_size})...")
        result = model.transcribe(audio, batch_size=batch_size)
        log.info(f"[{csv_path_prefix}] Transzkripció kész.")

        log.info(f"[{csv_path_prefix}] Szó szintű igazítás indítása...")
        align_model, metadata_align = whisperx.load_align_model(language_code=lang, device=device)
        result = whisperx.align(result["segments"], align_model, metadata_align, audio, device,
                                return_char_alignments=False)
        log.info(f"[{csv_path_prefix}] Igazítás kész.")

        if use_diarization:
            log.info(f"[{csv_path_prefix}] Beszélő diarizáció indítása...")
            if not hf_token:
                log.error(f"[{csv_path_prefix}] Hiba: Diarizációhoz HF token szükséges.")
                raise ValueError("Hiányzó Hugging Face token a diarizációhoz.")
            diarization_pipeline = DiarizationPipeline(use_auth_token=hf_token, device=device)
            log.info(f"[{csv_path_prefix}] Diarizációs pipeline betöltve.")
            diarize_segments = diarization_pipeline(input_audio_path, min_speakers=1, max_speakers=5)
            log.info(f"[{csv_path_prefix}] Diarizáció lefuttatva.")
            result = whisperx.assign_word_speakers(diarize_segments, result)
            log.info(f"[{csv_path_prefix}] Beszélők hozzárendelve.")
        else:
            log.info(f"[{csv_path_prefix}] Diarizáció kihagyva.")
            for segment_res in result['segments']:
                if 'speaker' not in segment_res: segment_res['speaker'] = 'SPEAKER_00'

        log.info(f"[{csv_path_prefix}] Audio szegmentálása és metaadatok generálása...")
        all_metadata_for_this_video = []
        segment_counter = {}

        log.info(f"[{csv_path_prefix}] Audio betöltése pydub számára vágáshoz: {input_audio_path}")
        full_audio_pydub = AudioSegment.from_wav(input_audio_path)
        log.info(f"[{csv_path_prefix}] Audio betöltve pydub-bal.")

        current_segment_start_time = None;
        current_segment_end_time = None
        current_segment_text = "";
        current_segment_words = [];
        current_speaker = None

        whisper_segments_list = result.get("segments", [])
        if not whisper_segments_list:
            log.warning(f"[{csv_path_prefix}] Nem találhatóak Whisper szegmensek az igazítás után.")
            return

        log.info(f"[{csv_path_prefix}] Szegmensek feldolgozása ({len(whisper_segments_list)} db)...")

        for i, segment_data in enumerate(whisper_segments_list):
            speaker = segment_data.get('speaker', 'SPEAKER_00')
            words = segment_data.get('words', [])
            if not words: continue

            segment_start = words[0]['start']
            segment_end = words[-1]['end']

            if current_speaker is None or speaker != current_speaker:
                if current_segment_text and current_segment_start_time is not None and \
                        (current_segment_end_time - current_segment_start_time) * 1000 >= min_dur_ms:
                    # --- Mentés Logika ---
                    segment_duration = (current_segment_end_time - current_segment_start_time) * 1000
                    avg_confidence = np.mean([w.get('score', 0) for w in current_segment_words if
                                              'score' in w]) if current_segment_words else 0.0
                    needs_review = avg_confidence < conf_threshold

                    if current_speaker not in segment_counter: segment_counter[current_speaker] = 0
                    segment_index = segment_counter[current_speaker]
                    wav_filename_only = f"segment_{current_speaker}_{segment_index:05d}.wav"  # Csak a fájlnév

                    # A WAV fájl mentési útvonala a videóspecifikus almappán belül
                    speaker_dir_for_wav_segments = segments_output_path_for_this_video  # Alapból ide
                    if output_structure == 'speaker_separated':
                        speaker_dir_for_wav_segments = os.path.join(segments_output_path_for_this_video,
                                                                    current_speaker)
                        os.makedirs(speaker_dir_for_wav_segments, exist_ok=True)

                    segment_output_wav_path = os.path.join(speaker_dir_for_wav_segments, wav_filename_only)

                    # A CSV-be kerülő relatív útvonal (a közös adathalmaz gyökeréhez képest)
                    # csv_path_prefix (pl. "video_neve") / segments_dirname_in_video_folder (pl. "segments") / [speaker_id] / wav_filename_only
                    relative_path_parts_for_csv = [csv_path_prefix, segments_dirname_in_video_folder]
                    if output_structure == 'speaker_separated':
                        relative_path_parts_for_csv.append(current_speaker)
                    relative_path_parts_for_csv.append(wav_filename_only)
                    file_path_for_csv = os.path.join(*relative_path_parts_for_csv).replace("\\", "/")

                    audio_chunk = full_audio_pydub[
                                  int(current_segment_start_time * 1000):int(current_segment_end_time * 1000)]
                    audio_chunk.export(segment_output_wav_path, format="wav")

                    all_metadata_for_this_video.append({
                        "file_path": file_path_for_csv, "transcription": current_segment_text.strip(),
                        "duration_ms": int(segment_duration), "speaker_id": current_speaker,
                        "confidence": round(avg_confidence, 4), "needs_review": needs_review
                    })
                    segment_counter[current_speaker] += 1
                    log.debug(
                        f"[{csv_path_prefix}] Mentve (beszélőváltás): {segment_output_wav_path}, CSV út: {file_path_for_csv}")
                    # --- Mentés Logika Vége ---

                current_speaker = speaker
                current_segment_start_time = segment_start
                current_segment_end_time = segment_end
                current_segment_text = segment_data.get('text', '').strip()
                current_segment_words = words
            else:
                potential_end_time = segment_end
                potential_duration_ms = (potential_end_time - current_segment_start_time) * 1000
                if potential_duration_ms > max_dur_ms:
                    if current_segment_text and current_segment_start_time is not None and \
                            (current_segment_end_time - current_segment_start_time) * 1000 >= min_dur_ms:
                        # --- Mentés Logika ---
                        segment_duration = (current_segment_end_time - current_segment_start_time) * 1000
                        avg_confidence = np.mean([w.get('score', 0) for w in current_segment_words if
                                                  'score' in w]) if current_segment_words else 0.0
                        needs_review = avg_confidence < conf_threshold

                        if current_speaker not in segment_counter: segment_counter[current_speaker] = 0
                        segment_index = segment_counter[current_speaker]
                        wav_filename_only = f"segment_{current_speaker}_{segment_index:05d}.wav"

                        speaker_dir_for_wav_segments = segments_output_path_for_this_video
                        if output_structure == 'speaker_separated':
                            speaker_dir_for_wav_segments = os.path.join(segments_output_path_for_this_video,
                                                                        current_speaker)
                            os.makedirs(speaker_dir_for_wav_segments, exist_ok=True)
                        segment_output_wav_path = os.path.join(speaker_dir_for_wav_segments, wav_filename_only)

                        relative_path_parts_for_csv = [csv_path_prefix, segments_dirname_in_video_folder]
                        if output_structure == 'speaker_separated':
                            relative_path_parts_for_csv.append(current_speaker)
                        relative_path_parts_for_csv.append(wav_filename_only)
                        file_path_for_csv = os.path.join(*relative_path_parts_for_csv).replace("\\", "/")

                        audio_chunk = full_audio_pydub[
                                      int(current_segment_start_time * 1000):int(current_segment_end_time * 1000)]
                        audio_chunk.export(segment_output_wav_path, format="wav")

                        all_metadata_for_this_video.append({
                            "file_path": file_path_for_csv, "transcription": current_segment_text.strip(),
                            "duration_ms": int(segment_duration), "speaker_id": current_speaker,
                            "confidence": round(avg_confidence, 4), "needs_review": needs_review
                        })
                        segment_counter[current_speaker] += 1
                        log.debug(
                            f"[{csv_path_prefix}] Mentve (max hossz): {segment_output_wav_path}, CSV út: {file_path_for_csv}")
                        # --- Mentés Logika Vége ---
                    current_segment_start_time = segment_start
                    current_segment_end_time = segment_end
                    current_segment_text = segment_data.get('text', '').strip()
                    current_segment_words = words
                else:
                    current_segment_text += " " + segment_data.get('text', '').strip()
                    current_segment_end_time = segment_end
                    current_segment_words.extend(words)

        if current_segment_text and current_segment_start_time is not None and \
                (current_segment_end_time - current_segment_start_time) * 1000 >= min_dur_ms:
            # --- Mentés Logika (utolsó) ---
            segment_duration = (current_segment_end_time - current_segment_start_time) * 1000
            avg_confidence = np.mean(
                [w.get('score', 0) for w in current_segment_words if 'score' in w]) if current_segment_words else 0.0
            needs_review = avg_confidence < conf_threshold

            if current_speaker not in segment_counter: segment_counter[current_speaker] = 0
            segment_index = segment_counter[current_speaker]
            wav_filename_only = f"segment_{current_speaker}_{segment_index:05d}.wav"

            speaker_dir_for_wav_segments = segments_output_path_for_this_video
            if output_structure == 'speaker_separated':
                speaker_dir_for_wav_segments = os.path.join(segments_output_path_for_this_video, current_speaker)
                os.makedirs(speaker_dir_for_wav_segments, exist_ok=True)
            segment_output_wav_path = os.path.join(speaker_dir_for_wav_segments, wav_filename_only)

            relative_path_parts_for_csv = [csv_path_prefix, segments_dirname_in_video_folder]
            if output_structure == 'speaker_separated':
                relative_path_parts_for_csv.append(current_speaker)
            relative_path_parts_for_csv.append(wav_filename_only)
            file_path_for_csv = os.path.join(*relative_path_parts_for_csv).replace("\\", "/")

            audio_chunk = full_audio_pydub[int(current_segment_start_time * 1000):int(current_segment_end_time * 1000)]
            audio_chunk.export(segment_output_wav_path, format="wav")

            all_metadata_for_this_video.append({
                "file_path": file_path_for_csv, "transcription": current_segment_text.strip(),
                "duration_ms": int(segment_duration), "speaker_id": current_speaker,
                "confidence": round(avg_confidence, 4), "needs_review": needs_review
            })
            segment_counter[current_speaker] += 1
            log.debug(f"[{csv_path_prefix}] Mentve (utolsó): {segment_output_wav_path}, CSV út: {file_path_for_csv}")
            # --- Mentés Logika Vége ---

        # Metaadatok mentése/bővítése a KÖZÖS CSV fájlba
        if all_metadata_for_this_video:
            df_new_segments = pd.DataFrame(all_metadata_for_this_video)
            column_order = ["file_path", "transcription", "duration_ms", "speaker_id", "confidence", "needs_review"]
            df_new_segments = df_new_segments[[col for col in column_order if col in df_new_segments.columns]]

            if os.path.exists(common_metadata_path):
                log.info(f"[{csv_path_prefix}] Meglévő közös metaadat fájl ({common_metadata_path}) bővítése...")
                try:
                    df_existing = pd.read_csv(common_metadata_path)
                    if not all(col in df_existing.columns for col in df_new_segments.columns) or \
                            not all(col in df_new_segments.columns for col in df_existing.columns) or \
                            list(df_existing.columns) != list(df_new_segments.columns):  # Szigorúbb oszlopellenőrzés
                        log.warning(
                            f"[{csv_path_prefix}] Oszlopeltérés a meglévő és új metaadatok között! Meglévő: {list(df_existing.columns)}, Új: {list(df_new_segments.columns)}. Felülírás az újjal.")
                        df_final = df_new_segments  # Vagy hibaüzenet és leállás
                    else:
                        df_final = pd.concat([df_existing, df_new_segments], ignore_index=True)
                except pd.errors.EmptyDataError:
                    log.info(f"[{csv_path_prefix}] A meglévő közös metaadat fájl üres volt.")
                    df_final = df_new_segments
                except Exception as e:
                    log.warning(
                        f"[{csv_path_prefix}] Hiba a meglévő közös metaadat fájl olvasása közben: {e}. Új fájl létrehozása (felülírja!).")
                    df_final = df_new_segments  # Felülírás, ha hiba van az olvasáskor
            else:
                log.info(f"[{csv_path_prefix}] Új közös metaadat fájl létrehozása: {common_metadata_path}")
                df_final = df_new_segments

            df_final.to_csv(common_metadata_path, index=False, encoding='utf-8')
            log.info(
                f"[{csv_path_prefix}] Metaadatok sikeresen elmentve/frissítve ide: {common_metadata_path} (Összesen {len(df_final)} szegmens)")
        else:
            log.warning(f"[{csv_path_prefix}] Nem generálódtak új metaadatok ehhez a videóhoz.")

    except Exception as e:
        log.error(f"[{csv_path_prefix}] Hiba a feldolgozás során: {e}", exc_info=True)
    finally:
        log.info(f"[{csv_path_prefix}] Erőforrások felszabadítása...")
        if 'model' in locals() and model is not None: del model
        if 'align_model' in locals() and align_model is not None: del align_model
        if 'metadata_align' in locals() and metadata_align is not None: del metadata_align
        if 'diarization_pipeline' in locals() and diarization_pipeline is not None: del diarization_pipeline
        if 'audio' in locals() and audio is not None: del audio
        if 'full_audio_pydub' in locals() and full_audio_pydub is not None: del full_audio_pydub
        if 'result' in locals() and result is not None: del result
        if 'whisper_segments_list' in locals() and whisper_segments_list is not None: del whisper_segments_list
        gc.collect()
        if device == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception as cuda_err:
                log.warning(f"[{csv_path_prefix}] CUDA cache ürítése sikertelen: {cuda_err}")
        log.info(f"[{csv_path_prefix}] Memória felszabadítva.")


# Ez a rész csak akkor fut le, ha a szkriptet közvetlenül indítják,
# Streamlitből importálva nem.
if __name__ == "__main__":
    log.info("=== Transzkripciós és Szegmentáló Szkript Közvetlen Indítása (Teszteléshez) ===")
    # Itt kellene egy teszt konfigurációt létrehozni, ha közvetlenül akarod futtatni
    # Például:
    # test_config_data = {
    # 'output_base_dir': 'output/test_video_output',
    # 'output_raw_audio_filename': 'audio_raw.wav',
    # 'output_clean_audio_filename': 'audio_clean.wav',
    # 'output_segments_dirname': 'segments',
    # 'output_metadata_file_absolute_path': 'output/test_metadata.csv', # Közös CSV
    # 'segments_relative_path_prefix': 'test_video_output', # Videó neve
    # 'language': 'hu', 'whisper_model_size': 'tiny', 'min_segment_duration_ms': 2000,
    # 'max_segment_duration_ms': 10000, 'use_demucs': False, 'use_diarization': False,
    # 'huggingface_token': None, 'confidence_threshold': 0.5, 'device': 'cpu',
    # 'output_dir_structure': 'flat'
    # }
    # # Győződj meg róla, hogy a 'output/test_video_output/audio_raw.wav' létezik a teszthez
    # os.makedirs(test_config_data['output_base_dir'], exist_ok=True)
    # if not os.path.exists(os.path.join(test_config_data['output_base_dir'], test_config_data['output_raw_audio_filename'])):
    #     log.error("Közvetlen futtatáshoz hozz létre egy dummy audio fájlt: output/test_video_output/audio_raw.wav")
    # else:
    #     transcribe_and_segment(test_config_data)

    log.warning(
        "Ez a szkript elsősorban az app.py-ból való használatra készült. Közvetlen futtatáshoz konfiguráció szükséges.")