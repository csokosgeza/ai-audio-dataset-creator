import os
import yaml
import subprocess
import sys
import logging

# === Logger beállítása ===
# Egyszerűbb logolás a konzolra
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler()])
log = logging.getLogger(__name__) # Jobb gyakorlat a logger névvel ellátása

# === Konfiguráció betöltése ===
def load_config(config_path='config.yaml'):
    """Betölti a konfigurációs fájlt."""
    if not os.path.exists(config_path):
        log.error(f"Hiba: A konfigurációs fájl nem található: {config_path}")
        print(f"\nHIBA: A konfigurációs fájl nem található itt: {os.path.abspath(config_path)}")
        print("Győződj meg róla, hogy a config.yaml ugyanabban a mappában van, ahonnan a szkriptet futtatod.")
        sys.exit(1)
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        log.info(f"Konfiguráció betöltve innen: {config_path}")
        return config
    except yaml.YAMLError as e:
        log.error(f"Hiba a konfigurációs fájl olvasása közben: {e}")
        sys.exit(1)
    except Exception as e:
        log.error(f"Ismeretlen hiba a konfiguráció betöltésekor: {e}")
        sys.exit(1)

# === FFmpeg elérhetőségének ellenőrzése ===
def check_ffmpeg():
    """Ellenőrzi, hogy az ffmpeg parancs elérhető-e."""
    try:
        subprocess.run(['ffmpeg', '-version'], check=True, capture_output=True)
        log.info("FFmpeg telepítve és elérhető.")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        log.error("FFmpeg nem található vagy nem futtatható.")
        print("\nHIBA: Az FFmpeg nem található vagy nem futtatható.")
        print("Kérlek, telepítsd az FFmpeg-et (https://ffmpeg.org/download.html) és győződj meg róla,")
        print("hogy hozzá van adva a rendszered PATH környezeti változójához.")
        return False
    except Exception as e:
        log.error(f"Váratlan hiba az FFmpeg ellenőrzésekor: {e}")
        return False


# === Fő funkció ===
def process_media_file(config):
    """
    Kinyeri az audio sávot egy videófájlból vagy konvertál egy audiofájlt
    FFmpeg segítségével a megadott mintavételezési frekvenciával (mono, WAV).
    """
    if not check_ffmpeg():
        return None # Kilépés, ha az FFmpeg nem elérhető

    input_media_path = config.get('input_path') # Változás: input_video -> input_path
    output_base_dir = config.get('output_base_dir', 'output')
    raw_audio_filename = config.get('output_raw_audio_filename', 'audio_raw.wav')
    target_sr = config.get('target_sample_rate', 24000)

    # Bemeneti útvonal ellenőrzése (lehet relatív vagy abszolút)
    if not os.path.isabs(input_media_path):
        input_media_path = os.path.abspath(input_media_path) # Relatív utat abszolúttá alakít

    if not input_media_path or not os.path.exists(input_media_path):
        log.error(f"Hiba: A bemeneti médiafájl nem található vagy nincs megadva: {input_media_path}")
        print(f"\nHIBA: A bemeneti médiafájl nem található: {input_media_path}")
        print("Ellenőrizd az elérési utat a config.yaml fájlban.")
        return None

    # Kimeneti mappa létrehozása, ha nem létezik
    try:
        os.makedirs(output_base_dir, exist_ok=True)
        output_audio_path = os.path.join(output_base_dir, raw_audio_filename)
        log.info(f"Kimeneti mappa ellenőrizve/létrehozva: {os.path.abspath(output_base_dir)}")
    except OSError as e:
        log.error(f"Hiba a kimeneti mappa létrehozásakor ({output_base_dir}): {e}")
        print(f"\nHIBA: Nem sikerült létrehozni a kimeneti mappát: {output_base_dir}")
        return None


    log.info(f"Audio előkészítése a következőből: {input_media_path}")
    log.info(f"Cél audio fájl: {output_audio_path}")
    log.info(f"Cél mintavételezési frekvencia: {target_sr} Hz, Csatornák: 1 (mono)")

    # FFmpeg parancs összeállítása
    ffmpeg_command = [
        'ffmpeg',
        '-i', input_video_path,
        '-vn',
        '-acodec', 'pcm_s16le',
        '-ar', str(target_sr),
        '-ac', '1',
        '-y', # Felülírás kérdés nélkül
        '-hide_banner', # Kevesebb log a konzolra
        '-loglevel', 'error', # Csak a hibákat logolja az ffmpeg (de a stderr-t még elkapjuk)
        output_audio_path
    ]

    try:
        log.info(f"FFmpeg parancs futtatása: {' '.join(ffmpeg_command)}")
        # FFmpeg futtatása alfolyamatként
        process = subprocess.run(ffmpeg_command, check=True, capture_output=True, text=False) # text=False, hogy a stderr bytes maradjon
        log.info("FFmpeg sikeresen lefutott.")
        # A kimenet általában a stderr-re kerül, még sikeres futás esetén is
        if process.stderr:
             try:
                 # Próbáljuk meg dekódolni a rendszer alapértelmezett kódolásával, hibák figyelmen kívül hagyásával
                 stderr_output = process.stderr.decode(sys.getdefaultencoding(), errors='ignore')
                 log.debug(f"FFmpeg kimenet (stderr):\n{stderr_output}")
             except Exception as decode_err:
                 log.warning(f"Nem sikerült dekódolni az FFmpeg stderr kimenetét: {decode_err}")
                 log.debug(f"FFmpeg nyers kimenet (stderr): {process.stderr}")

        log.info(f"Nyers audio sikeresen elmentve ide: {os.path.abspath(output_audio_path)}")
        return output_audio_path
    except subprocess.CalledProcessError as e:
        log.error(f"Hiba történt az FFmpeg futtatása közben:")
        log.error(f"Parancs: {' '.join(e.cmd)}")
        log.error(f"Visszatérési kód: {e.returncode}")
        # Próbáljuk meg kiírni a hibát
        try:
            stderr_output = e.stderr.decode(sys.getdefaultencoding(), errors='ignore')
        except AttributeError:
             stderr_output = str(e.stderr) # Ha már string (kevésbé valószínű)
        except Exception as decode_err:
            log.warning(f"Nem sikerült dekódolni az FFmpeg stderr hibaüzenetét: {decode_err}")
            stderr_output = str(e.stderr) # Nyers bytes kiírása

        log.error(f"FFmpeg hibaüzenet (stderr):\n{stderr_output}")
        print("\nHIBA: Az FFmpeg nem futott le sikeresen. Ellenőrizd a fenti logokat a részletekért.")
        return None
    except Exception as e:
        log.error(f"Váratlan hiba az audio kinyerése közben: {e}")
        return None

# === Szkript belépési pontja ===
if __name__ == "__main__":
    log.info("=== Audio Előkészítő Szkript Indítása ===") # Változás
    config = load_config()
    if config:
        # A 'config' szótárban az 'input_video' kulcsot 'input_path'-ra kell cserélni
        if 'input_video' in config:
            config['input_path'] = config.pop('input_video')
            
        prepared_audio_path = process_media_file(config) # Változás
        if prepared_audio_path:
            log.info(f"=== Audio előkészítés sikeresen befejezve: {prepared_audio_path} ===") # Változás
        else:
            log.error("=== Audio előkészítési folyamat sikertelen. ===") # Változás
    else:
         log.error("=== Konfiguráció betöltése sikertelen, a szkript leáll. ===")
