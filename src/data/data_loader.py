# src/data/data_loader.py

import os
import shutil
import logging
import logging.config

# Mengatur logging segera setelah impor
logging.config.fileConfig('configs/logging.conf')

# Dapatkan logger untuk modul ini
logger = logging.getLogger('src.data.data_loader')

def organize_images(raw_dir: str, output_dir: str) -> None:
    """
    Memindahkan file gambar .png dari struktur direktori awal ke struktur direktori baru yang terorganisir.

    Parameters
    ----------
    raw_dir : str
        Jalur ke direktori data mentah yang berisi subdirektori grup ('data/raw/').
    output_dir : str
        Jalur ke direktori keluaran untuk menyimpan gambar yang telah diorganisir (misalnya, 'data/processed/images_per_part/').

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        Jika direktori input tidak ditemukan.
    Exception
        Jika terjadi kesalahan selama proses pemindahan file.
    """
    try:
        # Memeriksa apakah direktori input ada
        if not os.path.exists(raw_dir):
            raise FileNotFoundError(f"Direktori {raw_dir} tidak ditemukan.")

        # Membuat struktur direktori output jika belum ada
        sides = ['Left', 'Right']
        groups = {
            'Control Group': 'CG',
            'DM Group': 'DM'
        }

        for side in sides:
            for group_short in groups.values():
                dest_dir = os.path.join(output_dir, side, f"{group_short} {side}")
                os.makedirs(dest_dir, exist_ok=True)

        # Iterasi melalui direktori grup di direktori raw
        for group_name, group_short in groups.items():
            group_path = os.path.join(raw_dir, group_name)
            if not os.path.exists(group_path):
                logger.warning(f"Direktori {group_path} tidak ditemukan, melewatkan grup ini.")
                continue

            # Iterasi melalui subdirektori sample (misalnya, 'CG001_M')
            for sample_name in os.listdir(group_path):
                sample_path = os.path.join(group_path, sample_name)
                if not os.path.isdir(sample_path):
                    continue

                # Iterasi melalui file dalam sample
                for file_name in os.listdir(sample_path):
                    if file_name.endswith('.png'):
                        file_path = os.path.join(sample_path, file_name)

                        # Menentukan sisi (Left/Right) berdasarkan nama file
                        if '_L.png' in file_name:
                            side = 'Left'
                        elif '_R.png' in file_name:
                            side = 'Right'
                        else:
                            logger.warning(f"Nama file {file_name} tidak sesuai format, melewatkan file ini.")
                            continue

                        # Menentukan direktori tujuan
                        dest_dir = os.path.join(output_dir, side, f"{group_short} {side}")
                        os.makedirs(dest_dir, exist_ok=True)

                        # Menentukan jalur file tujuan
                        dest_path = os.path.join(dest_dir, file_name)

                        # Memindahkan file
                        shutil.copy2(file_path, dest_path)
                        logger.info(f"Menyalin {file_path} ke {dest_path}")

    except Exception as e:
        logger.error(f"Terjadi kesalahan: {e}")
        raise

if __name__ == "__main__":
    # Jalur direktori input dan output
    raw_directory = './data/raw/'
    output_directory = './data/processed/images_per_part/'

    # Memanggil fungsi untuk mengorganisir gambar
    organize_images(raw_directory, output_directory)