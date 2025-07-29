import os
import random
import argparse

def hapus_data_ke_target(folder_path, target_jumlah):
    if not os.path.exists(folder_path):
        print(f"❌ Folder tidak ditemukan: {folder_path}")
        return

    semua_file = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    jumlah_sekarang = len(semua_file)

    if jumlah_sekarang <= target_jumlah:
        print(f"📦 Jumlah file saat ini ({jumlah_sekarang}) sudah kurang dari atau sama dengan target ({target_jumlah}). Tidak ada file yang dihapus.")
        return

    jumlah_dihapus = jumlah_sekarang - target_jumlah
    file_terpilih = random.sample(semua_file, jumlah_dihapus)

    for nama_file in file_terpilih:
        os.remove(os.path.join(folder_path, nama_file))

    print(f"✅ Berhasil menghapus {jumlah_dihapus} file agar jumlah total menjadi {target_jumlah} di folder {folder_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hapus file secara acak dari folder agar jumlahnya sesuai target")
    parser.add_argument("folder", type=str, help="Path ke folder target (contoh: dataset/train/happy)")
    parser.add_argument("target", type=int, help="Target jumlah file akhir setelah dihapus")

    args = parser.parse_args()
    hapus_data_ke_target(args.folder, args.target)
