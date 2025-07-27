import os
import random
import argparse

def hapus_data_random(folder_path, jumlah):
    if not os.path.exists(folder_path):
        print(f"❌ Folder tidak ditemukan: {folder_path}")
        return

    semua_file = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    
    if jumlah > len(semua_file):
        print(f"⚠️ Jumlah yang ingin dihapus ({jumlah}) lebih besar dari jumlah file ({len(semua_file)})")
        return

    file_terpilih = random.sample(semua_file, jumlah)

    for nama_file in file_terpilih:
        os.remove(os.path.join(folder_path, nama_file))

    print(f"✅ Berhasil menghapus {jumlah} file dari {folder_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hapus sejumlah file secara acak dari sebuah folder")
    parser.add_argument("folder", type=str, help="Path ke folder target (contoh: dataset/train/happy)")
    parser.add_argument("jumlah", type=int, help="Jumlah file yang ingin dihapus")

    args = parser.parse_args()
    hapus_data_random(args.folder, args.jumlah)
