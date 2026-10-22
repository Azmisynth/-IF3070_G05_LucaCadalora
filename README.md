# LucaCadalora

Tugas Besar IF3070 Dasar Inteligensi Artifisial — optimasi penempatan kendaraan pada geladak kapal menggunakan Hill Climbing, Simulated Annealing, dan Genetic Algorithm.

## Anggota

| Nama | NIM |
| --- | --- |
| Muhammad Syauqi Azizi | 18224046 |
| Rafi Azmi Zafarani | 18224082 |
| Athar Ramadhana Dermawan | 18224104 |

## Instalasi

Gunakan Python 3.10 atau lebih baru. Jalankan dari direktori utama repositori:

```bash
python -m pip install -r requirements.txt
```

## Menjalankan Program

Menu interaktif:

```bash
python -m src.main
```

Pilihan algoritma:

```bash
python -m src.main --algorithm hc --variant steepest
python -m src.main --algorithm sa
python -m src.main --algorithm ga --population 30 --generations 100
```

Multikapal dan eksperimen:

```bash
python -m src.main --input input/multi_ship_example.json --algorithm hc
python -m src.main --experiments
```

Input tersedia pada folder `input/`. Hasil pencarian, gambar, dan grafik disimpan pada folder `output/`.

## Pengujian

```bash
python -m unittest discover -s tests -q
```
