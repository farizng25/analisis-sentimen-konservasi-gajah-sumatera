# 🐘 Analisis Sentimen Opini Publik terhadap Gajah Sumatera di YouTube

Analisis sentimen berbasis *deep learning* terhadap komentar YouTube seputar isu konservasi Gajah Sumatera, menggunakan model **IndoBERTweet** yang di-*fine-tune* untuk klasifikasi tiga kelas (Positif, Negatif, Netral).

## 📌 Latar Belakang

Gajah Sumatera merupakan subspesies gajah Asia berstatus kritis (IUCN, sejak 2011), dengan populasi liar diperkirakan hanya sekitar 1.700 ekor. Deforestasi, ekspansi perkebunan kelapa sawit, dan konflik dengan manusia menjadi ancaman utama kelestariannya. Opini publik memainkan peran penting dalam legitimasi dan efektivitas kebijakan konservasi, sehingga proyek ini berupaya mengukur sentimen masyarakat melalui kolom komentar video berita YouTube.

## 🎯 Tujuan

- Menganalisis distribusi dan tren sentimen publik terhadap isu Gajah Sumatera di YouTube.
- Mengidentifikasi kata kunci utama yang memicu sentimen positif maupun negatif.
- Melakukan *fine-tuning* model IndoBERTweet untuk klasifikasi sentimen tiga kelas dan mengevaluasi performanya.
- Memberikan rekomendasi berbasis data bagi pemerintah dan lembaga konservasi.

## 🗂️ Data

- **Sumber:** Komentar pada video Kompas TV, *"Rumah Gajah Sumatera Digerus Kebun Sawit: Duduk Perkara Tesso Nilo Digeruduk Massa"*.
- **Metode pengumpulan:** *Web scraping* menggunakan library `youtube-comment-downloader` (Python).
- **Ukuran dataset:** 158 komentar unik (setelah deduplikasi dan seleksi).
- **Pelabelan:** Manual, ke dalam 3 kelas — Positif, Negatif, Netral — menggunakan antarmuka `ipywidgets` di Jupyter Notebook.
- **Distribusi kelas:** 60 Positif (38,0%), 95 Negatif (60,1%), 3 Netral (1,9%) — dataset bersifat *imbalanced*.

## ⚙️ Metodologi

1. **Preprocessing teks:** pembersihan (URL, mention, hashtag, emoji, angka, tanda baca), *case folding*, normalisasi slang/bahasa gaul YouTube, filter stopword, *stemming* (PySastrawi), dan tokenisasi.
2. **Exploratory Data Analysis (EDA):** distribusi sentimen, rata-rata *likes* per kategori, kata yang paling sering muncul, dan *word cloud* per kategori sentimen.
3. **Pemodelan:** *Fine-tuning* model pra-latih `indolem/indobertweet-base-uncased` (HuggingFace Transformers) menggunakan `AutoModelForSequenceClassification`.
4. **Evaluasi:** Perbandingan tiga konfigurasi *learning rate*, dengan pembagian data 80:20 (*stratified split*).

## 📊 Hasil Utama

### 1. Distribusi Sentimen

![Distribusi Sentimen](eda_1_distribusi.png)

Dari 158 komentar yang dikumpulkan, distribusi kelas sentimen adalah sebagai berikut:

| Kelas | Jumlah | Persentase | Rata-rata Panjang Karakter | Rata-rata Jumlah Token |
|---|---|---|---|---|
| Negatif | 95 | 60,1% | 108,7 | 10,3 |
| Positif | 60 | 38,0% | 112,4 | 10,8 |
| Netral | 3 | 1,9% | 95,3 | 9,7 |

Dominasi sentimen **Negatif** mengindikasikan tingginya keresahan publik terhadap isu deforestasi, ekspansi perkebunan sawit, dan lemahnya penegakan hukum perlindungan habitat Gajah Sumatera.

### 2. Rata-Rata Likes per Kategori Sentimen

![Rata-rata Likes](eda_2_likes_sentimen.png)

| Kategori | Rata-rata Likes |
|---|---|
| Positif | 2,0 |
| Negatif | 1,5 |
| Netral | 0,3 |

Meskipun jumlahnya lebih sedikit, komentar **positif** justru memperoleh apresiasi (*likes*) tertinggi — menunjukkan bahwa dukungan terhadap konservasi mendapat resonansi kuat dari penonton lain, meski secara volume komentar negatif lebih mendominasi.

### 3. Kata yang Paling Sering Muncul (Top 15)

![Top Words](eda_3_top_words.png)

Lima kata teratas: **"hutan"** (60x), **"sawit"** (46x), **"alam"** (37x), **"pemerintah"** (33x), **"manusia"** (30x), diikuti "gajah" dan "masyarakat" (masing-masing 25x). Pola ini mencerminkan inti perdebatan publik: konflik antara kepentingan bisnis perkebunan sawit vs. kelestarian hutan sebagai habitat gajah, serta tuntutan ketegasan pemerintah.

### 4. Word Cloud per Kategori Sentimen

![Word Cloud](eda_4_wordcloud.png)

- **Positif:** didominasi kata "alam", "gajah", "hutan", "jaga" — mencerminkan narasi dukungan konservasi dan harapan hubungan harmonis manusia-lingkungan.
- **Negatif:** didominasi kata "hutan", "pemerintah", "masyarakat", "rakyat", "warga", "sawit" — mencerminkan kritik terhadap pemerintah yang dinilai kurang tegas serta kekhawatiran atas ekspansi sawit.
- **Netral:** didominasi kata "gajah", "unjuk", "rasa", "ikut" — bersifat observasional tanpa muatan emosi kuat.

### 5. Performa Model IndoBERTweet

Tiga konfigurasi *learning rate* diuji, dengan hasil sebagai berikut:

| Learning Rate | Epoch | Save Steps | Accuracy | Recall | Precision | F1-Score |
|---|---|---|---|---|---|---|
| **5×10⁻⁸** ⭐ | 3 | 200 | **90,62%** | 90,62% | 94,79% | **91,90%** |
| 5×10⁻⁶ | 3 | 150 | 87,50% | 87,50% | 88,78% | 87,17% |
| 5×10⁻⁴ | 3 | 100 | 81,25% | 81,25% | 91,25% | 84,17% |

**Classification report** model terbaik (LR = 5×10⁻⁸):

| Kelas | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Positif | 1,00 | 0,83 | 0,91 | 12 |
| Negatif | 0,95 | 0,95 | 0,95 | 19 |
| Netral | 0,33 | 1,00 | 0,50 | 1 |
| **Weighted Avg** | **0,95** | **0,91** | **0,92** | 32 |

### 6. Confusion Matrix

![Confusion Matrix](eda_5_confusion_matrix.png)

- Kelas **Positif**: 10 dari 12 data uji diklasifikasikan benar (masing-masing 1 salah sebagai Negatif dan Netral).
- Kelas **Negatif**: 18 dari 19 data uji diklasifikasikan benar (1 salah sebagai Netral).
- Kelas **Netral**: 1 dari 1 data uji diklasifikasikan benar.

Kesalahan klasifikasi antara Positif dan Negatif dapat dipahami karena komentar yang mengkritik pemerintah namun dilatarbelakangi kepedulian konservasi kerap memiliki ambiguitas leksikal yang menyulitkan model.

## 🛠️ Tools & Libraries

`Python` · `HuggingFace Transformers` · `IndoBERTweet` · `scikit-learn` · `PySastrawi` · `ipywidgets` · `youtube-comment-downloader` · `Matplotlib/Seaborn` (visualisasi)

## 🔮 Pengembangan Selanjutnya

- Memperluas data dari berbagai video/platform terkait Gajah Sumatera.
- Membangun sistem pemantauan sentimen *real-time* sebagai *early warning system* konflik gajah–manusia.
- Menangani ketidakseimbangan kelas Netral melalui augmentasi data/*oversampling*.
Program Studi Statistika, Universitas Indonesia

---

*Proyek ini dikembangkan sebagai bagian dari mata kuliah Analisis Data Tidak Terstruktur.*
