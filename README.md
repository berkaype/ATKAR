KÜTÜPHANELER
-------------------------------------------------------------

pip install -r requirements.txt komutunu girerek bu app için gerekli olan bütün kütüphaneleri indirebilirsiniz.

    numpy==2.3.2
    pandas==2.3.2
    plotly==6.2.0
    scikit_learn==1.7.1
    streamlit==1.47.1
    tensorflow==2.20.0


TIKLA-ÇALIŞTIR BAT DOSYASI
-------------------------------------------------------------
Eğer bir tıkla-çalıştır dosya uzantısı kullanmak istiyorsanız.
ATKAR RUN.bat ın içindeki komut dosyasını sağ click düzenle şeçeneğinden 

@echo off

cd "C:\Users\xxx\ATKAR" #ATKAR.py dosyanızın bulunduğu lokasyonu buraya giriniz

streamlit run ATKAR.py

sonrasında bu bat dosyasının kısayolunu alıp masaüstünüze koyabilirsiniz.

XLSX FORMATI
-------------------------------------------------------------
Uygulama, "Veribankası" adında bir Excel sayfası (tab) içindeki verileri okuyacak şekilde tasarlanmıştır.

- **1. Satır:** Parametrelerin birimleri (ilk hücre boş bırakılmalı).
- **2. Satır:** Başlıklar (ilk sütun tarih, diğerleri parametre adları).
- **3. Satır ve sonrası:** Tarih ve sayısal veri değerleri.
- Ondalık ayıracının ne olduğu önemli değildir, program içinden seçilebilir.
