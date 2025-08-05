KÜTÜPHANELER
-------------------------------------------------------------

pip install -r requirements.txt komutunu girerek bu app için gerekli olan bütün kütüphaneleri indirebilirsiniz.

    streamlit
    pandas
    plotly


TIKLA-ÇALIŞTIR BAT DOSYASI
-------------------------------------------------------------
Eğer bir tıkla-çalıştır dosya uzantısı kullanmak istiyorsanız.
ATKAR RUN.bat ın içindeki komut dosyasını sağ click düzenle şeçeneğinden 

@echo off

cd "C:\Users\xxx\ATKAR" #ATKAR.py dosyanızın bulunduğu lokasyonu buraya giriniz

streamlit run ATKAR.py

sonrasında bu bat dosyasının kısayolunu alıp masaüstünüze koyabilirsiniz.

CSV FORMATI
-------------------------------------------------------------
Upload edeceğiniz CSV dökümanı en başında tarih olacak şekilde ilk satırda başlıklar ve sonraki satırlarda veri olmalı. CSV deki hücre formatı GENERAL şeçilmeli.
Ondalık ayıracının ne olduğu önemli değil kendi tercihinize göre programdan seçebilirsiniz.
