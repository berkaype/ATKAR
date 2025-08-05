import streamlit as st
import pandas as pd
from datetime import datetime # Tarih girişleri için datetime modülünü içeri aktar
import plotly.graph_objects as go # Plotly'nin düşük seviye grafik nesnelerini içeri aktar

# Sayfa düzenini geniş olarak ayarla
st.set_page_config(layout="wide", 
                   page_title="ATKAR - Atıksu Arıtma Tesisleri Karşılaştırma Platformu",
                   page_icon="💧" # Buraya istediğiniz emojiyi veya bir resim URL'sini ekleyebilirsiniz
                   )

st.title("Atıksu Arıtma Tesisleri Karşılaştırma Analizi")

st.info("Sol üstteki 'Browse files' veya 'Gözat' butonuyla **CSV dosyanızı seçip yükleyin**.")

# Dosya yükleme arayüzü
uploaded_file = st.file_uploader("CSV dosyasını yükleyin", type=["csv"])
if uploaded_file:
    # CSV dosyasını okumak için farklı kodlamaları ve ayırıcıları dener
    encodings = ['utf-8', 'latin1', 'cp1254']
    separators = [',', ';', '\t']
    df = None
    for enc in encodings:
        for sep in separators:
            try:
                # Dosyanın başına dönerek her denemede yeniden okunmasını sağla
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding=enc, sep=sep, engine='python')
                df.columns = df.columns.map(str).str.strip() # Sütun adlarındaki boşlukları temizle
                break # Başarılı olursa, ayırıcı döngüsünden çık
            except Exception:
                df = None
        if df is not None:
            break # Başarılı olursa, kodlama döngüsünden çık

    if df is None:
        st.error("Dosya okunamadı. Lütfen dosyanızın CSV formatında olduğundan ve doğru ayırıcıyı kullandığından emin olun.")
        st.stop()

    # İlk sütunun tarih sütunu olduğunu varsay
    date_col = df.columns[0]
    
    try:
        # Tarih sütununu standart bir formata dönüştür
        df['Tarih'] = pd.to_datetime(df[date_col], dayfirst=True, errors='coerce')
    except Exception as e:
        st.error(f"Tarih kolonu parse edilemedi. Lütfen ilk kolonda geçerli bir tarih formatı olduğundan emin olun: {e}")
        st.stop()

    # Tarih formatı dönüştürülemeyen satırları kaldır
    df.dropna(subset=['Tarih'], inplace=True)
    df.set_index('Tarih', inplace=True)

    # Kenar çubuğunda tarih aralığı seçimi
    min_date = df.index.min().date() if not df.empty else datetime.now().date()
    max_date = df.index.max().date() if not df.empty else datetime.now().date()

    st.sidebar.subheader("Filtreleme Seçenekleri")
    start_date = st.sidebar.date_input("Başlangıç Tarihi", value=min_date, min_value=min_date, max_value=max_date)
    end_date = st.sidebar.date_input("Bitiş Tarihi", value=max_date, min_value=min_date, max_value=max_date)

    if start_date > end_date:
        st.error("Başlangıç tarihi bitiş tarihinden sonra olamaz. Lütfen tarihleri kontrol edin.")
        st.stop()
    
    # Seçilen tarih aralığına göre DataFrame'i filtrele
    df = df.loc[start_date.strftime('%Y-%m-%d'):end_date.strftime('%Y-%m-%d')]

    if df.empty:
        st.warning("Seçilen tarih aralığında veri bulunamadı. Lütfen tarih aralığını genişletin.")
        st.stop()

    # Ondalık ayırıcı seçimi
    decimal_separator = st.sidebar.radio(
        "Ondalık Ayırıcıyı Seçin",
        (".", ","),
        help="CSV dosyanızdaki ondalık sayıların ayırıcısını seçin (örn. 1.23 için '.' veya 1,23 için ',')."
    )

    # Orijinal tarih sütunu dışındaki tüm sütunları veri sütunları olarak al
    data_cols = [c for c in df.columns if c != date_col]

    # Sütun başlıklarından benzersiz tesis adlarını çıkar
    plants = sorted({col.split()[0] for col in data_cols})
    
    # Çoklu tesis seçimi
    selected_plants = st.multiselect("Karşılaştırılacak Tesisleri Seçin", plants)

    # Seçilen tesislere ait parametreleri filtrele
    available_params_for_selection = []
    if selected_plants:
        for plant_name in selected_plants:
            available_params_for_selection.extend([c for c in data_cols if c.startswith(plant_name)])
    
    available_params_for_selection = sorted(list(set(available_params_for_selection)))

    # Kullanıcının analiz edeceği parametreleri seçmesi
    selected_params = st.multiselect("Grafikte Gösterilecek Parametreleri Seçin", available_params_for_selection)

    # Her parametre için grafik tipi, Y ekseni ve opasite seçimi için konteynerler
    chart_types_for_params = {}
    yaxis_assignments = {}
    opacity_values = {}

    if selected_params:
        st.subheader("Grafik Özelleştirme")
        col1, col2, col3 = st.columns(3)  # 3 sütun yapıyoruz
        
        with col1:
            st.markdown("##### Grafik Tipi")
            for param in selected_params:
                chart_type = st.selectbox(
                    f"'{param}' tipi",
                    ("Çizgi (Line)", "Çubuk (Bar)", "Nokta (Scatter)"),
                    key=f"chart_type_{param}"
                )
                chart_types_for_params[param] = chart_type
        
        with col2:
            st.markdown("##### Y Ekseni Ataması")
            for param in selected_params:
                axis_choice = st.selectbox(
                    f"'{param}' ekseni",
                    ('Birincil Eksen (Sol)', 'İkincil Eksen (Sağ)', 'Üçüncül Eksen (Sağ)'),
                    key=f"yaxis_{param}",
                    help="Farklı ölçekteki verileri aynı grafikte göstermek için kullanılır."
                )
                if 'İkincil' in axis_choice:
                    yaxis_assignments[param] = 'y2'
                elif 'Üçüncül' in axis_choice:
                    yaxis_assignments[param] = 'y3'
                else:
                    yaxis_assignments[param] = 'y'
        
        with col3:
            st.markdown("##### Opasite Ayarı")
            for param in selected_params:
                opacity = st.slider(
                    f"'{param}' opasite",
                    min_value=0.1,
                    max_value=1.0,
                    value=1.0,
                    step=0.1,
                    key=f"opacity_{param}",
                    help="Grafiğin şeffaflığını ayarlar (0.1=çok şeffaf, 1.0=opak)"
                )
                opacity_values[param] = opacity

    # Analiz için zaman dilimini seç
    granularity = st.sidebar.radio("Zaman Dilimi", ("Günlük", "Aylık", "Mevsimlik", "Yıllık"))

    if selected_params:
        # Seçilen parametreleri içeren DataFrame'i oluştur
        df_sel = df[selected_params].copy()
        
        # Seçilen parametre sütunlarını sayısal türe dönüştür
        for col in df_sel.columns:
            if decimal_separator == ',':
                df_sel[col] = df_sel[col].astype(str).str.replace(',', '.', regex=False)
            df_sel[col] = pd.to_numeric(df_sel[col], errors='coerce')

        # Boş değerleri işleme stratejisi seçimi
        missing_data_strategy = st.sidebar.radio(
            "Boş (NaN) Değerleri Nasıl İşleyelim?",
            ("Enterpole Et (Değerleri Birleştir)", "Boş Bırak (Grafikte Gösterme)")
        )

        if missing_data_strategy == "Enterpole Et (Değerleri Birleştir)":
            df_sel.interpolate(method='linear', limit_direction='both', inplace=True)
            df_sel.fillna(0, inplace=True)

        df_sel.dropna(axis=1, how='all', inplace=True)
        
        if df_sel.empty:
            st.warning("Seçilen filtreler sonucunda analiz edilecek geçerli veri bulunamadı.")
            st.stop()

        # Seçilen zaman dilimine göre veriyi yeniden örnekle (ortalama alarak)
        if granularity == "Günlük":
            res = df_sel.resample('D').mean()
        elif granularity == "Aylık":
            res = df_sel.resample('ME').mean()
            res.index = res.index.strftime('%Y-%m') # X eksenini daha okunaklı yap
        elif granularity == "Yıllık":
            res = df_sel.resample('YE').mean()
            res.index = res.index.strftime('%Y') # X eksenini daha okunaklı yap
        else: # Mevsimlik
            seasons = {
                12: 'Kış', 1: 'Kış', 2: 'Kış',
                3: 'İlkbahar', 4: 'İlkbahar', 5: 'İlkbahar',
                6: 'Yaz', 7: 'Yaz', 8: 'Yaz',
                9: 'Sonbahar', 10: 'Sonbahar', 11: 'Sonbahar'
            }
            df_sel_season = df_sel.copy() 
            df_sel_season['Season'] = df_sel_season.index.month.map(seasons)
            res = df_sel_season.groupby('Season').mean()
            # Mevsimleri doğru sırada göstermek için sırala
            season_order = ['İlkbahar', 'Yaz', 'Sonbahar', 'Kış']
            res = res.reindex(season_order).dropna()

        st.subheader("Özet Tablo")
        st.dataframe(res.style.format("{:.2f}"))

        st.subheader("Grafik")
        
        fig = go.Figure()
        used_yaxes = set()

        # Her parametre için seçilen grafik tipi, eksen ve opasite değerine göre izleri (trace) ekle
        for param in res.columns:
            chart_type = chart_types_for_params.get(param, "Çizgi (Line)")
            target_yaxis = yaxis_assignments.get(param, 'y')
            opacity = opacity_values.get(param, 1.0)  # Varsayılan opasite 1.0
            used_yaxes.add(target_yaxis)

            trace_args = {
                'x': res.index, 
                'y': res[param], 
                'name': param, 
                'yaxis': target_yaxis,
                'opacity': opacity  # Opasite değerini ekliyoruz
            }

            if chart_type == "Çizgi (Line)":
                fig.add_trace(go.Scatter(mode='lines+markers', **trace_args))
            elif chart_type == "Çubuk (Bar)":
                fig.add_trace(go.Bar(**trace_args))
            elif chart_type == "Nokta (Scatter)":
                fig.add_trace(go.Scatter(mode='markers', **trace_args))

        # Grafik düzenini dinamik olarak oluştur
        layout_options = {
            'title_text': "Seçilen Parametrelerin Zaman Serisi Grafiği",
            'xaxis_title': "Tarih" if granularity != "Mevsimlik" else "Mevsim",
            'hovermode': "x unified",
            'legend': {'orientation': 'h', 'yanchor': 'bottom', 'y': 1.02, 'xanchor': 'right', 'x': 1},
            'yaxis': {'title': {'text': 'Birincil Eksen Değeri'}, 'automargin': True},
            'xaxis': {'automargin': True}
        }

        # İkincil ve üçüncül eksenleri sadece kullanılıyorlarsa ekle
        if 'y2' in used_yaxes:
            layout_options['yaxis2'] = {
                'title': {
                    'text': 'İkincil Eksen Değeri',
                    'font': {'color': '#E55451'}
                },
                'overlaying': 'y',
                'side': 'right',
                'anchor': 'free',  # Serbest konumlandırma
                'position': 1.0,   # Normal sağ pozisyon
                'tickfont': {'color': '#E55451'},
                'automargin': True
            }
        
        if 'y3' in used_yaxes:
            layout_options['yaxis3'] = {
                'title': {
                    'text': 'Üçüncül Eksen Değeri',
                    'font': {'color': '#347C17'}
                },
                'overlaying': 'y',
                'side': 'right',
                'anchor': 'free',  # Serbest konumlandırma
                'position': 1.0,
                'shift': 80, # Daha sağa kaydır (8% padding)
                'tickfont': {'color': '#347C17'},
                'automargin': True
            }
        
        fig.update_layout(**layout_options)
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Lütfen analiz etmek için en az bir tesis ve bir parametre seçin.")