import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# --- Sayfa Yapılandırması ---
st.set_page_config(layout="wide",
                   page_title="ATKAR - Atıksu Arıtma Platformu",
                   page_icon="💧"
                   )

# --- Ana Başlık ---
st.title("Atıksu Arıtma Tesisleri Karşılaştırma ve Tahmin Platformu")

# --- Fonksiyonlar (Model Eğitimi ve Tahmin) ---
# Model eğitimi ve kaynakları önbelleğe alarak performansı artırıyoruz.
# Veri seti, parametre adı ve time_step değişmediği sürece bu fonksiyon tekrar çalışmaz.
@st.cache_resource
def train_and_predict_lstm(_df_cleaned, param_name, time_step=60):
    """
    Belirtilen parametre için bir LSTM modeli eğitir ve tahmin yapar.
    Modeli ve ölçekleyiciyi döndürür.
    """
    # 1. Veri Hazırlığı
    series = _df_cleaned[param_name].dropna()
    if len(series) < time_step + 10: # Modelin eğitilmesi için minimum veri
        return None, None, f"'{param_name}' parametresi için yeterli veri bulunmuyor (en az {time_step + 10} gün gerekli)."

    # Veriyi 0-1 arasına ölçekle
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(series.values.reshape(-1, 1))

    # Eğitim verisi oluştur
    X, y = [], []
    for i in range(len(scaled_data) - time_step):
        X.append(scaled_data[i:(i + time_step), 0])
        y.append(scaled_data[i + time_step, 0])
    
    X_train, y_train = np.array(X), np.array(y)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # 2. LSTM Modeli Oluşturma ve Eğitme
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(time_step, 1)),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, batch_size=1, epochs=10, verbose=0) # verbose=0 arayüzü temiz tutar

    # Modelin eğitim verisi üzerindeki performansını hesapla
    train_predict = model.predict(X_train)
    train_predict = scaler.inverse_transform(train_predict)
    y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1))
    rmse = np.sqrt(mean_squared_error(y_train_inv, train_predict))

    return model, scaler, rmse

def make_future_forecast(model, scaler, last_data, time_step, forecast_days):
    """
    Eğitilmiş bir modeli kullanarak geleceğe yönelik tahminler yapar.
    """
    last_sequence_scaled = scaler.transform(last_data.values.reshape(-1, 1))
    forecast_list = []

    for _ in range(forecast_days):
        current_batch = np.reshape(last_sequence_scaled, (1, time_step, 1))
        prediction = model.predict(current_batch, verbose=0)
        forecast_list.append(prediction[0, 0])
        last_sequence_scaled = np.append(last_sequence_scaled[1:], prediction, axis=0)
    
    forecast_values = scaler.inverse_transform(np.array(forecast_list).reshape(-1, 1)).flatten()
    return forecast_values


# --- Veri Yükleme (Uygulama Genelinde) ---
st.info("Başlamak için lütfen CSV formatındaki zaman serisi verilerinizi yükleyin.")
uploaded_file = st.file_uploader("CSV dosyasını yükleyin", type=["csv"], key="data_uploader")

if uploaded_file:
    @st.cache_data
    def load_data(file):
        # ... (Önceki koddaki veri yükleme fonksiyonu aynı) ...
        encodings = ['utf-8', 'latin1', 'cp1254']; separators = [',', ';', '\t']; df = None
        for enc in encodings:
            for sep in separators:
                try: file.seek(0); df = pd.read_csv(file, encoding=enc, sep=sep, engine='python'); df.columns = df.columns.map(str).str.strip(); return df
                except Exception: continue
        return None

    df_initial = load_data(uploaded_file)

    if df_initial is None:
        st.error("Dosya okunamadı. Lütfen dosyanızın CSV formatında olduğundan ve doğru ayırıcıyı kullandığından emin olun.")
        st.stop()

    try:
        date_col = df_initial.columns[0]
        df_initial['Tarih'] = pd.to_datetime(df_initial[date_col], dayfirst=True, errors='coerce')
        df_initial.dropna(subset=['Tarih'], inplace=True)
        df_initial.set_index('Tarih', inplace=True)
        df_initial.sort_index(inplace=True)
    except Exception as e:
        st.error(f"Tarih sütunu işlenemedi: {e}"); st.stop()

    decimal_separator = st.sidebar.radio("Verideki Ondalık Ayırıcı", (",", "."))
    data_cols = [c for c in df_initial.columns if c != date_col]
    df_cleaned = df_initial[data_cols].copy()
    for col in df_cleaned.columns:
        if decimal_separator == ',':
            df_cleaned[col] = df_cleaned[col].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
        df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')

    missing_data_strategy = st.sidebar.radio("Boş (NaN) Değerleri Nasıl İşleyelim?", ("Enterpole Et (Doğrusal Doldur)", "Boş Bırak"))
    if missing_data_strategy == "Enterpole Et (Doğrusal Doldur)":
        df_cleaned.interpolate(method='linear', limit_direction='both', inplace=True)

    # --- Sekmeleri Oluştur ---
    tab1, tab2 = st.tabs(["Veri Görselleştirme ve Karşılaştırma", "Zaman Serisi Tahmini (LSTM)"])

    # ============================ SEÇME 1: GÖRSELLEŞTİRME ============================
    with tab1:
        st.header("Tesis Verilerini Karşılaştırma")
        min_date, max_date = df_cleaned.index.min().date(), df_cleaned.index.max().date()

        st.sidebar.subheader("Görselleştirme Filtreleri")
        start_date = st.sidebar.date_input("Başlangıç Tarihi", value=min_date, min_value=min_date, max_value=max_date, key="start_date_tab1")
        end_date = st.sidebar.date_input("Bitiş Tarihi", value=max_date, min_value=min_date, max_value=max_date, key="end_date_tab1")

        if start_date > end_date:
            st.error("Başlangıç tarihi bitiş tarihinden sonra olamaz."); st.stop()
        
        df = df_cleaned.loc[start_date.strftime('%Y-%m-%d'):end_date.strftime('%Y-%m-%d')]

        plants = sorted({col.split()[0] for col in data_cols})
        selected_plants = st.multiselect("Karşılaştırılacak Tesisleri Seçin", plants, key="plants_tab1")
        
        available_params = sorted([c for c in data_cols if any(c.startswith(p) for p in selected_plants)])
        selected_params = st.multiselect("Grafikte Gösterilecek Parametreleri Seçin", available_params, key="params_tab1")
        
        # --- Orijinal Detaylı Grafik Kodunuz Buraya Geri Geldi ---
        if selected_params:
            st.subheader("Grafik Özelleştirme")
            chart_types_for_params, yaxis_assignments, opacity_values = {}, {}, {}
            col1, col2, col3 = st.columns(3)
            with col1:
                # ... (Kısalık adına gizlendi, orijinal kodunuzdaki grafik tipi seçimi)
                 for param in selected_params: chart_types_for_params[param] = st.selectbox(f"'{param}' tipi", ("Çizgi (Line)", "Çubuk (Bar)", "Nokta (Scatter)"), key=f"chart_type_{param}")
            with col2:
                # ... (Kısalık adına gizlendi, orijinal kodunuzdaki Y ekseni ataması)
                for param in selected_params: 
                    axis_choice = st.selectbox(f"'{param}' ekseni", ('Birincil Eksen (Sol)', 'İkincil Eksen (Sağ)'), key=f"yaxis_{param}")
                    yaxis_assignments[param] = 'y2' if 'İkincil' in axis_choice else 'y'
            with col3:
                # ... (Kısalık adına gizlendi, orijinal kodunuzdaki opasite ayarı)
                for param in selected_params: opacity_values[param] = st.slider(f"'{param}' opasite", 0.1, 1.0, 1.0, 0.1, key=f"opacity_{param}")
            
            st.subheader("Grafik")
            fig = go.Figure()
            # ... (Orijinal kodunuzdaki Plotly figür oluşturma ve çizim mantığı)
            for param in selected_params:
                chart_type, target_yaxis, opacity = chart_types_for_params.get(param, "Çizgi (Line)"), yaxis_assignments.get(param, 'y'), opacity_values.get(param, 1.0)
                trace_args = {'x': df.index, 'y': df[param], 'name': param, 'yaxis': target_yaxis, 'opacity': opacity}
                if chart_type == "Çizgi (Line)": fig.add_trace(go.Scatter(mode='lines+markers', **trace_args))
                elif chart_type == "Çubuk (Bar)": fig.add_trace(go.Bar(**trace_args))
                elif chart_type == "Nokta (Scatter)": fig.add_trace(go.Scatter(mode='markers', **trace_args))
            fig.update_layout(title_text="Zaman Serisi Grafiği", xaxis_title="Tarih", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Grafiği görüntülemek için lütfen en az bir tesis ve bir parametre seçin.")

    # ============================ SEÇME 2: LSTM TAHMİNİ ============================
    with tab2:
        st.header("LSTM ile Geleceğe Yönelik Parametre Tahmini")
        st.markdown("Bu modül, seçtiğiniz bir parametrenin geçmiş verilerini kullanarak gelecekteki değerlerini tahmin etmek için bir Uzun Kısa Süreli Bellek (LSTM) sinir ağı modeli kullanır.")

        plants_lstm = sorted({col.split()[0] for col in data_cols})
        selected_plant_lstm = st.selectbox("Tahmin Edilecek Tesisi Seçin", plants_lstm, key="plant_lstm")

        if selected_plant_lstm:
            params_for_plant = sorted([c for c in data_cols if c.startswith(selected_plant_lstm)])
            selected_param_lstm = st.selectbox("Tahmin Edilecek Parametreyi Seçin", params_for_plant, key="param_lstm")
            forecast_days = st.slider("Gelecek Kaç Gün Tahmin Edilsin?", min_value=7, max_value=90, value=30, key="forecast_days")
            time_step = 60 # Modelin geçmişe bakacağı gün sayısı

            if st.button("Tahmin Modelini Çalıştır", key="run_lstm", type="primary"):
                with st.spinner(f"1/3: Veri hazırlanıyor..."):
                    # Eğitilecek modeli ve ölçekleyiciyi al (önbellekten veya eğiterek)
                    model, scaler, rmse = train_and_predict_lstm(df_cleaned, selected_param_lstm, time_step)
                
                if model is None:
                    st.error(rmse) # Hata mesajını göster
                else:
                    st.metric(
                        label="Modelin Eğitim Verisi Üzerindeki Başarısı (RMSE)",
                        value=f"{rmse:.4f}",
                        help="Kök Ortalama Kare Hata (RMSE): Modelin tahminlerinin gerçek değerlerden ortalama ne kadar saptığını gösterir. Düşük değer daha iyi performansa işaret eder."
                    )
                    
                    with st.spinner(f"2/3: Model eğitildi. Gelecek {forecast_days} gün için tahminler yapılıyor..."):
                        series_for_forecast = df_cleaned[selected_param_lstm].dropna()
                        last_sequence = series_for_forecast[-time_step:]
                        forecast_values = make_future_forecast(model, scaler, last_sequence, time_step, forecast_days)

                    with st.spinner("3/3: Sonuçlar ve grafik oluşturuluyor..."):
                        # Grafik için tarihleri oluştur
                        last_date = series_for_forecast.index.max()
                        future_dates = pd.to_datetime([last_date + timedelta(days=i) for i in range(1, forecast_days + 1)])
                        
                        # Sonuçları görselleştir
                        fig_forecast = go.Figure()
                        fig_forecast.add_trace(go.Scatter(x=series_for_forecast.index, y=series_for_forecast.values, mode='lines', name='Geçmiş Veriler'))
                        fig_forecast.add_trace(go.Scatter(x=future_dates, y=forecast_values, mode='lines', name='Tahmin Edilen Değerler', line=dict(color='red', dash='dash')))
                        fig_forecast.update_layout(
                            title=f"{selected_plant_lstm} - '{selected_param_lstm}' Parametresi Tahmini",
                            xaxis_title="Tarih", yaxis_title="Değer", legend_title="Veri Tipi"
                        )
                        st.plotly_chart(fig_forecast, use_container_width=True)

                        # Tahmin tablosunu oluştur
                        df_forecast = pd.DataFrame({'Tarih': future_dates, 'Tahmin Edilen Değer': forecast_values})
                        st.subheader(f"Gelecek {forecast_days} Günlük Tahmin Değerleri")
                        st.dataframe(df_forecast.set_index('Tarih').style.format("{:.2f}"))

                    st.success("Tahmin işlemi başarıyla tamamlandı!")

# Eğer dosya yüklenmediyse başlangıç ekranı mesajı
else:
    st.warning("Lütfen analiz ve tahmin işlemlerine başlamak için bir CSV dosyası yükleyin.")