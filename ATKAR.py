import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense # type: ignore

# --- Sayfa Yapılandırması ---
st.set_page_config(layout="wide",
                   page_title="ATKAR - Atıksu Arıtma Tesisleri Karşılaştırma ve Tahmin Platformu",
                   page_icon="💧"
                   )

# --- Ana Başlık ---
st.title("Atıksu Arıtma Tesisleri Karşılaştırma ve Tahmin Platformu")

# --- Yardımcı Fonksiyonlar ---
def format_number(value, decimal_separator, include_thousands=True):
    """Sayıyı belirtilen ondalık ayırıcı ve binlik ayıracı ile formatlar"""
    if pd.isna(value):
        return ""
    
    # Binlik ayıracı ile formatla
    if include_thousands:
        formatted = f"{value:,.2f}"
    else:
        formatted = f"{value:.2f}"
    
    # Ondalık ayırıcıyı değiştir
    if decimal_separator == ",":
        if include_thousands:
            # İlk virgülleri (binlik ayıracı) geçici olarak değiştir
            parts = formatted.split(".")
            if len(parts) == 2:
                integer_part = parts[0].replace(",", ".")  # Binlik ayıracı olarak nokta
                decimal_part = parts[1]
                formatted = f"{integer_part},{decimal_part}"  # Ondalık ayırıcı olarak virgül
        else:
            formatted = formatted.replace(".", ",")
    
    return formatted

def format_dataframe(df, decimal_separator, include_thousands=True):
    """DataFrame'deki sayısal değerleri formatlar"""
    df_formatted = df.copy()
    for col in df_formatted.select_dtypes(include=[np.number]).columns:
        df_formatted[col] = df_formatted[col].apply(
            lambda x: format_number(x, decimal_separator, include_thousands)
        )
    return df_formatted

def resample_data(df, selected_params, frequency):
    """Veriyi belirtilen frekansa göre yeniden örnekler"""
    if frequency == "Günlük":
        return df[selected_params]
    elif frequency == "Aylık":
        # Aylık ortalamaları al
        resampled = df[selected_params].resample('ME').mean()
        # Index'i aylık formatla (YYYY-MM)
        resampled.index = resampled.index.strftime('%Y-%m')
        return resampled
    elif frequency == "Mevsimlik":
        # Mevsimlik gruplandırma
        df_copy = df[selected_params].copy()
        df_copy['mevsim'] = df_copy.index.month.map({
            12: 'Kış', 1: 'Kış', 2: 'Kış',
            3: 'İlkbahar', 4: 'İlkbahar', 5: 'İlkbahar',
            6: 'Yaz', 7: 'Yaz', 8: 'Yaz',
            9: 'Sonbahar', 10: 'Sonbahar', 11: 'Sonbahar'
        })
        
        # Mevsimlik ortalama al
        seasonal_data = df_copy.groupby('mevsim')[selected_params].mean()
        
        # Mevsim sıralaması
        season_order = ['İlkbahar', 'Yaz', 'Sonbahar', 'Kış']
        seasonal_data = seasonal_data.reindex([s for s in season_order if s in seasonal_data.index])
        
        return seasonal_data
    elif frequency == "Yıllık":
        # Yıllık ortalamaları al
        resampled = df[selected_params].resample('YE').mean()
        # Index'i yıl formatla
        resampled.index = resampled.index.year
        return resampled
    else:
        return df[selected_params]

def detect_outliers_3sigma(series):
    """3 sigma yöntemi ile outlier'ları tespit eder"""
    mean = series.mean()
    std = series.std()
    outliers = series[(series < (mean - 3*std)) | (series > (mean + 3*std))]
    return outliers

def remove_outliers_3sigma(df, columns):
    """3 sigma yöntemi ile outlier'ları kaldırır"""
    df_cleaned = df.copy()
    for col in columns:
        if col in df_cleaned.columns:
            mean = df_cleaned[col].mean()
            std = df_cleaned[col].std()
            df_cleaned.loc[(df_cleaned[col] < (mean - 3*std)) | 
                          (df_cleaned[col] > (mean + 3*std)), col] = np.nan
    return df_cleaned

def create_yearly_subplots(df, selected_params, years, chart_types_for_params, opacity_values, limit_values, show_labels, yaxis_assignments, decimal_separator):
    """Seçilen yıllar için alt alta grafik oluşturur"""
    fig = make_subplots(
        rows=len(years), 
        cols=1,
        subplot_titles=[f"Yıl: {year}" for year in years],
        shared_xaxes=False,
        vertical_spacing=0.08,
        specs=[[{"secondary_y": True}] for _ in years]  # Her subplot için ikincil y ekseni
    )
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    for i, year in enumerate(years):
        year_data = df[df.index.year == year]
        if len(year_data) == 0:
            continue
            
        for j, param in enumerate(selected_params):
            if param not in year_data.columns:
                continue
                
            chart_type = chart_types_for_params.get(param, "Çizgi (Line)")
            opacity = opacity_values.get(param, 1.0)
            color = colors[j % len(colors)]
            target_yaxis = yaxis_assignments.get(param, 'y')
            secondary_y = target_yaxis == 'y2'
            
            trace_args = {
                'x': year_data.index,
                'y': year_data[param],
                'name': f"{param} ({year})",
                'opacity': opacity,
                'showlegend': True
            }
            
            # Bar chart için özel ayarlar
            if chart_type == "Çubuk (Bar)":
                trace_args['marker'] = dict(color=color)
            else:
                trace_args['line'] = dict(color=color)
            
            if show_labels and chart_type != "Çubuk (Bar)":
                trace_args['mode'] = 'lines+markers+text'
                trace_args['text'] = [format_number(val, decimal_separator, False) if not pd.isna(val) else "" for val in year_data[param]]
                trace_args['textposition'] = 'top center'
                trace_args['textfont'] = dict(size=8)
            elif chart_type != "Çubuk (Bar)":
                trace_args['mode'] = 'lines'
                
            if chart_type == "Çizgi (Line)" or chart_type == "Nokta (Scatter)":
                fig.add_trace(go.Scatter(**trace_args), row=i+1, col=1, secondary_y=secondary_y)
            elif chart_type == "Çubuk (Bar)":
                fig.add_trace(go.Bar(**trace_args), row=i+1, col=1, secondary_y=secondary_y)
            
            # Limit değeri çizgisi ekle (doğru eksene)
            if param in limit_values and limit_values[param] is not None:
                # Y ekseni referansını belirle
                if len(years) == 1:
                    yref = 'y2' if secondary_y else 'y'
                else:
                    yref = f'y{2*(i+1)}' if secondary_y else f'y{2*i+1}' if i > 0 else 'y'
                
                # Limit çizgisini ekle
                fig.add_shape(
                    type="line",
                    x0=year_data.index.min(),
                    y0=limit_values[param],
                    x1=year_data.index.max(),
                    y1=limit_values[param],
                    line=dict(color="red", width=2, dash="dash"),
                    yref=yref,
                    row=i+1,
                    col=1
                )
                
                # Annotation ekle (sağ üst köşeye)
                fig.add_annotation(
                    x=year_data.index.max(),
                    y=limit_values[param],
                    text=f"{param} Limit: {format_number(limit_values[param], decimal_separator, False)}",
                    showarrow=False,
                    xshift=10,
                    yshift=10,
                    font=dict(color="red", size=10),
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="red",
                    borderwidth=1,
                    yref=yref,
                    row=i+1,
                    col=1
                )
    
    fig.update_layout(
        height=400*len(years),
        title_text="Yıllık Karşılaştırma Grafikleri",
        showlegend=True
    )
    
    return fig

# --- Model Eğitimi ve Tahmin Fonksiyonları ---
@st.cache_resource
def train_and_predict_lstm(_df_cleaned, param_name, time_step=60):
    """
    Belirtilen parametre için bir LSTM modeli eğitir ve tahmin yapar.
    Modeli ve ölçekleyiciyi döndürür.
    """
    series = _df_cleaned[param_name].dropna()
    if len(series) < time_step + 10:
        return None, None, f"'{param_name}' parametresi için yeterli veri bulunmuyor (en az {time_step + 10} gün gerekli)."

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(series.values.reshape(-1, 1))

    X, y = [], []
    for i in range(len(scaled_data) - time_step):
        X.append(scaled_data[i:(i + time_step), 0])
        y.append(scaled_data[i + time_step, 0])
    
    X_train, y_train = np.array(X), np.array(y)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(time_step, 1)),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, batch_size=1, epochs=10, verbose=0)

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

# --- Veri Yükleme ---
st.info("Başlamak için lütfen CSV formatındaki zaman serisi verilerinizi yükleyin.")
uploaded_file = st.file_uploader("CSV dosyasını yükleyin", type=["csv"], key="data_uploader")

if uploaded_file:
    @st.cache_data
    def load_data(file):
        encodings = ['utf-8', 'latin1', 'cp1254']
        separators = [',', ';', '\t']
        df = None
        for enc in encodings:
            for sep in separators:
                try:
                    file.seek(0)
                    df = pd.read_csv(file, encoding=enc, sep=sep, engine='python')
                    df.columns = df.columns.map(str).str.strip()
                    return df
                except Exception:
                    continue
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
        st.error(f"Tarih sütunu işlenemedi: {e}")
        st.stop()

    # Sidebar ayarları
    st.sidebar.header("Veri İşleme Ayarları")
    
    decimal_separator = st.sidebar.radio("Verideki Ondalık Ayırıcı", (",", "."))
    data_cols = [c for c in df_initial.columns if c != date_col]
    df_cleaned = df_initial[data_cols].copy()
    
    for col in df_cleaned.columns:
        if decimal_separator == ',':
            df_cleaned[col] = df_cleaned[col].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
        df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
    
    # **YENİ: 0 değerlerini NaN olarak işle**
    zero_handling = st.sidebar.checkbox("0 değerlerini boş hücre olarak işle", value=True, help="Bu seçenek, veri setindeki 0 değerlerini NaN (boş) olarak dönüştürür. Ortalama hesaplamalarında daha doğru sonuçlar verir.")
    if zero_handling:
        df_cleaned = df_cleaned.replace(0, np.nan)
    
    missing_data_strategy = st.sidebar.radio("Boş (NaN) Değerleri Nasıl İşleyelim?", ("Boş Bırak", "Enterpole Et (Doğrusal Doldur)"))
    if missing_data_strategy == "Enterpole Et (Doğrusal Doldur)":
        df_cleaned.interpolate(method='linear', limit_direction='both', inplace=True)

    # --- Sekmeleri Oluştur ---
    tab1, tab2 = st.tabs(["Veri Görselleştirme ve Karşılaştırma", "Zaman Serisi Tahmini (LSTM)"])

    # ============================ SEKME 1: GÖRSELLEŞTİRME ============================
    with tab1:
        st.header("Tesis Verilerini Karşılaştırma")
        min_date, max_date = df_cleaned.index.min().date(), df_cleaned.index.max().date()

        st.sidebar.subheader("Görselleştirme Filtreleri")
        start_date = st.sidebar.date_input("Başlangıç Tarihi", value=min_date, min_value=min_date, max_value=max_date, key="start_date_tab1")
        end_date = st.sidebar.date_input("Bitiş Tarihi", value=max_date, min_value=min_date, max_value=max_date, key="end_date_tab1")

        if start_date > end_date:
            st.error("Başlangıç tarihi bitiş tarihinden sonra olamaz.")
            st.stop()
        
        df = df_cleaned.loc[start_date.strftime('%Y-%m-%d'):end_date.strftime('%Y-%m-%d')]

        plants = sorted({col.split()[0] for col in data_cols})
        selected_plants = st.multiselect("Karşılaştırılacak Tesisleri Seçin", plants, key="plants_tab1")
        
        available_params = sorted([c for c in data_cols if any(c.startswith(p) for p in selected_plants)])
        selected_params = st.multiselect("Grafikte Gösterilecek Parametreleri Seçin", available_params, key="params_tab1")
        
        if selected_params:
            # Grafik görüntüleme seçenekleri
            col1, col2, col3 = st.columns(3)
            with col1:
                show_labels = st.checkbox("Veri Değerlerini Göster", value=False)
                yearly_view = st.checkbox("Yıllık Alt Alta Görünüm", value=False)
            
            with col2:
                show_limit_lines = st.checkbox("Limit Değer Çizgilerini Göster", value=False)
                remove_outliers = st.checkbox("Outlier Değerleri Kaldır", value=False)
                
            with col3:
                # Zaman aralığı seçimi
                time_frequency = st.selectbox(
                    "Zaman Aralığı:", 
                    ["Günlük", "Aylık", "Mevsimlik", "Yıllık"], 
                    index=0, 
                    key="time_freq"
                )
                # Outlier işlemi
                if remove_outliers:
                    # Outlier'ları kaldır
                    df_for_display = remove_outliers_3sigma(df, selected_params)
                else:
                    df_for_display = df
                    
            # Zaman frekansına göre veriyi yeniden örnekle
            df_resampled = resample_data(df_for_display, selected_params, time_frequency)
            
            st.subheader("Grafik Özelleştirme")
            chart_types_for_params, yaxis_assignments, opacity_values = {}, {}, {}
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**Grafik Tipi**")
                for param in selected_params:
                    chart_types_for_params[param] = st.selectbox(
                        f"'{param}' tipi", 
                        ("Çizgi (Line)", "Çubuk (Bar)", "Nokta (Scatter)"), 
                        key=f"chart_type_{param}"
                    )
            
            with col2:
                st.write("**Y Ekseni**")
                for param in selected_params:
                    axis_choice = st.selectbox(
                        f"'{param}' ekseni", 
                        ('Birincil Eksen (Sol)', 'İkincil Eksen (Sağ)'), 
                        key=f"yaxis_{param}"
                    )
                    yaxis_assignments[param] = 'y2' if 'İkincil' in axis_choice else 'y'
            
            with col3:
                st.write("**Opaklık**")
                for param in selected_params:
                    opacity_values[param] = st.slider(
                        f"'{param}' opaklık", 
                        0.1, 1.0, 1.0, 0.1, 
                        key=f"opacity_{param}"
                    )

            # Limit değerleri girişi
            limit_values = {}
            if show_limit_lines:
                st.subheader("Limit Değerleri")
                limit_cols = st.columns(3)
                for i, param in enumerate(selected_params):
                    with limit_cols[i % 3]:
                        # Hangi eksende olduğunu göster
                        axis_info = "Birincil" if yaxis_assignments.get(param, 'y') == 'y' else "İkincil"
                        limit_val = st.number_input(f"{param} için limit ({axis_info} Eksen):", value=None, key=f"limit_{param}")
                        limit_values[param] = limit_val

            # Yıllık görünüm için yıl seçimi
            if yearly_view:
                available_years = sorted(df_resampled.index.year.unique()) if time_frequency != "Yıllık" else sorted(df_resampled.index.unique())
                selected_years = st.multiselect("Gösterilecek Yılları Seçin:", available_years, default=available_years[-2:] if len(available_years) > 1 else available_years)
            
            st.subheader("Grafik")
            
            # Yıllık görünüm veya normal görünüm
            if yearly_view and selected_years:
                fig = create_yearly_subplots(df_resampled, selected_params, selected_years, chart_types_for_params, opacity_values, limit_values, show_labels, yaxis_assignments, decimal_separator)
            else:
                fig = go.Figure()
                
                for param in selected_params:
                    chart_type = chart_types_for_params.get(param, "Çizgi (Line)")
                    target_yaxis = yaxis_assignments.get(param, 'y')
                    opacity = opacity_values.get(param, 1.0)
                    
                    trace_args = {
                        'x': df_resampled.index, 
                        'y': df_resampled[param], 
                        'name': param, 
                        'yaxis': target_yaxis, 
                        'opacity': opacity
                    }
                    
                    # Bar chart için text modunu kaldır
                    if show_labels and chart_type != "Çubuk (Bar)":
                        trace_args['mode'] = 'lines+markers+text'
                        trace_args['text'] = [format_number(val, decimal_separator, False) if not pd.isna(val) else "" for val in df_resampled[param]]
                        trace_args['textposition'] = 'top center'
                        trace_args['textfont'] = dict(size=8)
                    elif chart_type != "Çubuk (Bar)":
                        trace_args['mode'] = 'lines'
                    
                    if chart_type == "Çizgi (Line)" or chart_type == "Nokta (Scatter)":
                        fig.add_trace(go.Scatter(**trace_args))
                    elif chart_type == "Çubuk (Bar)":
                        # Bar için mode parametresini kaldır
                        if 'mode' in trace_args:
                            del trace_args['mode']
                        if 'text' in trace_args:
                            del trace_args['text']
                        if 'textposition' in trace_args:
                            del trace_args['textposition']
                        if 'textfont' in trace_args:
                            del trace_args['textfont']
                        fig.add_trace(go.Bar(**trace_args))
                
                # Limit çizgileri ekle (doğru eksene)
                if show_limit_lines:
                    for param, limit_val in limit_values.items():
                        if limit_val is not None:
                            target_yaxis = yaxis_assignments.get(param, 'y')
                            fig.add_hline(
                                y=limit_val,
                                line_dash="dash",
                                line_color="red",
                                annotation_text=f"{param} Limit: {format_number(limit_val, decimal_separator, False)}",
                                yref='y2' if target_yaxis == 'y2' else 'y'
                            )
                
                # İkincil y ekseni ayarları
                if any(yaxis == 'y2' for yaxis in yaxis_assignments.values()):
                    fig.update_layout(yaxis2=dict(overlaying='y', side='right'))
                
                # X eksenini özel formatla
                if time_frequency == "Yıllık":
                    fig.update_layout(
                        xaxis_title="Yıl",
                        xaxis=dict(
                            tickmode='array',
                            tickvals=df_resampled.index,
                            ticktext=[str(year) for year in df_resampled.index]
                        )
                    )
                elif time_frequency == "Mevsimlik":
                    fig.update_layout(xaxis_title="Mevsim")
                elif time_frequency == "Aylık":
                    fig.update_layout(xaxis_title="Ay")
                else:
                    fig.update_layout(xaxis_title="Tarih")
                
                fig.update_layout(
                    title_text=f"Zaman Serisi Grafiği ({time_frequency})", 
                    hovermode="x unified"
                )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Betimleyici İstatistikler (Genişletilmiş)
            st.subheader("Betimleyici İstatistikler")
            if selected_params:
                # Kapsamlı istatistik tablosu oluştur
                stats_data = []
                for param in selected_params:
                    if param in df_resampled.columns:
                        series = df_resampled[param].dropna()
                        if len(series) > 0:
                            stats_data.append({
                                'Parametre': param,
                                'Count': len(series),
                                'Mean': series.mean(),
                                'Median': series.median(),
                                'Minimum': series.min(),
                                'Maximum': series.max(),
                                'Std Dev': series.std(),
                                'Variance': series.var(),
                                'Skewness': series.skew(),
                                'Kurtosis': series.kurtosis(),
                                '25%': series.quantile(0.25),
                                '50%': series.quantile(0.50),
                                '75%': series.quantile(0.75)
                            })
                
                if stats_data:
                    detailed_stats_df = pd.DataFrame(stats_data)
                    detailed_stats_df = detailed_stats_df.set_index('Parametre')
                    # Formatla
                    formatted_stats = format_dataframe(detailed_stats_df, decimal_separator, True)
                    # **YENİ: data_editor kullan**
                    st.data_editor(
                        formatted_stats,
                        use_container_width=True,
                        hide_index=False,
                        disabled=True,  # Düzenleme yapılmasın
                        key="stats_table"
                    )
                    
            # Outlier Analizi Sonuçları
            if remove_outliers:
                st.subheader("Outlier Analizi Sonuçları (3 Sigma)")
                outlier_details = []
                outlier_data_all = []
                
                for param in selected_params:
                    if param in df.columns:
                        original_series = df[param].dropna()
                        outliers = detect_outliers_3sigma(original_series)
                        
                        if len(outliers) > 0:
                            # Outlier detayları için
                            for date, value in outliers.items():
                                outlier_data_all.append({
                                    'Tarih': date.strftime('%Y-%m-%d'),  # Sadece tarih kısmı
                                    'Parametre': param,
                                    'Değer': format_number(value, decimal_separator, True)
                                })
                        
                        outlier_details.append({
                            'Parametre': param,
                            'Toplam Veri': len(original_series),
                            'Outlier Sayısı': len(outliers),
                            'Outlier Oranı (%)': round(len(outliers)/len(original_series)*100, 2) if len(original_series) > 0 else 0
                        })
                
                # Outlier özet tablosu
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Outlier Özeti**")
                    outlier_summary_df = pd.DataFrame(outlier_details)
                    # **YENİ: data_editor kullan**
                    st.data_editor(
                        outlier_summary_df,
                        use_container_width=True,
                        hide_index=True,
                        disabled=True,
                        key="outlier_summary_table"
                    )
                    
                    # Yıllık outlier dağılımı
                    if outlier_data_all:
                        st.write("**Yıllık Outlier Dağılımı**")
                        outlier_yearly = {}
                        for item in outlier_data_all:
                            year = item['Tarih'][:4]  # İlk 4 karakter yıl
                            outlier_yearly[year] = outlier_yearly.get(year, 0) + 1
                        
                        yearly_outlier_df = pd.DataFrame([
                            {'Yıl': year, 'Outlier Sayısı': count} 
                            for year, count in sorted(outlier_yearly.items())
                        ])
                        # **YENİ: data_editor kullan**
                        st.data_editor(
                            yearly_outlier_df,
                            use_container_width=True,
                            hide_index=True,
                            disabled=True,
                            key="yearly_outlier_table"
                        )
                
                with col2:
                    if outlier_data_all:
                        st.write("**Tespit Edilen Outlier Değerler**")
                        outlier_details_df = pd.DataFrame(outlier_data_all)
                        outlier_details_df = outlier_details_df.sort_values(['Parametre', 'Tarih'])
                        # **YENİ: data_editor kullan**
                        st.data_editor(
                            outlier_details_df,
                            use_container_width=True,
                            hide_index=True,
                            disabled=True,
                            key="outlier_details_table"
                        )
            
            # Veri Tablosu (en alta taşındı)
            st.subheader("Veri Tablosu")
            if selected_params:
                # Seçilen tarih aralığı ve parametreler için tüm veriyi göster
                display_data = df_resampled[selected_params].copy()
                display_data.index.name = 'Tarih'
                
                # Tarihi formatla - sadece tarih kısmını göster
                display_data_copy = display_data.copy()
                
                # Zaman frekansına göre index formatı ayarla
                if time_frequency == "Günlük":
                    # DateTime index'i sadece tarih formatına çevir
                    if hasattr(display_data_copy.index, 'strftime'):
                        display_data_copy.index = display_data_copy.index.strftime('%Y-%m-%d')
                elif time_frequency == "Aylık":
                    # Zaten string formatında (YYYY-MM)
                    pass
                elif time_frequency == "Yıllık":
                    # Zaten yıl formatında
                    pass
                elif time_frequency == "Mevsimlik":
                    # Zaten mevsim isimlerinde
                    pass
                
                # Formatlanmış veri tablosu
                formatted_display = format_dataframe(display_data_copy, decimal_separator, True)
                
                # **YENİ: data_editor kullan - kopyala yapıştır için daha uygun**
                st.data_editor(
                    formatted_display,
                    use_container_width=True,
                    hide_index=False,
                    disabled=True,  # Düzenleme yapılmasın
                    key="main_data_table"
                )
                
                # Veri indirme seçeneği (UTF-8 encoding ile)
                csv = display_data_copy.to_csv(encoding='utf-8-sig')  # UTF-8 BOM ile
                st.download_button(
                    label="Veriyi CSV olarak indir",
                    data=csv,
                    file_name=f'veri_tablosu_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                    mime='text/csv'
                )
        else:
            st.info("Grafiği görüntülemek için lütfen en az bir tesis ve bir parametre seçin.")

    # ============================ SEKME 2: LSTM TAHMİNİ ============================
    with tab2:
        st.header("LSTM ile Geleceğe Yönelik Parametre Tahmini")
        st.markdown("Bu modül, seçtiğiniz bir parametrenin geçmiş verilerini kullanarak gelecekteki değerlerini tahmin etmek için bir Uzun Kısa Süreli Bellek (LSTM) sinir ağı modeli kullanır.")

        plants_lstm = sorted({col.split()[0] for col in data_cols})
        selected_plant_lstm = st.selectbox("Tahmin Edilecek Tesisi Seçin", plants_lstm, key="plant_lstm")

        if selected_plant_lstm:
            params_for_plant = sorted([c for c in data_cols if c.startswith(selected_plant_lstm)])
            selected_param_lstm = st.selectbox("Tahmin Edilecek Parametreyi Seçin", params_for_plant, key="param_lstm")
            forecast_days = st.slider("Gelecek Kaç Gün Tahmin Edilsin?", min_value=7, max_value=90, value=30, key="forecast_days")
            time_step = 60

            if st.button("Tahmin Modelini Çalıştır", key="run_lstm", type="primary"):
                with st.spinner(f"1/3: Veri hazırlanıyor..."):
                    model, scaler, rmse = train_and_predict_lstm(df_cleaned, selected_param_lstm, time_step)
                
                if model is None:
                    st.error(rmse)
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
                        last_date = series_for_forecast.index.max()
                        future_dates = pd.to_datetime([last_date + timedelta(days=i) for i in range(1, forecast_days + 1)])
                        
                        fig_forecast = go.Figure()
                        fig_forecast.add_trace(go.Scatter(x=series_for_forecast.index, y=series_for_forecast.values, mode='lines', name='Geçmiş Veriler'))
                        fig_forecast.add_trace(go.Scatter(x=future_dates, y=forecast_values, mode='lines', name='Tahmin Edilen Değerler', line=dict(color='red', dash='dash')))
                        fig_forecast.update_layout(
                            title=f"{selected_plant_lstm} - '{selected_param_lstm}' Parametresi Tahmini",
                            xaxis_title="Tarih", yaxis_title="Değer", legend_title="Veri Tipi"
                        )
                        st.plotly_chart(fig_forecast, use_container_width=True)

                        df_forecast = pd.DataFrame({'Tarih': future_dates, 'Tahmin Edilen Değer': forecast_values})
                        # Tarih formatını düzenle
                        df_forecast['Tarih'] = df_forecast['Tarih'].dt.strftime('%Y-%m-%d')
                        
                        # Tahmin değerlerini formatla
                        df_forecast_formatted = df_forecast.copy()
                        df_forecast_formatted['Tahmin Edilen Değer'] = df_forecast_formatted['Tahmin Edilen Değer'].apply(
                            lambda x: format_number(x, decimal_separator, True)
                        )
                        
                        st.subheader(f"Gelecek {forecast_days} Günlük Tahmin Değerleri")
                        # **YENİ: data_editor kullan**
                        st.data_editor(
                            df_forecast_formatted.set_index('Tarih'),
                            use_container_width=True,
                            hide_index=False,
                            disabled=True,
                            key="forecast_table"
                        )

                    st.success("Tahmin işlemi başarıyla tamamlandı!")

else:
    st.warning("Lütfen analiz ve tahmin işlemlerine başlamak için bir CSV dosyası yükleyin.")