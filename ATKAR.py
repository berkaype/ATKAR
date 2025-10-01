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

def get_param_with_unit(param_name, units_dict):
    """Parametre adını birimi ile birlikte döndürür"""
    unit = units_dict.get(param_name, "")
    if unit and unit.strip():
        return f"{param_name} ({unit})"
    return param_name

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

def create_yearly_subplots(df, selected_params, years, chart_types_for_params, opacity_values, limit_values, show_labels, yaxis_assignments, decimal_separator, units_dict):
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
            param_with_unit = get_param_with_unit(param, units_dict)
            
            trace_args = {
                'x': year_data.index,
                'y': year_data[param],
                'name': f"{param_with_unit} ({year})",
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
                unit = units_dict.get(param, "")
                limit_text = f"{param} Limit: {format_number(limit_values[param], decimal_separator, False)}"
                if unit:
                    limit_text += f" {unit}"
                    
                fig.add_annotation(
                    x=year_data.index.max(),
                    y=limit_values[param],
                    text=limit_text,
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
st.markdown("""
**CSV Formatı:**
- 1. satır: Parametrelerin birimleri (ilk hücre boş, sonraki hücreler birimler)
- 2. satır: Başlıklar (tarih sütunu + parametre adları)
- 3. satırdan itibaren: Tarih ve veri değerleri
""")

uploaded_file = st.file_uploader("CSV dosyasını yükleyin", type=["csv"], key="data_uploader")

if uploaded_file:
    @st.cache_data
    def load_data_with_units(file):
        encodings = ['utf-8', 'latin1', 'cp1254']
        separators = [';', '\t', ',']
        
        for enc in encodings:
            for sep in separators:
                try:
                    file.seek(0)
                    units_row = pd.read_csv(file, encoding=enc, sep=sep, nrows=1, header=None)

                    if units_row.shape[1] <= 1:
                        continue
                        
                    file.seek(0)
                    headers_row = pd.read_csv(file, encoding=enc, sep=sep, nrows=1, skiprows=1, header=None)
                    file.seek(0)
                    df = pd.read_csv(file, encoding=enc, sep=sep, skiprows=2, header=None)
                    df.columns = headers_row.iloc[0].values
                    df.columns = df.columns.map(str).str.strip()
                    units_dict = {}
                    if len(units_row.columns) > 1:
                        param_columns = headers_row.iloc[0].values[1:]
                        unit_values = units_row.iloc[0].values[1:]
                        for i, param in enumerate(param_columns):
                            if i < len(unit_values) and pd.notna(unit_values[i]):
                                units_dict[str(param).strip()] = str(unit_values[i]).strip()
                            else:
                                units_dict[str(param).strip()] = ""
                    return df, units_dict
                except Exception as e:
                    continue
        return None, None
    df_initial, units_dict = load_data_with_units(uploaded_file)

    if df_initial is None:
        st.error("Dosya okunamadı. Lütfen dosyanızın CSV formatında olduğundan ve doğru ayırıcıyı kullandığından emin olun.")
        st.stop()
    
    # Birimler sözlüğünü göster
    if units_dict:
        with st.expander("Tespit Edilen Parametre Birimleri"):
            units_df = pd.DataFrame([
                {'Parametre': param, 'Birim': unit if unit else "Birim belirtilmemiş"}
                for param, unit in units_dict.items()
            ])
            st.dataframe(units_df, use_container_width=True)

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
        
        # Parametreleri birimli olarak göster
        param_options = []
        param_mapping = {}  # Birimli gösterim -> orijinal parametre adı
        for param in available_params:
            param_with_unit = get_param_with_unit(param, units_dict)
            param_options.append(param_with_unit)
            param_mapping[param_with_unit] = param
        
        selected_params_with_units = st.multiselect("Grafikte Gösterilecek Parametreleri Seçin", param_options, key="params_tab1")
        selected_params = [param_mapping[param] for param in selected_params_with_units]
        
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
                    param_with_unit = get_param_with_unit(param, units_dict)
                    chart_types_for_params[param] = st.selectbox(
                        f"'{param_with_unit}' tipi", 
                        ("Çizgi (Line)", "Çubuk (Bar)", "Nokta (Scatter)"), 
                        key=f"chart_type_{param}"
                    )
            
            with col2:
                st.write("**Y Ekseni**")
                for param in selected_params:
                    param_with_unit = get_param_with_unit(param, units_dict)
                    axis_choice = st.selectbox(
                        f"'{param_with_unit}' ekseni", 
                        ('Birincil Eksen (Sol)', 'İkincil Eksen (Sağ)'), 
                        key=f"yaxis_{param}"
                    )
                    yaxis_assignments[param] = 'y2' if 'İkincil' in axis_choice else 'y'
            
            with col3:
                st.write("**Opaklık**")
                for param in selected_params:
                    param_with_unit = get_param_with_unit(param, units_dict)
                    opacity_values[param] = st.slider(
                        f"'{param_with_unit}' opaklık", 
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
                        param_with_unit = get_param_with_unit(param, units_dict)
                        # Hangi eksende olduğunu göster
                        axis_info = "Birincil" if yaxis_assignments.get(param, 'y') == 'y' else "İkincil"
                        limit_val = st.number_input(f"{param_with_unit} için limit ({axis_info} Eksen):", value=None, key=f"limit_{param}")
                        limit_values[param] = limit_val

            # Yıllık görünüm için yıl seçimi
            if yearly_view:
                available_years = sorted(df_resampled.index.year.unique()) if time_frequency != "Yıllık" else sorted(df_resampled.index.unique())
                selected_years = st.multiselect("Gösterilecek Yılları Seçin:", available_years, default=available_years[-2:] if len(available_years) > 1 else available_years)
            
            st.subheader("Grafik")
            
            # Yıllık görünüm veya normal görünüm
            if yearly_view and selected_years:
                fig = create_yearly_subplots(df_resampled, selected_params, selected_years, chart_types_for_params, opacity_values, limit_values, show_labels, yaxis_assignments, decimal_separator, units_dict)
            else:
                fig = go.Figure()
                
                for param in selected_params:
                    chart_type = chart_types_for_params.get(param, "Çizgi (Line)")
                    target_yaxis = yaxis_assignments.get(param, 'y')
                    opacity = opacity_values.get(param, 1.0)
                    param_with_unit = get_param_with_unit(param, units_dict)
                    
                    trace_args = {
                        'x': df_resampled.index, 
                        'y': df_resampled[param], 
                        'name': param_with_unit, 
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
                            unit = units_dict.get(param, "")
                            limit_text = f"{param} Limit: {format_number(limit_val, decimal_separator, False)}"
                            if unit:
                                limit_text += f" {unit}"
                            
                            fig.add_hline(
                                y=limit_val,
                                line_dash="dash",
                                line_color="red",
                                annotation_text=limit_text,
                                yref='y2' if target_yaxis == 'y2' else 'y'
                            )
                
                # Y ekseni etiketlerini birimlerle güncelle
                primary_params = [p for p in selected_params if yaxis_assignments.get(p, 'y') == 'y']
                secondary_params = [p for p in selected_params if yaxis_assignments.get(p, 'y') == 'y2']
                
                # Birincil Y ekseni etiketi
                if primary_params:
                    primary_units = list(set([units_dict.get(p, "") for p in primary_params if units_dict.get(p, "")]))
                    if len(primary_units) == 1:
                        fig.update_layout(yaxis_title=f"Değer ({primary_units[0]})")
                    elif len(primary_units) > 1:
                        fig.update_layout(yaxis_title=f"Değer ({', '.join(primary_units)})")
                    else:
                        fig.update_layout(yaxis_title="Değer")
                
                # İkincil Y ekseni ayarları ve etiketi
                if secondary_params:
                    secondary_units = list(set([units_dict.get(p, "") for p in secondary_params if units_dict.get(p, "")]))
                    if len(secondary_units) == 1:
                        yaxis2_title = f"Değer ({secondary_units[0]})"
                    elif len(secondary_units) > 1:
                        yaxis2_title = f"Değer ({', '.join(secondary_units)})"
                    else:
                        yaxis2_title = "Değer"
                    
                    fig.update_layout(yaxis2=dict(overlaying='y', side='right', title=yaxis2_title))
                elif any(yaxis == 'y2' for yaxis in yaxis_assignments.values()):
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
                # Histogram için parametre seçimi
                col1, col2 = st.columns([2, 1])
                with col1:
                    histogram_param_options = [get_param_with_unit(param, units_dict) for param in selected_params]
                    histogram_param_with_unit = st.selectbox(
                        "Histogram için parametre seçin:", 
                        histogram_param_options, 
                        key="histogram_param"
                    )
                    # Orijinal parametre adını bul
                    histogram_param = None
                    for param in selected_params:
                        if get_param_with_unit(param, units_dict) == histogram_param_with_unit:
                            histogram_param = param
                            break
                
                with col2:
                    show_histogram = st.checkbox("Histogram Göster", value=False, key="show_histogram")
                    if show_histogram:
                        bin_count = st.slider("Bin Sayısı:", min_value=5, max_value=50, value=20, key="bin_count")
                
                # Kapsamlı istatistik tablosu oluştur
                stats_data = []
                for param in selected_params:
                    if param in df_resampled.columns:
                        series = df_resampled[param].dropna()
                        if len(series) > 0:
                            mean_val = series.mean()
                            std_val = series.std()
                            cv_val = (std_val / abs(mean_val)) * 100 if mean_val != 0 else np.inf  # CV yüzde olarak
                            skewness_val = series.skew()
                            kurtosis_val = series.kurtosis()
                            
                            param_with_unit = get_param_with_unit(param, units_dict)
                            
                            stats_data.append({
                                'Parametre': param_with_unit,
                                'Count': len(series),
                                'Mean': mean_val,
                                'Median': series.median(),
                                'Minimum': series.min(),
                                'Maximum': series.max(),
                                'Std Dev': std_val,
                                'Variance': series.var(),
                                'CV (%)': cv_val,  # Coefficient of Variation
                                'Skewness': skewness_val,
                                'Kurtosis': kurtosis_val,
                                '25%': series.quantile(0.25),
                                '50%': series.quantile(0.50),
                                '75%': series.quantile(0.75)
                            })
                
                if stats_data:
                    detailed_stats_df = pd.DataFrame(stats_data)
                    detailed_stats_df = detailed_stats_df.set_index('Parametre')
                    
                    # İstatistikleri formatla (CV için özel formatting)
                    formatted_stats = detailed_stats_df.copy()
                    for col in formatted_stats.columns:
                        if col == 'CV (%)':
                            # CV için özel formatla (sonsuz değerleri işle)
                            formatted_stats[col] = formatted_stats[col].apply(
                                lambda x: "∞" if np.isinf(x) else format_number(x, decimal_separator, True)
                            )
                        elif col == 'Count':
                            # Count için tam sayı formatla
                            formatted_stats[col] = formatted_stats[col].astype(int)
                        else:
                            formatted_stats[col] = formatted_stats[col].apply(
                                lambda x: format_number(x, decimal_separator, True)
                            )
                    
                    st.data_editor(
                        formatted_stats,
                        use_container_width=True,
                        hide_index=False,
                        disabled=True,
                        key="stats_table"
                    )
                    
                    # Skewness ve Kurtosis yorumları
                    st.subheader("İstatistiksel Yorumlar")
                    
                    # Her parametre için yorum tablosu
                    interpretation_data = []
                    for param in selected_params:
                        param_with_unit = get_param_with_unit(param, units_dict)
                        if param_with_unit in detailed_stats_df.index:
                            skew_val = detailed_stats_df.loc[param_with_unit, 'Skewness']
                            kurt_val = detailed_stats_df.loc[param_with_unit, 'Kurtosis']
                            cv_val = detailed_stats_df.loc[param_with_unit, 'CV (%)']
                            
                            # Skewness yorumu
                            if abs(skew_val) < 0.5:
                                skew_interpretation = "Yaklaşık simetrik"
                            elif abs(skew_val) < 1.0:
                                skew_interpretation = "Hafif çarpık" + (" (sağa)" if skew_val > 0 else " (sola)")
                            else:
                                skew_interpretation = "Oldukça çarpık" + (" (sağa)" if skew_val > 0 else " (sola)")
                            
                            # Kurtosis yorumu
                            if kurt_val < -1:
                                kurt_interpretation = "Platykurtic (düz dağılım)"
                            elif kurt_val > 1:
                                kurt_interpretation = "Leptokurtic (sivri dağılım)"
                            else:
                                kurt_interpretation = "Mesokurtic (normal benzeri)"
                            
                            # CV yorumu
                            if np.isinf(cv_val):
                                cv_interpretation = "Tanımsız (ortalama sıfır)"
                            elif cv_val < 10:
                                cv_interpretation = "Düşük değişkenlik"
                            elif cv_val < 25:
                                cv_interpretation = "Orta değişkenlik"
                            else:
                                cv_interpretation = "Yüksek değişkenlik"
                            
                            interpretation_data.append({
                                'Parametre': param_with_unit,
                                'Çarpıklık (Skewness)': f"{format_number(skew_val, decimal_separator, False)} - {skew_interpretation}",
                                'Basıklık (Kurtosis)': f"{format_number(kurt_val, decimal_separator, False)} - {kurt_interpretation}",
                                'Değişim Katsayısı': f"{format_number(cv_val, decimal_separator, False) if not np.isinf(cv_val) else '∞'}% - {cv_interpretation}"
                            })
                    
                    interpretation_df = pd.DataFrame(interpretation_data)
                    st.data_editor(
                        interpretation_df.set_index('Parametre'),
                        use_container_width=True,
                        hide_index=False,
                        disabled=True,
                        key="interpretation_table"
                    )
                    
                    with st.expander("İstatistiksel Terimlerin Açıklamaları"):
                        st.markdown("""
                        Bu bölümde, yukarıdaki tabloda yer alan istatistiksel terimlerin ne anlama geldiği açıklanmaktadır.

                        ### Çarpıklık (Skewness)
                        Veri dağılımının simetrisini ölçer.
                        - **Yaklaşık Simetrik:** Veriler, ortalama değer etrafında dengeli bir şekilde dağılmıştır. Normal dağılıma benzer bir yapı gösterir.
                        - **Sağa Çarpık (Pozitif Çarpıklık):** Dağılımın kuyruğu sağa doğru uzar. Bu durum, veri setinde ortalamayı yukarı çeken birkaç tane aykırı yüksek değer olduğunu gösterir. Genellikle `Ortalama > Medyan` olur.
                        - **Sola Çarpık (Negatif Çarpıklık):** Dağılımın kuyruğu sola doğru uzar. Bu durum, veri setinde ortalamayı aşağı çeken birkaç tane aykırı düşük değer olduğunu gösterir. Genellikle `Ortalama < Medyan` olur.

                        ---

                        ### Basıklık (Kurtosis)
                        Veri dağılımının ne kadar 'sivri' veya 'düz' olduğunu normal dağılıma göre ölçer. Aykırı değerlerin varlığı hakkında fikir verir.
                        - **Leptokurtic (Sivri Dağılım):** Dağılım, normal dağılıma göre daha sivri bir tepeye ve daha kalın kuyruklara sahiptir. Bu, veri setinde daha fazla aykırı değer olma olasılığını gösterir.
                        - **Mesokurtic (Normal Benzeri):** Dağılımın basıklığı normal dağılıma çok benzer. Aykırı değerler beklendiği gibidir.
                        - **Platykurtic (Düz Dağılım):** Dağılım, normal dağılıma göre daha basık bir tepeye ve daha ince kuyruklara sahiptir. Bu, veri setinde daha az aykırı değer olma olasılığını gösterir.

                        ---

                        ### Değişim Katsayısı (Coefficient of Variation - CV)
                        Standart sapmanın ortalamaya bölünmesiyle bulunan, yüzde olarak ifade edilen göreceli bir değişkenlik ölçüsüdür. Farklı birimlere veya farklı ortalamalara sahip veri setlerinin değişkenliğini karşılaştırmak için çok kullanışlıdır.
                        - **Düşük Değişkenlik:** Veri noktaları ortalamaya çok yakındır. Veri seti tutarlıdır.
                        - **Orta Değişkenlik:** Verilerde makul ve beklenebilecek bir yayılım vardır.
                        - **Yüksek Değişkenlik:** Veri noktaları ortalamadan oldukça uzağa yayılmıştır. Veri seti tutarsızdır veya büyük dalgalanmalar göstermektedir.
                        - **Tanımsız:** Ortalama değer sıfır ise standart sapmayı sıfıra bölmek matematiksel olarak tanımsız olduğundan bu katsayı hesaplanamaz.
                        """)
                
                # Histogram ve Bin Analizi
                if show_histogram and histogram_param and histogram_param in df_resampled.columns:
                    histogram_param_with_unit = get_param_with_unit(histogram_param, units_dict)
                    st.subheader(f"Histogram Analizi: {histogram_param_with_unit}")
                    
                    series_for_hist = df_resampled[histogram_param].dropna()
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        # Histogram oluştur
                        fig_hist = go.Figure()
                        fig_hist.add_trace(go.Histogram(
                            x=series_for_hist,
                            nbinsx=bin_count,
                            name=histogram_param_with_unit,
                            marker_color='skyblue',
                            marker_line_color='darkblue',
                            marker_line_width=1
                        ))
                        
                        # Ortalama ve medyan çizgileri ekle
                        mean_val = series_for_hist.mean()
                        median_val = series_for_hist.median()
                        
                        fig_hist.add_vline(
                            x=mean_val, 
                            line_dash="dash", 
                            line_color="red"
                        )
                        fig_hist.add_vline(
                            x=median_val, 
                            line_dash="dash", 
                            line_color="green"
                        )
                        
                        # X eksenine birim ekle
                        unit = units_dict.get(histogram_param, "")
                        x_title = f"Değer ({unit})" if unit else "Değer"
                        
                        fig_hist.update_layout(
                            title=f"{histogram_param_with_unit} - Frekans Dağılımı",
                            xaxis_title=x_title,
                            yaxis_title="Frekans",
                            showlegend=False
                        )
                        
                        st.plotly_chart(fig_hist, use_container_width=True)
                        
                        # Histogram lejandı
                        unit_display = f" {unit}" if unit else ""
                        st.markdown(f"""
                        <div style="background-color: var(--secondary-background-color); border: 1px solid var(--gray-30); padding: 10px; border-radius: 5px; margin-top: -10px;">
                        <small>
                        <span style="color: red;">━ ━ ━</span> <strong>Ortalama:</strong> {format_number(mean_val, decimal_separator, False)}{unit_display} &nbsp;&nbsp;&nbsp;
                        <span style="color: green;">━ ━ ━</span> <strong>Medyan:</strong> {format_number(median_val, decimal_separator, False)}{unit_display}
                        </small>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        # Bin aralıkları tablosu
                        st.write("**Bin Aralıkları ve Frekanslar**")
                        
                        # Numpy histogram ile bin bilgilerini al
                        hist_counts, bin_edges = np.histogram(series_for_hist, bins=bin_count)
                        
                        bin_data = []
                        for i in range(len(hist_counts)):
                            bin_start = bin_edges[i]
                            bin_end = bin_edges[i+1]
                            bin_data.append({
                                'Bin': f"{i+1}",
                                'Aralık': f"[{format_number(bin_start, decimal_separator, False)}, {format_number(bin_end, decimal_separator, False)})",
                                'Frekans': int(hist_counts[i]),
                                'Yüzde (%)': round((hist_counts[i] / len(series_for_hist)) * 100, 2)
                            })
                        
                        bin_df = pd.DataFrame(bin_data)
                        st.data_editor(
                            bin_df.set_index('Bin'),
                            use_container_width=True,
                            hide_index=False,
                            disabled=True,
                            key="bin_table"
                        )
                        
                        # Bin istatistikleri özeti
                        st.write("**Bin İstatistikleri**")
                        st.metric("Toplam Bin Sayısı", bin_count)
                        st.metric("En Yüksek Frekans", int(hist_counts.max()))
                        st.metric("En Düşük Frekans", int(hist_counts.min()))
                        
                        # En yüksek frekanslı bin
                        max_freq_idx = np.argmax(hist_counts)
                        max_bin_start = format_number(bin_edges[max_freq_idx], decimal_separator, False)
                        max_bin_end = format_number(bin_edges[max_freq_idx + 1], decimal_separator, False)
                        unit_display = f" {unit}" if unit else ""
                        st.info(f"**En Yüksek Frekanslı Aralık:**\n[{max_bin_start}, {max_bin_end}){unit_display}")
                    
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
                            param_with_unit = get_param_with_unit(param, units_dict)
                            unit = units_dict.get(param, "")
                            for date, value in outliers.items():
                                formatted_value = format_number(value, decimal_separator, True)
                                if unit:
                                    formatted_value += f" {unit}"
                                outlier_data_all.append({
                                    'Tarih': date.strftime('%Y-%m-%d'),
                                    'Parametre': param_with_unit,
                                    'Değer': formatted_value
                                })
                        
                        param_with_unit = get_param_with_unit(param, units_dict)
                        outlier_details.append({
                            'Parametre': param_with_unit,
                            'Toplam Veri': len(original_series),
                            'Outlier Sayısı': len(outliers),
                            'Outlier Oranı (%)': round(len(outliers)/len(original_series)*100, 2) if len(original_series) > 0 else 0
                        })
                
                # Outlier özet tablosu
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Outlier Özeti**")
                    outlier_summary_df = pd.DataFrame(outlier_details)
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
                
                # Kolon isimlerini birimlerle güncelle
                new_columns = {}
                for col in display_data.columns:
                    new_columns[col] = get_param_with_unit(col, units_dict)
                display_data = display_data.rename(columns=new_columns)
                
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
                
                st.data_editor(
                    formatted_display,
                    use_container_width=True,
                    hide_index=False,
                    disabled=True,
                    key="main_data_table"
                )
                
                # Veri indirme seçeneği (UTF-8 encoding ile)
                csv = display_data_copy.to_csv(encoding='utf-8-sig')
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
            
            # Parametreleri birimlerle göster
            param_options_lstm = []
            param_mapping_lstm = {}
            for param in params_for_plant:
                param_with_unit = get_param_with_unit(param, units_dict)
                param_options_lstm.append(param_with_unit)
                param_mapping_lstm[param_with_unit] = param
            
            selected_param_lstm_with_unit = st.selectbox("Tahmin Edilecek Parametreyi Seçin", param_options_lstm, key="param_lstm")
            selected_param_lstm = param_mapping_lstm[selected_param_lstm_with_unit]
            
            forecast_days = st.slider("Gelecek Kaç Gün Tahmin Edilsin?", min_value=7, max_value=90, value=30, key="forecast_days")
            time_step = 60

            if st.button("Tahmin Modelini Çalıştır", key="run_lstm", type="primary"):
                with st.spinner(f"1/3: Veri hazırlanıyor..."):
                    model, scaler, rmse = train_and_predict_lstm(df_cleaned, selected_param_lstm, time_step)
                
                if model is None:
                    st.error(rmse)
                else:
                    unit = units_dict.get(selected_param_lstm, "")
                    rmse_display = f"{rmse:.4f}"
                    if unit:
                        rmse_display += f" {unit}"
                    
                    st.metric(
                        label="Modelin Eğitim Verisi Üzerindeki Başarısı (RMSE)",
                        value=rmse_display,
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
                        
                        # Y eksenine birim ekle
                        y_title = f"Değer ({unit})" if unit else "Değer"
                        
                        fig_forecast.update_layout(
                            title=f"{selected_plant_lstm} - '{selected_param_lstm_with_unit}' Parametresi Tahmini",
                            xaxis_title="Tarih", 
                            yaxis_title=y_title, 
                            legend_title="Veri Tipi"
                        )
                        st.plotly_chart(fig_forecast, use_container_width=True)

                        df_forecast = pd.DataFrame({'Tarih': future_dates, 'Tahmin Edilen Değer': forecast_values})
                        # Tarih formatını düzenle
                        df_forecast['Tarih'] = df_forecast['Tarih'].dt.strftime('%Y-%m-%d')
                        
                        # Tahmin değerlerini formatla ve birim ekle
                        df_forecast_formatted = df_forecast.copy()
                        if unit:
                            df_forecast_formatted['Tahmin Edilen Değer'] = df_forecast_formatted['Tahmin Edilen Değer'].apply(
                                lambda x: f"{format_number(x, decimal_separator, True)} {unit}"
                            )
                        else:
                            df_forecast_formatted['Tahmin Edilen Değer'] = df_forecast_formatted['Tahmin Edilen Değer'].apply(
                                lambda x: format_number(x, decimal_separator, True)
                            )
                        
                        st.subheader(f"Gelecek {forecast_days} Günlük Tahmin Değerleri")
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