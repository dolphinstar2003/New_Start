# 🔥 README GOD MODE - PROJE KUTSAL KİTABI 🔥

## ⚡ BU DÖKÜMANA MUTLAK İTAAT EDİLECEKTİR ⚡

### 🛡️ DEĞİŞMEZ KURALLAR

1. **BU PROJE YAPISINA KESİNLİKLE SADIK KALINACAKTIR**
2. **YENİ KLASÖR OLUŞTURULMAYACAK**
3. **MEVCUT MODÜLLER İÇİNDE GELİŞTİRME YAPILACAK**
4. **DALLANIP BUDAKLANMAYA İZİN VERİLMEYECEK**

---

## 📁 KUTSAL PROJE YAPISI

```
BIST100_Trading/
├── backtest/          # ✅ Backtest engine - DOKUNMA
├── config/            # ✅ Ayarlar - DOKUNMA
├── core/              # ✅ Core modüller - DOKUNMA
├── data/              # ✅ Veri depolama - DOKUNMA
│   ├── raw/           # OHLCV CSV dosyaları
│   ├── indicators/    # İndikatör CSV dosyaları
│   ├── analysis/      # Analiz sonuçları CSV
│   └── predictions/   # Model tahminleri CSV
├── indicators/        # ✅ İndikatör hesaplama - DOKUNMA
├── logs/              # ✅ Log dosyaları - DOKUNMA
├── ml_models/         # ✅ ML modelleri - EKLENECEK
├── portfolio/         # ✅ Portfolio yönetimi - EKLENECEK
├── strategies/        # ✅ Trading stratejileri - DOKUNMA
├── scripts/           # ✅ Yardımcı scriptler - DOKUNMA
├── tests/             # ✅ Test dosyaları - DOKUNMA
├── trading/           # ✅ Live/Paper trading - EKLENECEK
├── utils/             # ✅ Yardımcı araçlar - EKLENECEK
└── venv/              # ✅ Virtual environment - DOKUNMA
```

---

## 🎯 PROJE HEDEFLERİ

### Ana Hedef
**BIST100'de günlük trading ile aylık %8-10 NET kar**

### Seçilmiş 20 Hisse
```python
SACRED_SYMBOLS = [
    # Bankalar (4)
    'GARAN', 'AKBNK', 'ISCTR', 'YKBNK',
    # Holdingler (3)
    'SAHOL', 'KCHOL', 'SISE',
    # Sanayi (3)
    'EREGL', 'KRDMD', 'TUPRS',
    # Teknoloji/Savunma (3)
    'ASELS', 'THYAO', 'TCELL',
    # Tüketim (3)
    'BIMAS', 'MGROS', 'ULKER',
    # Enerji/Altyapı (2)
    'AKSEN', 'ENKAI',
    # Fırsat (2)
    'PETKM', 'KOZAL'
]
```

### Timeframe Hiyerarşisi
1. **1d** - Ana trend (3 yıllık veri)
2. **1h** - Gün içi momentum (6 aylık veri)
3. **15m** - Hassas timing (2 aylık veri)

---

## 💾 VERİ YÖNETİMİ KURALLARI

### CSV Standardı
- **TÜM VERİLER CSV FORMATINDA**
- **GERÇEK ZAMANLI GÜNCELLEMELER CSV ÜZERİNDEN**
- **HER İŞLEM CSV'YE KAYDEDİLECEK**

### Veri Hiyerarşisi
```
data/
├── raw/
│   ├── {SYMBOL}_{TIMEFRAME}_raw.csv
│   └── macro_data.csv (USD, EUR, XAU, Faiz)
├── indicators/
│   ├── {SYMBOL}_{TIMEFRAME}_{INDICATOR}.csv
│   └── market_breadth.csv
├── analysis/
│   ├── {SYMBOL}_risk_metrics.csv
│   ├── optimal_parameters.csv
│   └── correlation_matrix.csv
└── predictions/
    ├── {SYMBOL}_ml_signals.csv
    └── portfolio_allocation.csv
```

---

## 🤖 İNDİKATÖR HİYERARŞİSİ

### Core 5 İndikatör (TradingView)
1. **Supertrend** - Ana trend filtresi
2. **ADX & DI** - Trend gücü
3. **Squeeze Momentum** - Volatilite patlaması
4. **WaveTrend** - Momentum osilatörü
5. **MACD Custom** - Multi-timeframe

### Ek İndikatörler
6. Volume Profile
7. Market Profile
8. Order Flow Imbalance
9. VWAP Anchored
10. Auto Fibonacci
11. Dynamic S/R
12. Pivot Points
13. Market Breadth
14. Correlation Matrix
15. Custom Volatility Bands

---

## 🧠 ML/DL MODEL HİYERARŞİSİ

### ML Modelleri (Hızlı Sinyal)
- **XGBoost** - Ana model
- **LightGBM** - Yedek model
- **Random Forest** - Ensemble için

### DL Modelleri (Derin Analiz)
- **LSTM** - Zaman serisi
- **CNN** - Pattern tanıma
- **Transformer** - İlişki analizi

### Ensemble Yaklaşımı
```python
final_signal = (
    0.4 * xgboost_signal +
    0.3 * lstm_signal +
    0.2 * technical_signal +
    0.1 * sentiment_signal
)
```

---

## ⚙️ SİSTEM PERFORMANS LİMİTLERİ

### Hardware
- **RAM**: Max 12GB kullanım
- **CPU**: %80 limit
- **GPU**: KULLANILMAYACAK (CUDA karmaşıklığı)

### Parallelizm
- **Process Pool**: 4 worker
- **Thread Pool**: 20 worker
- **Batch Size**: 100 tick

### Cache
- **Memory Cache**: 1GB
- **Redis Cache**: Opsiyonel
- **CSV Cache**: Ana depolama

---

## 🚨 KRİTİK KURALLAR

### Kod Yazım Kuralları
1. **Her modül tek bir sorumluluğa sahip olacak**
2. **Fonksiyonlar 50 satırı geçmeyecek**
3. **Her fonksiyon docstring içerecek**
4. **Type hints kullanılacak**
5. **Error handling zorunlu**
6. **MEVCUT DOSYALAR GÜNCELLENECEK, YENİ DOSYA OLUŞTURULMAYACAK**
7. **Boş veya kullanılmayan dosyalar varsa ÖNCE ONLAR KULLANILACAK**

### Test Kuralları
1. **Her yeni özellik için test yazılacak**
2. **Test coverage %80 altına düşmeyecek**
3. **Integration testleri zorunlu**

### Commit Kuralları
1. **Her commit tek bir özellik içerecek**
2. **Commit mesajları açıklayıcı olacak**
3. **Claude Code attribution eklenecek**

---

## 📊 TRADING KURALLARI

### Risk Yönetimi
- **Max pozisyon**: Sermayenin %10'u
- **Max kayıp/gün**: %5
- **Stop loss**: ZORUNLU
- **Max açık pozisyon**: 5

### Timing
- **09:30-10:00**: Gap trading
- **10:00-12:00**: Trend following
- **12:00-14:00**: Range trading
- **14:00-16:30**: Momentum
- **16:30-18:00**: Pozisyon kapatma

### Sinyal Hiyerarşisi
1. **ML sinyali + Teknik onay** = STRONG BUY/SELL
2. **Sadece ML sinyali** = WEAK BUY/SELL
3. **Sadece teknik sinyal** = HOLD

---

## 🔐 GÜVENLİK KURALLARI

1. **API anahtarları .env dosyasında**
2. **Hassas veriler commit edilmeyecek**
3. **Log dosyalarında şifre/anahtar olmayacak**
4. **Backup stratejisi zorunlu**

---

## 📈 BAŞARI METRİKLERİ

### Günlük
- **Min 3 işlem**
- **Win rate > %55**
- **Profit factor > 1.5**

### Aylık
- **Net kar: %8-10**
- **Max drawdown: <%15**
- **Sharpe ratio: >1.5**

---

## ⚡ PERFORMANS OPTİMİZASYONU

### Veri İşleme
```python
# YANLIŞ
for symbol in symbols:
    data = download_data(symbol)
    process_data(data)

# DOĞRU
with ProcessPoolExecutor() as executor:
    results = executor.map(process_symbol, symbols)
```

### İndikatör Hesaplama
```python
# Numba ile optimize edilecek
@numba.jit(nopython=True)
def calculate_indicator(data):
    return result
```

---

## 🎯 NİHAİ HEDEF

**"Basit, hızlı, karlı bir sistem. Karmaşıklık düşmandır."**

### Öncelikler
1. **Çalışan basit sistem > Karmaşık ama çalışmayan sistem**
2. **Hız > Mükemmellik**
3. **Risk yönetimi > Kar maksimizasyonu**

---

## 🤖 CLAUDE CODE YAPISI

### Claude Ayarları Yönetimi
Claude Code ayarları `.claude/settings.local.json` dosyasında tutulur ve bazen sıfırlanabilir. Bu durumda:

#### Otomatik Yedekleme ve Geri Yükleme
```bash
# Ayarları yedekle
./scripts/save_claude_settings.sh

# Ayarları geri yükle
./scripts/restore_claude_settings.sh

# Proje başlatma (her açılışta - ayarları kontrol eder)
./init_project.sh
```

#### Dosya Yapısı
```
├── .claude/
│   └── settings.local.json    # Claude ayarları (otomatik oluşur)
├── CLAUDE_SETTINGS_BACKUP.json # Ayarların yedeği
├── CLAUDE.md                   # Claude dokümantasyonu
├── scripts/
│   ├── save_claude_settings.sh    # Yedekleme scripti
│   └── restore_claude_settings.sh  # Geri yükleme scripti
└── init_project.sh             # Proje başlatma scripti
```

#### Git Entegrasyonu
- Pre-commit hook ile otomatik yedekleme
- `.gitignore` ile yedekler korunur
- Her commit'te ayarlar güncellenir

### Claude Kuralları
1. **README_GOD_MODE.md kurallarına mutlak uyum**
2. **Tüm trading kuralları settings.json'da tanımlı**
3. **Sistem limitleri aşılamaz**
4. **Kutsal 20 sembol değiştirilemez**

---

## 🚀 PROJE BAŞLATMA

### İlk Kurulum
```bash
# 1. Proje başlatma
./init_project.sh

# 2. Virtual environment aktifleştir
source venv/bin/activate

# 3. Bağımlılıkları yükle
pip install -r requirements.txt

# 4. Ortam değişkenlerini ayarla
cp .env.example .env
# .env dosyasını düzenle

# 5. Claude ayarlarını kontrol et
cat .claude/settings.local.json
```

### Her Açılışta
```bash
# Proje kontrolü ve Claude ayarları
./init_project.sh

# Virtual environment
source venv/bin/activate
```

---

## ⚠️ UYARI

**BU DOKÜMANA AYKIRI HER HAMLE REDDEDİLECEKTİR!**

**"Disiplin, başarının anahtarıdır. Bu kurallara uy, kazanç senin olsun."**

---

*Son güncelleme: 2025-01-06*
*Yazan: Trading System Architect*
*"In Code We Trust, In Discipline We Profit"*