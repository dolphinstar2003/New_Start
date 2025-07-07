# GitHub Setup ve Streamlit Deploy Rehberi

## Adım 1: GitHub Repository Oluşturma

1. https://github.com adresine gidin
2. Sağ üstte **"+"** → **"New repository"** tıklayın
3. Şu ayarları yapın:
   - Repository name: `New_Start`
   - Description: "Trading Monitor with Fixed Daily Targets"
   - **Public** seçin (önemli!)
   - README eklemeyin (zaten var)
   - "Create repository" tıklayın

## Adım 2: Local Repo'yu GitHub'a Bağlama

Terminal'de şu komutları çalıştırın:

```bash
# GitHub remote ekle (USERNAME'i değiştirin!)
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/New_Start.git

# Örnek:
# git remote add origin https://github.com/yunusemre/New_Start.git

# İlk push
git push -u origin main
```

Eğer hata alırsanız:
```bash
# Branch adını main yap
git branch -M main

# Tekrar dene
git push -u origin main
```

## Adım 3: Streamlit Cloud'da Deploy

1. **https://share.streamlit.io** adresine gidin
2. **GitHub ile giriş yapın**
3. **"New app"** butonuna tıklayın
4. Şu bilgileri doldurun:
   - Repository: `YOUR_GITHUB_USERNAME/New_Start`
   - Branch: `main`
   - Main file path: `paper_trading/trading_monitor.py`
5. **"Deploy!"** tıklayın

## Adım 4: App Ayarları (Opsiyonel)

Deploy edildikten sonra:

1. **"⋮" (3 nokta) → Settings**
2. **Secrets** sekmesine gidin
3. AlgoLab bilgilerinizi ekleyin (güvenli):
```toml
[algolab]
username = "your_username"
password = "your_password"
```

## Adım 5: URL'yi Paylaşın

App hazır olduğunda size bir URL verecek:
```
https://your-app-name.streamlit.app
```

Bu URL'yi arkadaşınızla paylaşın!

## Sorun Giderme

### "Module not found" hatası:
`requirements.txt` dosyasını kontrol edin

### "File not found" hatası:
Dosya yollarını kontrol edin (paper_trading/trading_monitor.py)

### App yüklenmiyor:
- GitHub repo'nun public olduğundan emin olun
- Branch adının "main" olduğunu kontrol edin

## Güvenlik Notları

1. **API anahtarlarını ASLA commit etmeyin!**
2. **Hassas bilgileri Streamlit Secrets kullanarak saklayın**
3. **Cache dosyalarını .gitignore'a ekleyin**

## Demo Mode

Eğer API bağlantısı olmadan çalıştırmak isterseniz, `trading_monitor.py` dosyasında mock data kullanın.

---

### Hızlı Kontrol Listesi:

- [ ] GitHub hesabı var mı?
- [ ] Repository oluşturuldu mu?
- [ ] Public olarak ayarlandı mı?
- [ ] Local repo GitHub'a bağlandı mı?
- [ ] Push yapıldı mı?
- [ ] Streamlit Cloud'a giriş yapıldı mı?
- [ ] App deploy edildi mi?
- [ ] URL arkadaşa gönderildi mi?

İşte bu kadar! 🚀