# Streamlit Dashboard Deployment Guide

## Option 1: Streamlit Community Cloud (Ücretsiz)

### Adımlar:

1. **GitHub'a Push Edin**
   ```bash
   git push origin main
   ```

2. **Streamlit Cloud'a Giriş**
   - https://share.streamlit.io/ adresine gidin
   - GitHub hesabınızla giriş yapın

3. **Yeni App Oluşturma**
   - "New app" butonuna tıklayın
   - Repository: `your-github-username/New_Start`
   - Branch: `main`
   - Main file path: `paper_trading/trading_monitor.py`

4. **Environment Variables (Opsiyonel)**
   - AlgoLab credentials için secrets ekleyin
   - Settings > Secrets kısmından

5. **Deploy**
   - Deploy butonuna tıklayın
   - URL'yi arkadaşınızla paylaşın

### Requirements.txt Oluşturma
```bash
cd /home/yunus/Belgeler/New_Start
pip freeze > requirements.txt
```

## Option 2: Ngrok ile Local Tunneling

### Kurulum:
```bash
# Ngrok kurulumu
wget https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz
tar -xzf ngrok-v3-stable-linux-amd64.tgz
sudo mv ngrok /usr/local/bin/

# Ngrok hesap oluşturun: https://ngrok.com/
# Auth token'ı alın ve yapılandırın
ngrok config add-authtoken YOUR_AUTH_TOKEN
```

### Kullanım:
```bash
# Terminal 1: Streamlit'i başlatın
cd paper_trading
streamlit run trading_monitor.py --server.port 8501

# Terminal 2: Ngrok tüneli açın
ngrok http 8501
```

Ngrok size bir URL verecek (örn: https://abc123.ngrok.io)
Bu URL'yi arkadaşınızla paylaşın.

## Option 3: Localtunnel (Daha Basit)

```bash
# Kurulum
npm install -g localtunnel

# Kullanım
cd paper_trading
streamlit run trading_monitor.py --server.port 8501 &
lt --port 8501 --subdomain my-trading-app
```

URL: https://my-trading-app.loca.lt

## Option 4: VPS'e Deploy (Kalıcı Çözüm)

### DigitalOcean/AWS/Hetzner vb. üzerinde:

1. **VPS Kirala** (5-10$/ay)

2. **Sunucuyu Hazırla**
```bash
# SSH ile bağlan
ssh root@your-server-ip

# Gerekli paketleri kur
apt update && apt upgrade -y
apt install python3-pip python3-venv git nginx -y

# Projeyi klonla
git clone https://github.com/yourusername/New_Start.git
cd New_Start

# Virtual environment oluştur
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. **Systemd Service Oluştur**
```bash
nano /etc/systemd/system/streamlit.service
```

İçerik:
```ini
[Unit]
Description=Streamlit Trading Monitor
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/New_Start/paper_trading
Environment="PATH=/root/New_Start/venv/bin"
ExecStart=/root/New_Start/venv/bin/streamlit run trading_monitor.py --server.port 8501 --server.address 0.0.0.0
Restart=always

[Install]
WantedBy=multi-user.target
```

4. **Servisi Başlat**
```bash
systemctl enable streamlit
systemctl start streamlit
```

5. **Nginx Proxy Ayarla**
```bash
nano /etc/nginx/sites-available/streamlit
```

İçerik:
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

```bash
ln -s /etc/nginx/sites-available/streamlit /etc/nginx/sites-enabled/
nginx -t
systemctl restart nginx
```

## Option 5: Docker ile Paylaşım

### Dockerfile Oluştur:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "paper_trading/trading_monitor.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Docker Image Oluştur ve Çalıştır:
```bash
docker build -t trading-monitor .
docker run -p 8501:8501 trading-monitor
```

## En Basit Yöntem: Streamlit Cloud

1. GitHub'a push edin
2. https://share.streamlit.io adresinden deploy edin
3. URL'yi paylaşın

Arkadaşınız hiçbir kurulum yapmadan tarayıcıdan açabilir!

## Güvenlik Notları:
- AlgoLab credentials'ları paylaşmayın
- Streamlit secrets kullanın
- HTTPS kullanın
- Rate limiting ekleyin

## Demo Mode:
Eğer sadece demo amaçlı göstermek istiyorsanız, `trading_monitor.py` içinde mock data kullanın ve API çağrılarını devre dışı bırakın.