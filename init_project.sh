#!/bin/bash
# Proje başlatma scripti - Her açılışta çalıştırılabilir

echo "🚀 Initializing BIST100 Trading System..."

# Claude ayarlarını kontrol et ve gerekirse geri yükle
if [ ! -s ".claude/settings.local.json" ] || [ $(wc -c < .claude/settings.local.json) -lt 100 ]; then
    echo "⚠️  Claude settings missing or corrupted, restoring from backup..."
    if [ -f "CLAUDE_SETTINGS_BACKUP.json" ]; then
        cp CLAUDE_SETTINGS_BACKUP.json .claude/settings.local.json
        chmod 644 .claude/settings.local.json
        echo "✅ Claude settings restored!"
    else
        echo "❌ No backup found! Please configure Claude settings manually."
    fi
else
    echo "✅ Claude settings OK"
fi

# Virtual environment kontrolü
if [ -d "venv" ]; then
    echo "✅ Virtual environment exists"
else
    echo "⚠️  Creating virtual environment..."
    python3 -m venv venv
fi

# Klasör yapısı kontrolü
directories=("backtest" "config" "core" "data/raw" "data/indicators" "data/analysis" "data/predictions" "indicators" "logs" "ml_models" "portfolio" "strategies" "trading" "utils" "scripts")

for dir in "${directories[@]}"; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        echo "📁 Created directory: $dir"
    fi
done

# __init__.py dosyaları kontrolü
init_dirs=("backtest" "config" "core" "data" "indicators" "ml_models" "portfolio" "strategies" "trading" "utils")

for dir in "${init_dirs[@]}"; do
    if [ ! -f "$dir/__init__.py" ]; then
        touch "$dir/__init__.py"
        echo "📄 Created __init__.py in $dir"
    fi
done

echo ""
echo "✅ Project initialization complete!"
echo ""
echo "Next steps:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Install dependencies: pip install -r requirements.txt"
echo "3. Copy .env.example to .env and configure"
echo "4. Start developing!"