# Claude Code Configuration

Bu dosya, Claude Code'un proje ayarlarını içerir ve .claude/settings.local.json dosyasının yedeğidir.

## Önemli Notlar

1. `.claude/settings.local.json` dosyası bazen sıfırlanabilir
2. Bu durumda aşağıdaki komutu çalıştırın:
   ```bash
   cp CLAUDE_SETTINGS_BACKUP.json .claude/settings.local.json
   ```

## Ayarların Korunması

1. Her değişiklikten sonra yedek alın:
   ```bash
   cp .claude/settings.local.json CLAUDE_SETTINGS_BACKUP.json
   ```

2. Git'e ekleyin (opsiyonel):
   ```bash
   git add .claude/settings.local.json CLAUDE_SETTINGS_BACKUP.json
   git commit -m "Update Claude settings"
   ```

## Otomatik Yedekleme Script

```bash
#!/bin/bash
# save_claude_settings.sh
cp .claude/settings.local.json CLAUDE_SETTINGS_BACKUP.json
echo "Claude settings backed up successfully!"
```

## Otomatik Geri Yükleme Script

```bash
#!/bin/bash
# restore_claude_settings.sh
if [ -f "CLAUDE_SETTINGS_BACKUP.json" ]; then
    cp CLAUDE_SETTINGS_BACKUP.json .claude/settings.local.json
    echo "Claude settings restored successfully!"
else
    echo "Backup file not found!"
fi
```

## Trading System Rules

Bu proje README_GOD_MODE.md kurallarına göre yapılandırılmıştır:
- Kutsal 20 sembol
- Core 5 indikatör
- Sistem limitleri
- Risk yönetimi kuralları

Detaylar için README_GOD_MODE.md dosyasına bakın.