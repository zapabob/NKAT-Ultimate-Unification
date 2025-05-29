# 🚀 NKAT理論 RTX3080 電源復旧システム 使用方法

## 📋 セットアップ完了項目

✅ 自動復旧スクリプト
✅ Windowsスタートアップ登録
✅ バッチファイル作成
✅ 設定ファイル生成
✅ デスクトップショートカット

## 🔄 電源復旧の流れ

### 1. 自動復旧（推奨）
- Windows起動時に自動実行
- 30秒待機後、前回の学習状況を検出
- 自動的に学習を再開

### 2. 手動復旧
- デスクトップの「NKATクイック復旧」をダブルクリック
- または以下のコマンドを実行:
  ```
  py -3 scripts/quick_recovery.py
  ```

### 3. 完全手動復旧
- 以下のコマンドで学習+ダッシュボード起動:
  ```
  py -3 scripts/run_rtx3080_training.py --mode both
  ```

## 📊 監視とアクセス

### ダッシュボード
- URL: http://localhost:8501
- リアルタイム学習状況監視
- GPU使用量・温度確認

### ログファイル
- 自動復旧ログ: logs/auto_recovery.log
- 電源管理ログ: logs/power_recovery.log
- 学習ログ: logs/rtx3080_training/

## ⚙️ 設定変更

### 自動復旧設定
ファイル: config/auto_recovery_config.json
- startup_delay_seconds: 起動待機時間
- gpu_temperature_threshold: GPU温度閾値
- vram_usage_threshold: VRAM使用量閾値

### 電源管理設定
ファイル: config/power_management_config.json
- safe_shutdown_delay_seconds: 安全シャットダウン待機時間
- checkpoint_save_interval_seconds: チェックポイント保存間隔

## 🛠️ トラブルシューティング

### 自動復旧が動作しない
1. Windowsスタートアップ確認:
   ```
   py -3 scripts/auto_recovery_startup.py --setup-startup
   ```

2. 手動テスト:
   ```
   py -3 scripts/auto_recovery_startup.py --check-only
   ```

### 学習が再開されない
1. チェックポイント確認:
   ```
   py -3 check_training_status.py
   ```

2. GPU状況確認:
   ```
   py -3 -c "import GPUtil; print(GPUtil.getGPUs()[0].name)"
   ```

## 📞 サポート

問題が発生した場合は、以下のログを確認してください:
- logs/auto_recovery.log
- logs/power_recovery.log
- logs/rtx3080_training/training.log

---
NKAT理論 RTX3080 電源復旧システム v1.0.0
