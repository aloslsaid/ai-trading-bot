from dotenv import load_dotenv
from gold_ai.notifications import telegram_send
load_dotenv();telegram_send('✅ GOLD AI Telegram connection test successful.')
print('Telegram test sent.')
