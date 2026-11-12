import os, requests

def telegram_send(message):
 token=os.getenv('TELEGRAM_BOT_TOKEN','').strip();chat=os.getenv('TELEGRAM_CHAT_ID','').strip()
 if not token or not chat: raise RuntimeError('Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env')
 r=requests.post(f'https://api.telegram.org/bot{token}/sendMessage',json={'chat_id':chat,'text':message,'disable_web_page_preview':True},timeout=20);r.raise_for_status();data=r.json()
 if not data.get('ok'): raise RuntimeError(f'Telegram API error: {data}')
 return data
