import os, requests

def telegram_send(message):
 token=os.getenv('TELEGRAM_BOT_TOKEN','').strip();chat=os.getenv('TELEGRAM_CHAT_ID','').strip()
 if not token or not chat: raise RuntimeError('Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env')
 try:
  r=requests.post(f'https://api.telegram.org/bot{token}/sendMessage',json={'chat_id':chat,'text':message,'disable_web_page_preview':True},timeout=20)
 except requests.RequestException as exc:
  # The bot token is embedded in the Telegram URL; never echo request exceptions.
  raise RuntimeError(f'Telegram request failed ({type(exc).__name__}); check network and credentials.') from None
 if r.status_code >= 400:
  raise RuntimeError(f'Telegram returned HTTP {r.status_code}; response details were redacted.')
 data=r.json()
 if not data.get('ok'): raise RuntimeError('Telegram API rejected the message; response details were redacted.')
 return data
