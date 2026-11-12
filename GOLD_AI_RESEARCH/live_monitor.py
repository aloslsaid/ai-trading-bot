"""Live analysis loop. Sends Telegram alerts; NEVER places orders."""
import os,time,json,logging
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
from gold_ai.live_data import fetch_live_candles
from gold_ai.ml_engine import predict_latest
from gold_ai.indicators.classic import compute_classic
from gold_ai.notifications import telegram_send
from gold_ai.openrouter import ask_researcher
load_dotenv();logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')

def main():
 model=Path(os.getenv('MODEL_PATH','models/xau_m15.joblib'));interval=int(os.getenv('POLL_SECONDS','60'));tf=os.getenv('LIVE_INTERVAL','15min');minp=float(os.getenv('ALERT_MIN_PROB','0.65'));last_key=None
 if not model.exists():raise SystemExit(f'Model not found: {model}. Run python train_model.py first.')
 telegram_send('🟡 GOLD AI started. Live analysis only; no orders will be placed.')
 while True:
  try:
   df=fetch_live_candles(interval=tf,outputsize=1000)
   if len(df)<250:raise RuntimeError(f'Not enough live candles: {len(df)}')
   pred=predict_latest(df,model);feat=compute_classic(df).iloc[-1];price=float(df.close.iloc[-1])
   atr_value=float(feat.atr_14) if pd.notna(feat.atr_14) else price*0.002
   stop_distance=max(atr_value*1.5, price*0.0005)
   suggested_sl=price-stop_distance if pred['signal']=='BUY' else price+stop_distance
   suggested_tp=price+2*stop_distance if pred['signal']=='BUY' else price-2*stop_distance
   key=f"{df.index[-1]}:{pred['signal']}"
   logging.info('price=%.3f prediction=%s p_buy=%.3f',price,pred['signal'],pred['probability_buy'])
   if key!=last_key and ((pred['signal']=='BUY' and pred['probability_buy']>=minp) or (pred['signal']=='SELL' and pred['probability_sell']>=minp)):
    packet={'symbol':os.getenv('LIVE_SYMBOL','XAU/USD'),'time':str(df.index[-1]),'price':price,'prediction':pred,'indicators':{k:(float(feat[k]) if pd.notna(feat[k]) else None) for k in ['rsi_14','macd','macd_signal','adx','atr_14','bb_upper','bb_lower','supertrend_direction']}}
    ai=ask_researcher(packet)
    msg=(f"🚨 GOLD AI {pred['signal']} SIGNAL\nSymbol: {packet['symbol']}\n"
         f"Twelve Data provider candle close (not broker Bid/Ask): {price:.3f}\n"
         f"P(BUY): {pred['probability_buy']:.1%} | P(SELL): {pred['probability_sell']:.1%}\n"
         f"Indicative SL: {suggested_sl:.3f} | Indicative TP (2R): {suggested_tp:.3f}\n"
         f"RSI: {packet['indicators']['rsi_14']} | ADX: {packet['indicators']['adx']} | ATR: {packet['indicators']['atr_14']}\n"
         f"Time UTC: {packet['time']}\nAI Research: {ai[:2500]}\n\nResearch alert only. No entry/SL/TP is guaranteed; calculate broker-specific costs and risk manually.")
    telegram_send(msg);last_key=key
  except Exception as e:logging.exception('Live cycle failed: %s',e)
  time.sleep(max(15,interval))
if __name__=='__main__':main()
