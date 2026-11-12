"""Causal classic technical indicators for XAUUSD. Uses mid-price OHLC."""
from __future__ import annotations
import numpy as np
import pandas as pd

def mid_ohlc(df):
 if {'bid_open','ask_open','bid_high','ask_high','bid_low','ask_low','bid_close','ask_close'}.issubset(df.columns):
  return pd.DataFrame({'open':(df.bid_open+df.ask_open)/2,'high':(df.bid_high+df.ask_high)/2,'low':(df.bid_low+df.ask_low)/2,'close':(df.bid_close+df.ask_close)/2,'volume':df.volume},index=df.index)
 if {'open','high','low','close'}.issubset(df.columns):
  return df[['open','high','low','close','volume']].copy()
 raise ValueError('OHLC input must contain either Bid/Ask candles or provider OHLC candles')
def sma(s,n=20): return s.rolling(n,min_periods=n).mean()
def ema(s,n=20): return s.ewm(span=n,adjust=False,min_periods=n).mean()
def _w(s,n): return s.ewm(alpha=1/n,adjust=False,min_periods=n).mean()
def rsi(c,n=14):
 d=c.diff();u=_w(d.clip(lower=0),n);dn=_w((-d).clip(lower=0),n);rs=u/dn.replace(0,np.nan);return (100-100/(1+rs)).where(dn!=0,100.).where(u.notna())
def macd(c,fast=12,slow=26,signal=9):
 line=ema(c,fast)-ema(c,slow);sig=line.ewm(span=signal,adjust=False,min_periods=signal).mean();return pd.DataFrame({'macd':line,'macd_signal':sig,'macd_hist':line-sig})
def true_range(h,l,c):
 pc=c.shift();return pd.concat([h-l,(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1)
def atr(h,l,c,n=14): return _w(true_range(h,l,c),n)
def adx(h,l,c,n=14):
 up,dn=h.diff(),-l.diff();p=pd.Series(np.where((up>dn)&(up>0),up,0.),index=h.index);m=pd.Series(np.where((dn>up)&(dn>0),dn,0.),index=h.index);a=atr(h,l,c,n);pdi=100*_w(p,n)/a;mdi=100*_w(m,n)/a;dx=100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan);return pd.DataFrame({'adx':_w(dx.fillna(0),n).where(a.notna()),'plus_di':pdi,'minus_di':mdi})
def bollinger(c,n=20,k=2):
 mid=sma(c,n);sd=c.rolling(n,min_periods=n).std(ddof=0);up=mid+k*sd;lo=mid-k*sd;return pd.DataFrame({'bb_mid':mid,'bb_upper':up,'bb_lower':lo,'bb_pctb':(c-lo)/(up-lo).replace(0,np.nan),'bb_width':(up-lo)/mid})
def stochastic(h,l,c,k=14,d=3):
 hh=h.rolling(k,min_periods=k).max();ll=l.rolling(k,min_periods=k).min();pk=100*(c-ll)/(hh-ll).replace(0,np.nan);return pd.DataFrame({'stoch_k':pk,'stoch_d':pk.rolling(d,min_periods=d).mean()})
def cci(h,l,c,n=20):
 tp=(h+l+c)/3;ma=tp.rolling(n,min_periods=n).mean();md=tp.rolling(n,min_periods=n).apply(lambda x:np.abs(x-x.mean()).mean(),raw=True);return (tp-ma)/(.015*md.replace(0,np.nan))
def roc(c,n=12): return 100*(c/c.shift(n)-1)
def williams_r(h,l,c,n=14):
 hh=h.rolling(n,min_periods=n).max();ll=l.rolling(n,min_periods=n).min();return -100*(hh-c)/(hh-ll).replace(0,np.nan)
def obv(c,v): return (np.sign(c.diff()).fillna(0)*v).cumsum()
def vwap_daily(h,l,c,v):
 tp=(h+l+c)/3;day=tp.index.floor('D');return (tp*v).groupby(day).cumsum()/v.groupby(day).cumsum().replace(0,np.nan)
def mfi(h,l,c,v,n=14):
 tp=(h+l+c)/3;flow=tp*v;pos=flow.where(tp.diff()>0,0).rolling(n).sum();neg=flow.where(tp.diff()<0,0).rolling(n).sum();r=pos/neg.replace(0,np.nan);return (100-100/(1+r)).where(neg!=0,100.)
def cmf(h,l,c,v,n=20):
 mfm=((c-l)-(h-c))/(h-l).replace(0,np.nan);return (mfm*v).rolling(n).sum()/v.rolling(n).sum().replace(0,np.nan)
def donchian(h,l,n=20): return pd.DataFrame({'donchian_upper':h.rolling(n).max(),'donchian_lower':l.rolling(n).min(),'donchian_mid':(h.rolling(n).max()+l.rolling(n).min())/2})
def keltner(h,l,c,n=20,mult=2):
 mid=ema(c,n);a=atr(h,l,c,10);return pd.DataFrame({'keltner_mid':mid,'keltner_upper':mid+mult*a,'keltner_lower':mid-mult*a})
def ichimoku(h,l):
 ten=(h.rolling(9).max()+l.rolling(9).min())/2;kij=(h.rolling(26).max()+l.rolling(26).min())/2;sa=(ten+kij)/2;sb=(h.rolling(52).max()+l.rolling(52).min())/2
 # No forward shift: features remain aligned and causal.
 return pd.DataFrame({'ichi_tenkan':ten,'ichi_kijun':kij,'ichi_senkou_a_unshifted':sa,'ichi_senkou_b_unshifted':sb})
def aroon(h,l,n=25):
 return pd.DataFrame({'aroon_up':h.rolling(n+1).apply(lambda x:100*(np.argmax(x)/n),raw=True),'aroon_down':l.rolling(n+1).apply(lambda x:100*(np.argmin(x)/n),raw=True)})
def psar(h,l,step=.02,maximum=.2):
 # Compact iterative Parabolic SAR implementation.
 out=np.full(len(h),np.nan);hh=h.to_numpy();ll=l.to_numpy()
 if len(h)<2:return pd.Series(out,index=h.index)
 bull=True;ep=hh[0];sar=ll[0];af=step
 for i in range(1,len(h)):
  sar=sar+af*(ep-sar)
  if bull:
   sar=min(sar,ll[i-1],ll[i-2] if i>1 else ll[i-1])
   if ll[i]<sar: bull=False;sar=ep;ep=ll[i];af=step
   elif hh[i]>ep: ep=hh[i];af=min(maximum,af+step)
  else:
   sar=max(sar,hh[i-1],hh[i-2] if i>1 else hh[i-1])
   if hh[i]>sar: bull=True;sar=ep;ep=hh[i];af=step
   elif ll[i]<ep: ep=ll[i];af=min(maximum,af+step)
  out[i]=sar
 return pd.Series(out,index=h.index,name='psar')
def supertrend(h,l,c,n=10,mult=3):
 a=atr(h,l,c,n);mid=(h+l)/2;ub=mid+mult*a;lb=mid-mult*a;fu=ub.copy();fl=lb.copy();st=pd.Series(np.nan,index=c.index);trend=pd.Series(0,index=c.index,dtype=int)
 for i in range(1,len(c)):
  if pd.isna(a.iloc[i]):continue
  fu.iloc[i]=ub.iloc[i] if (ub.iloc[i]<fu.iloc[i-1] or c.iloc[i-1]>fu.iloc[i-1]) else fu.iloc[i-1]
  fl.iloc[i]=lb.iloc[i] if (lb.iloc[i]>fl.iloc[i-1] or c.iloc[i-1]<fl.iloc[i-1]) else fl.iloc[i-1]
  if pd.isna(st.iloc[i-1]): trend.iloc[i]=1 if c.iloc[i]>=mid.iloc[i] else -1
  elif st.iloc[i-1]==fu.iloc[i-1]: trend.iloc[i]=1 if c.iloc[i]>fu.iloc[i] else -1
  else: trend.iloc[i]=-1 if c.iloc[i]<fl.iloc[i] else 1
  st.iloc[i]=fl.iloc[i] if trend.iloc[i]>0 else fu.iloc[i]
 return pd.DataFrame({'supertrend':st,'supertrend_direction':trend})
def compute_classic(df):
 m=mid_ohlc(df);h,l,c,v=m.high,m.low,m.close,m.volume;out=m.copy();out['spread']=df['spread_close'] if 'spread_close' in df else np.nan
 for n in (10,20,50,100,200):out[f'sma_{n}']=sma(c,n)
 for n in (9,12,21,26,50,100,200):out[f'ema_{n}']=ema(c,n)
 out['rsi_14']=rsi(c);out=out.join(macd(c)).join(adx(h,l,c)).join(bollinger(c)).join(stochastic(h,l,c))
 out['atr_14']=atr(h,l,c);out['cci_20']=cci(h,l,c);out['roc_12']=roc(c);out['willr_14']=williams_r(h,l,c);out['obv']=obv(c,v);out['vwap']=vwap_daily(h,l,c,v)
 out['mfi_14']=mfi(h,l,c,v);out['cmf_20']=cmf(h,l,c,v);out=out.join(donchian(h,l)).join(keltner(h,l,c)).join(ichimoku(h,l)).join(aroon(h,l)).join(supertrend(h,l,c));out['psar']=psar(h,l)
 out['wma_20']=c.rolling(20).apply(lambda x:np.dot(x,np.arange(1,21))/210,raw=True)
 out['momentum_10']=c-c.shift(10);out['stddev_20']=c.rolling(20).std();out['ulcer_index_14']=np.sqrt(((100*(c/c.rolling(14).max()-1))**2).rolling(14).mean())
 return out
