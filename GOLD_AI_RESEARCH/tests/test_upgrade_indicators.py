import numpy as np, pandas as pd
from gold_ai.indicators.classic import compute_classic

def test_expanded_indicators_exist_and_are_causal_shape():
 n=400;idx=pd.date_range('2024-01-01',periods=n,freq='min',tz='UTC');c=2000+np.cumsum(np.random.default_rng(3).normal(0,.2,n))
 d=pd.DataFrame(index=idx)
 for side,offset in [('bid',0),('ask',.2)]:
  for k,m in [('open',0),('high',.3),('low',-.3),('close',0)]:d[f'{side}_{k}']=c+offset+m
 d['volume']=100.;d['spread_close']=.2
 f=compute_classic(d)
 for col in ['bb_upper','bb_lower','macd','rsi_14','mfi_14','cmf_20','donchian_upper','keltner_upper','ichi_tenkan','aroon_up','psar','supertrend','wma_20','ulcer_index_14']:
  assert col in f.columns
 assert len(f)==n
