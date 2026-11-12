import os, requests, json

def ask_researcher(payload):
 key=os.getenv('OPENROUTER_API_KEY','').strip();model=os.getenv('OPENROUTER_MODEL','').strip()
 if not key or not model: return 'OpenRouter is not configured; deterministic ML/rule signals remain available.'
 prompt=('You are a cautious XAUUSD research assistant. Analyze supplied market features and model output. '
 'Do not claim certainty or invent news/prices. Return concise JSON with bias (BUY/SELL/HOLD), rationale, '
 'risks, invalidation, and confidence 0-100. Treat this as research, not a trade instruction. Data: '+json.dumps(payload,default=str))
 r=requests.post('https://openrouter.ai/api/v1/chat/completions',headers={'Authorization':f'Bearer {key}','Content-Type':'application/json','HTTP-Referer':'https://localhost','X-Title':'Gold AI Research'},json={'model':model,'messages':[{'role':'user','content':prompt}],'temperature':0.2},timeout=60);r.raise_for_status();return r.json()['choices'][0]['message']['content']
