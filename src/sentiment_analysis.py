import os
import requests
import json
from dotenv import load_dotenv

# Carregar variáveis de ambiente (chaves de API)
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
# Modelo da Google utilizado para análise de sentimento via OpenRouter
OPENROUTER_MODEL = "google/gemini-2.5-flash-lite" 

def analyze_sentiment(news_headlines):
    """
    Analisa o sentimento de uma lista de manchetes usando o modelo LLM Gemini via OpenRouter.
    
    Esta função envia as manchetes mais recentes do mercado para a Inteligência Artificial 
    e pede que ela classifique o sentimento geral do mercado cripto. O resultado é um 
    número usado como um 'filtro de segurança' pelo bot de trade.
    
    Args:
        news_headlines (list of str): Lista contendo as strings das manchetes.
        
    Returns:
        float: Um score numérico de -1.0 (Muito Negativo/Bearish) a 1.0 (Muito Positivo/Bullish).
               Retorna 0.0 em caso de erro ou se não houver notícias (Neutro).
    """
    if not news_headlines:
        return 0.0 # Sentimento neutro se não houver notícias para analisar

    # Preparar o Prompt para o LLM
    headlines_text = "\n".join([f"- {h}" for h in news_headlines])
    prompt = f"""
    Analise as seguintes manchetes financeiras sobre Bitcoin e Criptomoedas:
    
    {headlines_text}
    
    Classifique o sentimento geral do mercado com base nelas.
    Responda APENAS com um número float entre -1.0 (Extremamente Pessimista/Bearish) e 1.0 (Extremamente Otimista/Bullish).
    Não dê explicações, apenas o número.
    """

    # Headers de autenticação da OpenRouter
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        # "HTTP-Referer": "https://meusite.com", # Opcional para OpenRouter
        # "X-Title": "ML Trading Bot", # Opcional
    }
    
    # Payload da requisição
    data = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1 # Temperatura baixa para garantir respostas mais determinísticas (menos criatividade)
    }
    
    try:
        print(f"🧠 Consultando Gemini ({OPENROUTER_MODEL}) via OpenRouter...")
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=15
        )
        
        if response.status_code == 200:
            result = response.json()
            # Extrair o conteúdo da resposta da IA
            content = result['choices'][0]['message']['content'].strip()
            
            # Tentar converter a resposta textual (ex: "0.8") para um número float real
            try:
                score = float(content)
                # Garantir que o score fique estritamente dentro dos limites [-1.0, 1.0]
                score = max(-1.0, min(1.0, score))
                print(f"✅ Sentimento Calculado: {score}")
                return score
            except ValueError:
                print(f"⚠️ Resposta não numérica do Gemini: {content}")
                return 0.0
        else:
            print(f"❌ Erro na API OpenRouter: {response.status_code} - {response.text}")
            return 0.0
            
    except Exception as e:
        print(f"❌ Erro na requisição de sentimento: {e}")
        return 0.0

if __name__ == "__main__":
    # Teste Rápido para verificar se a API e o Prompt estão funcionando corretamente
    manchetes_teste = [
        "Bitcoin atinge nova máxima histórica acima de $75.000",
        "Investidores institucionais aumentam exposição em cripto",
        "SEC aprova novo ETF de Bitcoin Spot"
    ]
    print(f"Teste Positivo: {analyze_sentiment(manchetes_teste)}")
    
    manchetes_negativas = [
        "Binance sofre processo regulatório em 3 países",
        "Bitcoin cai 10% após discurso do Fed sobre juros",
        "Hackers roubam $50 milhões de protocolo DeFi"
    ]
    print(f"Teste Negativo: {analyze_sentiment(manchetes_negativas)}")
