import time
import logging
import sys
import os
from datetime import datetime
import traceback
from dotenv import load_dotenv

# Configurar encoding para Windows para evitar erros com emojis no terminal
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Adicionar diretório src ao path para permitir imports internos
sys.path.append(os.getcwd())

from src.trading_bot_multi import trade_logic_multi

# Configuração de Logs
# Cria o diretório 'logs' caso não exista
os.makedirs("logs", exist_ok=True)

# Define o formato do log para salvar em arquivo e mostrar no terminal simultaneamente
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/trading_bot.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger()

def run_continuous():
    """
    Loop principal do Bot de Trading.
    Mantém o bot rodando de forma contínua, sincronizando-se com o fechamento
    dos candles de hora em hora para executar a análise e disparar ordens.
    """
    logger.info("🚀 Iniciando Bot de Trading Multi-Ativo (BTC+ETH)...")
    logger.info("Pressione Ctrl+C para parar.")
    
    while True:
        try:
            # Calcular o tempo exato restante até a próxima virada de hora.
            # Como o modelo usa dados H1 (1 hora), ele só precisa tomar decisão
            # no momento exato em que um novo candle abre.
            now = datetime.now()
            next_hour = now.replace(minute=0, second=0, microsecond=0).timestamp() + 3600
            current_ts = now.timestamp()
            
            # Adiciona 60 segundos de margem de segurança para garantir que
            # a corretora (Alpaca) já fechou e disponibilizou o candle da última hora.
            sleep_seconds = next_hour - current_ts + 60 
            
            logger.info(f"⏳ Aguardando {int(sleep_seconds/60)} minutos para o próximo candle...")
            
            # Colocar a thread em pausa até o horário calculado
            time.sleep(sleep_seconds) 
            
            # Acorda e executa a lógica principal de trading (ML + IA + Risco)
            logger.info("🔄 Executando análise de mercado (Multi-Ativo)...")
            trade_logic_multi()
            logger.info("✅ Análise concluída.")
            
        except KeyboardInterrupt:
            # Interrupção manual pelo usuário (Ctrl+C)
            logger.info("👋 Bot parado pelo usuário.")
            break
        except Exception as e:
            # Captura de erros inesperados para evitar que o bot crashe silenciosamente
            logger.error(f"❌ Erro crítico no loop principal: {e}")
            logger.error(traceback.format_exc())
            
            # Em caso de falha de API ou erro na rede, espera 1 minuto antes 
            # de tentar rodar o loop novamente, prevenindo flood de requisições.
            time.sleep(60)

if __name__ == "__main__":
    run_continuous()
