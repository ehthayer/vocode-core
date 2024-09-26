# Standard library imports
import os
import sys

from dotenv import load_dotenv

# Third-party imports
from fastapi import FastAPI
from loguru import logger
from pyngrok import ngrok

# Local application/library specific imports
# from speller_agent import SpellerAgentFactory

from vocode.logging import configure_pretty_logging
from vocode.streaming.models.agent import ChatGPTAgentConfig
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.models.synthesizer import ElevenLabsSynthesizerConfig
from vocode.streaming.models.telephony import TwilioConfig
from vocode.streaming.models.transcriber import DeepgramTranscriberConfig
from vocode.streaming.telephony.config_manager.redis_config_manager import RedisConfigManager
from vocode.streaming.telephony.constants import DEFAULT_SAMPLING_RATE, DEFAULT_AUDIO_ENCODING, DEFAULT_CHUNK_SIZE
from vocode.streaming.telephony.server.base import TelephonyServer, TwilioInboundCallConfig
from vocode.streaming.transcriber.deepgram_transcriber import DeepgramEndpointingConfig, TimeSilentConfig

# if running from python, this will load the local .env
# docker-compose will load the .env file by itself
load_dotenv()

configure_pretty_logging()

app = FastAPI(docs_url=None)

config_manager = RedisConfigManager()

BASE_URL = os.getenv("BASE_URL")

if not BASE_URL:
    ngrok_auth = os.environ.get("NGROK_AUTH_TOKEN")
    if ngrok_auth is not None:
        ngrok.set_auth_token(ngrok_auth)
    port = sys.argv[sys.argv.index("--port") + 1] if "--port" in sys.argv else 3000

    # Open a ngrok tunnel to the dev server
    BASE_URL = ngrok.connect(port).public_url.replace("https://", "")
    logger.info('ngrok tunnel "{}" -> "http://127.0.0.1:{}"'.format(BASE_URL, port))

if not BASE_URL:
    raise ValueError("BASE_URL must be set in environment if not using pyngrok")

telephony_server = TelephonyServer(
    base_url=BASE_URL,
    config_manager=config_manager,
    inbound_call_configs=[
        TwilioInboundCallConfig(
            url="/inbound_call",
            agent_config=ChatGPTAgentConfig(
                initial_message=BaseMessage(text="What up"),
                prompt_preamble="Have a pleasant conversation about life",
                generate_responses=True,
            ),
            synthesizer_config=ElevenLabsSynthesizerConfig(
                api_key=os.getenv("ELEVEN_LABS_API_KEY"),
                similarity_boost=1,
                stability=1,
                style=1,
                use_speaker_boost=True,
                voice_id=os.getenv("ELEVEN_LABS_VOICE_ID"),
                publish_audio=True,
                experimental_websocket=True,
                experimental_streaming=True,
                optimize_streaming_latency=0,
                model_id="eleven_turbo_v2",
                sampling_rate=8000,
                audio_encoding='mulaw',
            ),
            transcriber_config=DeepgramTranscriberConfig(
                sampling_rate=DEFAULT_SAMPLING_RATE,
                audio_encoding=DEFAULT_AUDIO_ENCODING,
                chunk_size=DEFAULT_CHUNK_SIZE,
                model="nova-2-phonecall",
                min_interrupt_confidence=0.95,
                endpointing_config=DeepgramEndpointingConfig(
                    utterance_cutoff_ms=750,
                    vad_threshold_ms=400,
                    time_silent_config=TimeSilentConfig(
                        time_cutoff_seconds=0.75,
                        post_punctuation_time_seconds=0.5
                    )
                )
            ),
            # uncomment this to use the speller agent instead
            # agent_config=SpellerAgentConfig(
            #     initial_message=BaseMessage(
            #         text="im a speller agent, say something to me and ill spell it out for you"
            #     ),
            #     generate_responses=False,
            # ),
            twilio_config=TwilioConfig(
                account_sid=os.environ["TWILIO_ACCOUNT_SID"],
                auth_token=os.environ["TWILIO_AUTH_TOKEN"],
            ),
        )
    ],
    # agent_factory=SpellerAgentFactory(),
)

app.include_router(telephony_server.get_router())
