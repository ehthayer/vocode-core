import json
import os
import random
from datetime import datetime
from time import sleep
from typing import Optional

import libhoney
from libhoney import Client
from loguru import logger

from vocode.streaming.models.transcriber import Transcription
from vocode.streaming.transcriber.deepgram_transcriber import DeepgramTranscriptionResult


class E2ELatency():
    asr_latency: Optional[float] = None
    asr_queue_latency: Optional[float] = None
    agent_latency: Optional[float] = None
    agent_queue_latency: Optional[float] = None
    tts_latency: Optional[float] = None

    def stamp_asr_latency(self, latency: float):
        self.asr_latency = latency

    def stamp_asr_queue_latency(self, latency: float):
        self.asr_queue_latency = latency

    def stamp_agent_latency(self, latency: float):
        self.agent_latency = latency

    def stamp_agent_queue_latency(self, latency: float):
        self.agent_queue_latency = latency

    def stamp_tts_latency(self, latency: float):
        self.tts_latency = latency

    def clear(self):
        self.asr_latency = self.asr_queue_latency = self.agent_latency = self.agent_queue_latency = self.tts_latency = None


class CallEventReporter:
    client: Client
    turns: int
    stream_start_timestamp: float
    call_log_uuid: str
    conversation_id: str
    agent_begin: Optional[float] = None
    tts_begin:  Optional[float] = None
    deepgram_params: dict
    tts_provider: str
    tts_endpoint: str
    tts_config_json: str
    e2e_latency: E2ELatency = E2ELatency()

    def __init__(self):

        self.turns: int = 0

        self.client = libhoney.Client(
            writekey=os.getenv("HONEYCOMB_API_KEY"),
            dataset=os.getenv("HONEYCOMB_DATASET"),
            debug=True)
        self.client.add_field('app', 'telephony_app')

    def new_call(self, call_log_uuid: str, conversation_id: str):
        self.call_log_uuid = call_log_uuid
        self.conversation_id = conversation_id

    def terminate(self):
        logger.debug("Closing honeycomb client")
        self.client.close()

    def call_start(self, timestamp: float):
        self.stream_start_timestamp = timestamp

    def set_deepgram_params(self, params: dict):
        self.deepgram_params = {'deepgram_' + key: value for key, value in params.items()}

    def report_asr_latency(self, latency: float, audio_latency: float, final: bool, confidence: float):
        self.e2e_latency.stamp_asr_latency(latency)

        event = self.client.new_event()
        event.add_field('type', 'asr_latency')
        event.add_field('latency_sec', latency)
        event.add_field('audio_latency', audio_latency)

        event.add_field('turn', self.turns)
        event.add_field('call_log_uuid', self.call_log_uuid)
        event.add_field('conversation_id', self.conversation_id)
        event.add_field('final', final)
        event.add_field('confidence', confidence)
        if self.deepgram_params is not None:
            for item in self.deepgram_params.items():
                event.add_field(item[0], item[1])
        self.client.send(event)
        logger.debug(f'sent asr_latency {latency}')

    def report_asr_queue_latency(self, latency: float):
        event = self.client.new_event()
        event.add_field('type', 'asr_queue_latency')
        event.add_field('latency_sec', latency)

        event.add_field('turn', self.turns)
        event.add_field('call_log_uuid', self.call_log_uuid)
        event.add_field('conversation_id', self.conversation_id)
        event.send()
        logger.debug(f'sent asr_queue_latency')
        self.e2e_latency.stamp_asr_queue_latency(latency)

    def report_endpointing_latency(self, latency: float):
        event = self.client.new_event()
        event.add_field('type', 'endpointing_latency')
        event.add_field('latency_sec', latency)

        event.add_field('turn', self.turns)
        event.add_field('call_log_uuid', self.call_log_uuid)
        event.add_field('conversation_id', self.conversation_id)
        if self.deepgram_params is not None:
            for item in self.deepgram_params.items():
                event.add_field(item[0], item[1])

        self.client.send(event)

    def stamp_agent_begin(self):
        self.agent_begin = datetime.utcnow().timestamp()

    def report_agent_latency(self):
        if self.agent_begin is None:
            logger.warning('agent begin timestamp missing')
            return

        latency = datetime.utcnow().timestamp() - self.agent_begin
        self.e2e_latency.stamp_agent_latency(latency)

        event = self.client.new_event()
        event.add_field('type', 'agent_latency')
        event.add_field('latency_sec', latency)

        event.add_field('turn', self.turns)
        event.add_field('call_log_uuid', self.call_log_uuid)
        event.add_field('conversation_id', self.conversation_id)
        logger.debug(f'sent agent_latency {latency}')
        self.client.send(event)
        self.agent_begin = None

    def report_token_latency(self, latency: float, num: int):
        event = self.client.new_event()
        event.add_field('type', 'token_latency')
        event.add_field('latency_sec', latency)
        event.add_field('token_num', num)

        event.add_field('turn', self.turns)
        event.add_field('call_log_uuid', self.call_log_uuid)
        event.add_field('conversation_id', self.conversation_id)
        self.client.send(event)


    def report_agent_queue_latency(self, latency: float):
        event = self.client.new_event()
        event.add_field('type', 'agent_queue_latency')
        event.add_field('turn', self.turns)
        event.add_field('call_log_uuid', self.call_log_uuid)
        event.add_field('conversation_id', self.conversation_id)
        event.add_field('latency_sec', latency)
        event.send()
        logger.debug(f'sent agent_queue_latency')
        self.e2e_latency.stamp_agent_queue_latency(latency)

    def set_tts_endpoint(self, endpoint: str):
        self.tts_endpoint = endpoint

    def set_tts_config(self, provider: str, config_json: str):
        self.tts_config_json = config_json
        self.tts_provider = provider

    def stamp_tts_begin(self):
        self.tts_begin = datetime.utcnow().timestamp()

    def report_tts_latency(self):
        if self.tts_begin is None:
            logger.warning('tts begin timestamp missing')
            return

        latency = datetime.utcnow().timestamp() - self.tts_begin
        self.e2e_latency.stamp_tts_latency(latency)
        self.report_e2e_latency()

        event = self.client.new_event()
        event.add_field('type', 'tts_latency')

        event.add_field('turn', self.turns)
        event.add_field('call_log_uuid', self.call_log_uuid)
        event.add_field('conversation_id', self.conversation_id)
        event.add_field('latency_sec', latency)
#        event.add_field('tts_provider', self.tts_provider)
#        event.add_field('tts_config', self.tts_config_json)
#        event.add_field('tts_endpoint', self.tts_endpoint)
        logger.debug(f'sent tts_latency {latency}')
        self.client.send(event)

    def report_e2e_latency(self):
        if self.e2e_latency.asr_latency is None:
            # happens when internal agent responds.  e.g. "are you there?" on idle timeout
            return

        if any( x is None for x in [
            self.e2e_latency.asr_latency,
            self.e2e_latency.asr_queue_latency,
            self.e2e_latency.agent_latency,
            self.e2e_latency.agent_queue_latency,
            self.e2e_latency.tts_latency
        ]):
            logger.warning('Some latency values were None. Skipping e2e latency...')
            return

        event = self.client.new_event()
        event.add_field('type', 'e2e_latency')
        event.add_field('turn', self.turns)
        event.add_field('call_log_uuid', self.call_log_uuid)
        event.add_field('conversation_id', self.conversation_id)
        try:
            latency_sec = (self.e2e_latency.asr_latency +
                           self.e2e_latency.asr_queue_latency +
                           self.e2e_latency.agent_latency +
                           self.e2e_latency.agent_queue_latency +
                           self.e2e_latency.tts_latency)
        except Exception as e:
            logger.exception('Unexpected error computing latency_sec')

        event.add_field('latency_sec', latency_sec)
        event.client.send(event)
        self.e2e_latency.clear()
        logger.debug(f'sent e2e_latency {latency_sec}')


    def end_turn(self):
        self.turns += 1

    def report_rate_limit(self):
        event = self.client.new_event()
        event.add_field('turn', self.turns)
        event.add_field('call_log_uuid', self.call_log_uuid)
        event.add_field('conversation_id', self.conversation_id)
        event.add_field('rate_limited', "true")

    def report_transcription(self, transcription: Transcription, ignored=False, agent_input=False):
        try:
            event = self.client.new_event()
            event.add_field('turn', self.turns)
            event.add_field('call_log_uuid', self.call_log_uuid)
            event.add_field('conversation_id', self.conversation_id)
            event.add_field('ignored', ignored)
            event.add_field('agent_input', agent_input)
            event.add_field('type', 'transcription')
            event.add(dict(transcription))
            self.client.send(event)
        except Exception as e:
            logger.error(f"couldn't report transcription {e}")

    def report_deepgram_result(self, request_id: str, result: DeepgramTranscriptionResult):
        try:
            event = self.client.new_event()
            event.add_field('turn', self.turns)
            event.add_field('call_log_uuid', self.call_log_uuid)
            event.add_field('conversation_id', self.conversation_id)
            event.add_field('type', 'deepgram_result')
            event.add_field('request_id', request_id)
            event.add(dict(result))
            self.client.send(event)
        except Exception as e:
            logger.error(f"couldn't report deepgram_result {e}")


    def report_call(self):
        event = self.client.new_event()
        event.add_field('type', 'call_summary')
        event.add_field('call_duration_sec', datetime.utcnow().timestamp() - self.stream_start_timestamp)
        event.add_field('num_turns', self.turns)
        event.add_field('call_log_uuid', self.call_log_uuid)
        event.add_field('conversation_id', self.conversation_id)
        logger.debug(f'sent call_summary')
        self.client.send(event)


if __name__ == "__main__":
    print('*** test ***')
    call_event_reporter = CallEventReporter()
    for c in range(3):
        call_event_reporter.new_call(f"unit_test_uuid_{c}", f"unit_test_conversation_id_{c}")
        call_event_reporter.call_start(datetime.utcnow().timestamp())
        call_event_reporter.set_deepgram_params({})
        for i in range(5):
            call_event_reporter.report_asr_latency(0.5, 0.0, True, 0.9)
            call_event_reporter.stamp_agent_begin()
            sleep(random.uniform(0.25, 0.75))
            call_event_reporter.report_asr_queue_latency(0.001)
            call_event_reporter.report_agent_latency()
            call_event_reporter.report_agent_queue_latency(0.001)
            call_event_reporter.stamp_tts_begin()
            sleep(random.uniform(0.25, 0.75))
            call_event_reporter.report_tts_latency()
            call_event_reporter.end_turn()
        call_event_reporter.report_call()
