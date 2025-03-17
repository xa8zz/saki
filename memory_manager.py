# memory_manager.py
import time
import requests
from google import genai
from PIL import Image
import io
from datetime import datetime, timezone, timedelta
from config import (
    GEMINI_API_KEY,
    GEMINI_LIGHT_MODEL,
    MESSAGE_SUMMARIZE_THRESHOLD,
    CONTEXT_SUMMARIZE_TRIGGER,
    MAX_TOKEN_LIMIT,
    SETUP_MESSAGES
)

def get_est_time():
    utc_time = datetime.now(timezone.utc)
    est_offset = timedelta(hours=-5)
    est_time = utc_time.astimezone(timezone(est_offset))
    return est_time

def format_est_time(dt=None):
    """Format a datetime object as EST time string or get current EST time"""
    if dt is None:
        dt = get_est_time()
    elif dt.tzinfo is None:
        # If the datetime has no timezone, assume it's UTC and convert to EST
        dt = dt.replace(tzinfo=timezone.utc).astimezone(timezone(timedelta(hours=-5)))
    elif dt.tzinfo != timezone(timedelta(hours=-5)):
        # If datetime has a different timezone, convert to EST
        dt = dt.astimezone(timezone(timedelta(hours=-5)))
    return dt.strftime("%Y-%m-%d %H:%M:%S")

class MemoryManager:
    """
    Memory manager that stores conversation histories per channel (or DM).
    Summarizes older conversation slices to keep token usage manageable.
    """

    def __init__(self):
        # Dictionary for channel_id -> list of message dicts
        # Each entry: {
        #   'role': 'user' or 'assistant',
        #   'content': 'the message text',
        #   'time': 'timestamp string'
        # }
        self.histories = {}
        # We'll store token usage estimates for each channel separately
        self.token_usage = {}

    def add_message(self, channel_id, role, content, timestamp=None):
        """Add a message to the channel history.
        
        Args:
            channel_id: A unique identifier for the channel (string or int)
            role: Either 'user' or 'assistant'
            content: The message content
            timestamp: Optional timestamp string. If None, current EST time is used.
        """
        if timestamp is None:
            timestamp = format_est_time(get_est_time())
        
        message = {
            'role': role,
            'content': content,
            'time': timestamp
        }
        
        # Ensure there is a history list for the given channel_id
        if channel_id not in self.histories:
            self.histories[channel_id] = []
            self.token_usage[channel_id] = 0  # Initialize token usage as well
        
        self.histories[channel_id].append(message)
        # Optionally, update token usage:
        self.token_usage[channel_id] += len(content.split())
        
        # Check if we need to summarize
        self.maybe_summarize(channel_id)


    def maybe_summarize(self, channel_id):
        if channel_id not in self.histories:
            return
        if len(self.histories[channel_id]) < MESSAGE_SUMMARIZE_THRESHOLD:
            return
        if self.token_usage[channel_id] < CONTEXT_SUMMARIZE_TRIGGER:
            return

        older_segment = self.histories[channel_id][:MESSAGE_SUMMARIZE_THRESHOLD]
        summary_text = self._summarize_chunk(older_segment)

        self.histories[channel_id] = [{
            'role': 'assistant',
            'content': f"(SUMMARY) {summary_text}",
            'time': format_est_time()
        }] + self.histories[channel_id][MESSAGE_SUMMARIZE_THRESHOLD:]

        new_token_usage = sum(len(msg["content"].split()) for msg in self.histories[channel_id])
        self.token_usage[channel_id] = new_token_usage

    def _summarize_chunk(self, messages_chunk):
        """
        Summarizes a given list of messages using Gemini's light model.
        Uses the official Python SDK to generate a summary.
        """
        # Create a user prompt with just the messages to summarize
        user_content = "Summarize these messages:\n\n"
        for m in messages_chunk:
            user_content += f"{m['time']} - {m['role'].upper()}: {m['content']}\n"
        
        # Create a system instruction for the summary task
        summary_system_instruction = """
        You are an assistant that creates concise and accurate summaries of conversation snippets.
        Focus on the key details, main topics, and any important decisions or information shared.
        Keep summaries under 50 words and include only the essential information.
        Your summary should cover the general topic and any significant details without unnecessary elaboration.
        """
        
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model=GEMINI_LIGHT_MODEL,
            contents=[{"role": "user", "parts": [{"text": user_content}]}],
            config={
                "system_instruction": summary_system_instruction,
                "temperature": 0.2, 
                "max_output_tokens": 120
            }
        )

        summary = response.text.strip()
        return summary



