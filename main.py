# main.py
import discord
import asyncio
import time
import re
import json
import io
import aiohttp
import os
import base64
import random
from google.genai.types import GenerateContentConfig
from google import genai
from PIL import Image
import io
from datetime import datetime, timedelta, timezone
from discord import File
from config import (
    DISCORD_BOT_TOKEN,
    SYSTEM_INSTRUCTION,
    SETUP_MESSAGES,
    TYPING_DELAY_PER_WORD,
    GEMINI_API_KEY,
    GEMINI_HEAVY_MODEL,
    GEMINI_LIGHT_MODEL,
    GEMINI_SEARCH_MODEL,
    MAX_TOKENS_CURRENT,
    SEARCH_SYSTEM_INSTRUCTION,
    SEARCH_CONTEXT_MESSAGE_COUNT
)
from memory_manager import MemoryManager
import requests

# Define EST timezone (UTC-5)
EST = timezone(timedelta(hours=-5))

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


last_bot_message = {}
context_save_file = "saved_contexts.json"
attachment_transcriptions_file = "attachment_transcriptions.json"
log_config_file = "log_config.json"  # File to store log configuration
attachment_transcriptions = {}  # Format: {attachment_url: {'type': 'image/video', 'transcription': 'text'}}

# Log channel configuration
log_channel_id = "1345381968950591539"  # Default log channel ID
use_log_channel = True  # Flag to control whether logs are sent to Discord
detailed_logging = False  # Flag for verbose logging
log_message_queue = []  # Queue to store messages before sending to Discord
log_queue_lock = asyncio.Lock()  # Lock for thread-safe queue operations
log_task_running = False  # Flag to track if the log sending task is running

# Track user's messages for potential !reload with edits
user_messages = {}  # Format: {channel_id: [{'message_object': message, 'content': content, 'time': timestamp}]}

# Load log configuration if file exists
def load_log_config():
    global log_channel_id, use_log_channel, detailed_logging
    if os.path.exists(log_config_file):
        try:
            with open(log_config_file, 'r') as f:
                config = json.load(f)
                log_channel_id = config.get('log_channel_id', log_channel_id)
                use_log_channel = config.get('use_log_channel', use_log_channel)
                detailed_logging = config.get('detailed_logging', detailed_logging)
                print(f"Loaded log configuration: channel={log_channel_id}, enabled={use_log_channel}, verbose={detailed_logging}")
        except json.JSONDecodeError:
            print("Error loading log configuration, using defaults")

# Save log configuration to file
async def save_log_config():
    config = {
        'log_channel_id': log_channel_id,
        'use_log_channel': use_log_channel,
        'detailed_logging': detailed_logging
    }
    
    try:
        with open(log_config_file, 'w') as f:
            json.dump(config, f)
        await log_message("Saved log configuration", send_to_discord=False)
        return True
    except Exception as e:
        print(f"Error saving log configuration: {e}")
        return False

async def process_log_queue():
    """Process the log message queue and send bundled messages to Discord."""
    global log_task_running
    try:
        log_task_running = True
        while True:
            # Wait for 5 seconds to collect messages
            await asyncio.sleep(5)
            
            # Get messages from queue
            async with log_queue_lock:
                if not log_message_queue:
                    continue
                
                messages_to_send = log_message_queue.copy()
                log_message_queue.clear()
            
            # Group similar messages
            grouped_messages = []
            current_group = []
            current_prefix = None
            
            for msg in messages_to_send:
                # Try to identify message prefixes for grouping (like "Scheduling next freewill")
                if "[INFO] Scheduling next freewill" in msg:
                    prefix = "freewill_schedule"
                elif "[INFO] Running freewill" in msg:
                    prefix = "freewill_run"
                elif "[INFO] Next freewill" in msg:
                    prefix = "freewill_next"
                else:
                    prefix = None
                
                if prefix != current_prefix and current_group:
                    # Start a new group
                    grouped_messages.append("\n".join(current_group))
                    current_group = [msg]
                    current_prefix = prefix
                else:
                    current_group.append(msg)
                    current_prefix = prefix
            
            if current_group:
                grouped_messages.append("\n".join(current_group))
            
            # Send the grouped messages
            if grouped_messages:
                try:
                    # Split into chunks to avoid message length limits
                    chunks = []
                    current_chunk = ""
                    
                    for group in grouped_messages:
                        if len(current_chunk) + len(group) + 1 > 1900:  # Discord limit is 2000
                            chunks.append(current_chunk)
                            current_chunk = group
                        else:
                            if current_chunk:
                                current_chunk += "\n\n" + group
                            else:
                                current_chunk = group
                    
                    if current_chunk:
                        chunks.append(current_chunk)
                    
                    for chunk in chunks:
                        if use_log_channel and log_channel_id:
                            channel = client.get_channel(int(log_channel_id))
                            if channel:
                                # Use a CodeBlock for better formatting
                                await channel.send(f"```log\n{chunk}\n```")
                            
                except Exception as e:
                    print(f"Error sending bundled log messages: {e}")
    except Exception as e:
        print(f"Error in log queue processor: {e}")
    finally:
        log_task_running = False

async def log_message(message, level="INFO", send_to_discord=True):
    """Log a message to console and optionally to Discord."""
    global log_task_running
    global memory_manager
    
    # Format the message with timestamp and level
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    formatted_message = f"[{current_time}] [{level}] {message}"
    
    # Always print to console
    print(formatted_message)
    
    # Add to queue for Discord if enabled
    if send_to_discord and use_log_channel and log_channel_id:
        async with log_queue_lock:
            log_message_queue.append(formatted_message)
        
        # Start the queue processor task if not running
        if not log_task_running:
            asyncio.create_task(process_log_queue())

# Load attachment transcriptions if file exists
if os.path.exists(attachment_transcriptions_file):
    try:
        with open(attachment_transcriptions_file, 'r') as f:
            attachment_transcriptions = json.load(f)
    except json.JSONDecodeError:
        attachment_transcriptions = {}

async def save_attachment_transcription(url, attachment_type, transcription):
    """
    Saves an attachment transcription to the json file for future reference.
    
    Args:
        url: The attachment URL as a string
        attachment_type: Either 'image' or 'video'
        transcription: The text transcription/description of the attachment
    """
    global attachment_transcriptions
    global memory_manager
    
    # Add to in-memory storage
    attachment_transcriptions[url] = {
        'type': attachment_type,
        'transcription': transcription,
        'timestamp': format_est_time()
    }
    
    # Save to file
    try:
        with open(attachment_transcriptions_file, 'w') as f:
            json.dump(attachment_transcriptions, f)
        await log_message(f"Saved {attachment_type} transcription for {url}")
        return True
    except Exception as e:
        await log_message(f"Error saving attachment transcription: {e}", level="ERROR")
        return False

# Track freewill-enabled channels and their state
freewill_channels = {}  # Format: {channel_id: {'enabled': bool, 'last_message_time': timestamp, 'freewill_state': int, 'task': asyncio.Task}}

intents = discord.Intents.default()
intents.messages = True
intents.guilds = True
intents.typing = False  # We'll manage typing ourselves
intents.message_content = True


client = discord.Client(intents=intents)
memory_manager = MemoryManager()

# We will keep track of how many messages have been sent in a channel to check 
# if we need to show setup messages.
recent_conversation_counter = {}


@client.event
async def on_ready():
    # Load log configuration
    load_log_config()
    
    await log_message(f"We have logged in as {client.user}")
    
    # Start the cleanup task
    asyncio.create_task(cleanup_freewill_tasks())
    
    # Start the log queue processor
    if not log_task_running:
        asyncio.create_task(process_log_queue())

async def fetch_and_encode_image(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            if resp.status == 200:
                data = await resp.read()
                return base64.b64encode(data).decode('utf-8')
            else:
                return None
    
    
import aiohttp
import asyncio


async def get_image_description(image_url):
    """
    Uses the light model to generate a brief description (<30 words) of an image using Gemini.
    This version uses the official Python SDK.
    """
    # Check if we already have a transcription for this image
    if image_url in attachment_transcriptions:
        await log_message(f"Using cached image transcription for {image_url}")
        return attachment_transcriptions[image_url]['transcription']
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(image_url) as resp:
                if resp.status == 200:
                    data = await resp.read()
                    # Open the image with PIL from the downloaded bytes
                    image = Image.open(io.BytesIO(data))
                else:
                    await log_message(f"Failed to fetch image: Status {resp.status}")
                    return "Image (no description available)"
    except Exception as e:
        await log_message("Error fetching image: " + str(e))
        return "Image (no description available)"

    # Create a Gemini client using your API key
    client = genai.Client(api_key=GEMINI_API_KEY)
    # Convert image to base64
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="PNG")
    image_base64 = base64.b64encode(image_bytes.getvalue()).decode("utf-8")

    # Create system instruction for image description
    image_system_instruction = "You are an assistant that describes images briefly and accurately. Be descriptive about this image in less than 50 words that capture the main elements and context of the image."

    config = GenerateContentConfig(
        system_instruction=[image_system_instruction],
        temperature=0.6
    )

    response = client.models.generate_content(
        model=GEMINI_LIGHT_MODEL,
        contents=[
            {"role": "user", "parts": [
                {"text": "Describe this image briefly."},
                {"inlineData": {"mimeType": "image/png", "data": image_base64}}
            ]}
        ],
        config=config
    )

    description = response.text.strip()
    await log_message(f"Got image description: {description}")
    # Save the image description
    await save_attachment_transcription(image_url, 'image', description)
    return description


async def get_video_description(video_url):
    """
    Uses the light model to generate a brief description (<50 words) of a video using Gemini.
    """
    # Check if we already have a transcription for this video
    if video_url in attachment_transcriptions:
        await log_message(f"Using cached video transcription for {video_url}")
        return attachment_transcriptions[video_url]['transcription']
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(video_url) as resp:
                if resp.status == 200:
                    data = await resp.read()
                    # Create a temporary file to hold the video
                    temp_video_path = f"temp_video_{int(time.time())}.mp4"
                    with open(temp_video_path, "wb") as temp_file:
                        temp_file.write(data)
                else:
                    await log_message(f"Failed to fetch video: Status {resp.status}")
                    return "Video (no description available)"
    except Exception as e:
        await log_message("Error fetching video: " + str(e))
        return "Video (no description available)"

    try:
        # Create a Gemini client using your API key
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        # Read the video file and encode as base64
        with open(temp_video_path, "rb") as video_file:
            video_bytes = video_file.read()
        video_base64 = base64.b64encode(video_bytes).decode("utf-8")
        
        # Create system instruction for video description
        video_system_instruction = """
        You are an assistant that describes videos concisely and accurately. 
        Focus on providing valuable information that would be relevant in a chat context.
        Keep descriptions under 80 words, emphasizing the most notable elements.
        Be specific about what's happening in the video without unnecessary elaboration.
        """
        
        config = GenerateContentConfig(
            system_instruction=[video_system_instruction],
            temperature=0.6
        )
        
        # Call Gemini with the video data
        response = client.models.generate_content(
            model=GEMINI_LIGHT_MODEL,
            contents=[
                {"role": "user", "parts": [
                    {"text": "Describe this video briefly."},
                    {"inlineData": {"mimeType": "video/mp4", "data": video_base64}}
                ]}
            ],
            config=config
        )
        
        # Clean up the temporary file
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        
        description = response.text.strip()
        await log_message(f"Got video description: {description}")
        
        # Save the video description
        await save_attachment_transcription(video_url, 'video', description)
        
        return description
    except Exception as e:
        await log_message(f"Error processing video: {e}", level="ERROR")
        # Clean up the temporary file in case of error
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        return "Video (could not be processed)"


async def freewill_evaluation(channel_id, channel):
    """
    Evaluate whether the bot should initiate a conversation using the light model.
    Returns !messagenow or !messagelater decision.
    """
    # Get the current time and the time of the last message
    current_time = get_est_time()
    last_message_time = freewill_channels[channel_id].get('last_message_time', current_time - timedelta(minutes=15))
    time_diff = current_time - last_message_time
    
    # Create system instruction for freewill decision
    freewill_system_instruction = f"""
    You are "Saki", a 19-year-old college student who plays games and is online on Discord.
    
    You need to decide whether to message user "Om" now or later using !messagenow or !messagelater
    
    Consider:
    - The current time is in Eastern Standard Time (EST, UTC-5)
    - Whether you're likely to be sleeping based on your conversation history and time (sometimes you're online at night)
    - Your current activities based on time and conversation context
    - If there's any unfinished business or planned activities discussed earlier
    - If you (Saki) planned on waking up or talking to Om soon or whether she would want to talk to him soon
    
    Examples of when to message now:
    - If you mentioned you'd be free after a certain time and it's now that time
    - If you discussed doing something together (like playing a game) at around this time
    - If the conversation ended abruptly earlier and you're normally online at this time
    
    Examples of when to message later:
    - If it's a time/day you'd be asleep and you'd likely be sleeping
    - If you've already sent multiple messages without response
    - If nothing urgent or planned was discussed in the previous conversation
    
    CRITICAL: You MUST respond with ONLY one of these two options:
    !messagenow - If you decide to send a message now
    !messagelater - If you decide to wait and evaluate again later
    
    No other text should be included in your response.
    """
    
    # If this is a repeat freewill evaluation (state > 0), add additional context
    if freewill_channels[channel_id].get('freewill_state', 0) > 0:
        freewill_system_instruction += f"""
        IMPORTANT: You have already sent at least one message that has not received a response.
        Only choose !messagenow if there's a compelling reason to follow up (like reminding about something time-sensitive).
        In most cases, you should wait longer by choosing !messagelater unless you have a specific goal in mind.
        """
    
    # Create the message content (just the facts)
    message_content = f"""
    The current time is {format_est_time(current_time)} Eastern Time (EST, UTC-5).
    The last message in this conversation was sent at {format_est_time(last_message_time)} ({int(time_diff.total_seconds() / 60)} minutes ago).
    Freewill state: {freewill_channels[channel_id].get('freewill_state', 0)}
    (If the freewill state is > 0, it means you've already sent at least one message without a response.)
    """
    
    # Call the light model to make the decision
    client = genai.Client(api_key=GEMINI_API_KEY)
    config = GenerateContentConfig(
        system_instruction=[freewill_system_instruction],
        temperature=0.8
    )
    response = client.models.generate_content(
        model=GEMINI_LIGHT_MODEL,
        contents=[{"role": "user", "parts": [{"text": message_content}]}],
        config=config
    )
    
    decision = response.text.strip().lower()
    await log_message(f"Freewill evaluation for channel {channel_id}: {decision}")
    
    # Ensure we get a valid response
    if decision not in ["!messagenow", "!messagelater"]:
        await log_message(f"Invalid freewill decision: {decision}, defaulting to !messagelater")
        decision = "!messagelater"
    
    return decision


async def handle_freewill_timer(channel_id, channel):
    """
    Handles the freewill timer for a channel, evaluating whether to send a message
    and scheduling the next evaluation as needed.
    """
    global memory_manager
    try:
        # Check if freewill is still enabled for this channel
        if channel_id not in freewill_channels or not freewill_channels[channel_id]['enabled']:
            await log_message(f"Freewill no longer enabled for channel {channel_id}")
            return
        
        await log_message(f"Running freewill evaluation for channel {channel_id}")
        
        # Get the decision from the light model
        decision = await freewill_evaluation(channel_id, channel)
        
        if decision == "!messagenow":
            # Increment the freewill state
            freewill_channels[channel_id]['freewill_state'] = freewill_channels[channel_id].get('freewill_state', 0) + 1
            await log_message(f"Channel {channel_id} freewill state incremented to {freewill_channels[channel_id]['freewill_state']}")
            
            # Generate and send a message
            bot_reply = await generate_freewill_message(channel_id)
            if bot_reply and bot_reply.lower() != "!noresponse":
                # Add the message to memory
                memory_manager.add_message(channel_id, "assistant", bot_reply, format_est_time())
                
                # Send the message
                sent_message = await send_with_typing_delay(channel, bot_reply)
                
                # Update the last message time
                freewill_channels[channel_id]['last_message_time'] = get_est_time()
                await log_message(f"Sent freewill message to channel {channel_id}")
                
                # Store as the last bot message for potential reload
                if channel_id not in last_bot_message:
                    last_bot_message[channel_id] = {}
                last_bot_message[channel_id] = {
                    "content": bot_reply,
                    "message_object": sent_message
                }
        
        # Schedule the next evaluation - always use fixed 20 minutes
        wait_time = 20  # Fixed 20 minute timer
        
        await log_message(f"Next freewill evaluation for channel {channel_id} in {wait_time} minutes (at {format_est_time(get_est_time() + timedelta(minutes=wait_time))})")
        
        # Schedule the next timer
        freewill_channels[channel_id]['task'] = asyncio.create_task(
            schedule_next_freewill(channel_id, channel, wait_time)
        )
    
    except Exception as e:
        await log_message(f"Error in freewill timer for channel {channel_id}: {e}", level="ERROR")
        # If there was an error, try to reschedule after a delay to avoid rapid error loops
        await asyncio.sleep(60)  # Wait 1 minute
        if channel_id in freewill_channels and freewill_channels[channel_id]['enabled']:
            await log_message(f"Rescheduling freewill timer for channel {channel_id} after error")
            freewill_channels[channel_id]['task'] = asyncio.create_task(
                schedule_next_freewill(channel_id, channel, 20)
            )


async def schedule_next_freewill(channel_id, channel, minutes):
    """
    Schedules the next freewill evaluation after the specified delay
    """
    await log_message(f"Scheduling next freewill evaluation for channel {channel_id} in {minutes} minutes (at {format_est_time(get_est_time() + timedelta(minutes=minutes))})")
    
    # Wait for the specified time
    await asyncio.sleep(minutes * 60)
    
    # Run the handler
    await handle_freewill_timer(channel_id, channel)


async def generate_freewill_message(channel_id):
    """
    Generates a message using the heavy model for freewill initiation.
    """
    global memory_manager
    # Prepare the conversation context from memory
    conversation_history = memory_manager.histories.get(channel_id, [])
    current_time = format_est_time()

    # Build structured contents
    contents = []
    if len(conversation_history) >= 0:
        for setup_msg in SETUP_MESSAGES:
            contents.append({"role": "assistant", "parts": [{"text": setup_msg}]})
    for msg in conversation_history:
        role = "user" if msg["role"] == "user" else "assistant"
        contents.append({"role": role, "parts": [{"text": f"[{msg['time']}] {msg['content']}"}]})
    # Prompt the assistant to start the conversation
    contents.append({"role": "user", "parts": [{"text": f"The current time is {current_time} Eastern Time (EST, UTC-5). You have opened Discord and decided to text."}]})

    # Create a specialized system instruction for freewill messaging
    freewill_system_instruction = SYSTEM_INSTRUCTION + "\n\n" + """
    You have opened Discord and decided to text Om first.
    Consider the following when crafting your message:
    1. What did you recently talk about? What time is it?
    2. Reference shared interests, previous conversations, or activities you might be doing
    3. Consider gm/gn if applicable
    4. It's okay to ask questions or share something that happened to you!
    5. Be authentic to your character - a 19-year-old college student female named Saki who plays games.

    Your message should feel like a natural continuation of your relationship with Om.
    """

    # Create a Gemini client and send the prompt
    client = genai.Client(api_key=GEMINI_API_KEY)
    config = GenerateContentConfig(
        system_instruction=[freewill_system_instruction],
        temperature=1.0,
        max_output_tokens=MAX_TOKENS_CURRENT
    )
    response = client.models.generate_content(
        model=GEMINI_HEAVY_MODEL,
        contents=contents,
        config=config
    )
    bot_content = response.text.strip()
    
    # Clean up the response
    bot_content = re.sub(r'^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\]\s*', '', bot_content)
    bot_content = re.sub(r'^sakixoxo:\s*', '', bot_content)
    bot_content = re.sub(r'^ASSISTANT:\s*sakixoxo:\s*', '', bot_content)
    bot_content = re.sub(r'^"(.*)"$', r'\1', bot_content)
    
    return bot_content


async def handle_freewill_command(message):
    """
    Handles the /freewill command to toggle freewill mode for a channel.
    """
    channel_id = str(message.channel.id)
    content_lower = message.content.strip().lower()
    
    # Check if this is a status request
    if content_lower == "/freewill status":
        # Get status of all freewill channels
        status_msg = "**Freewill Status**\n"
        for ch_id, data in freewill_channels.items():
            if data.get('enabled', False):
                channel = client.get_channel(int(ch_id))
                channel_name = channel.name if channel else "Unknown Channel"
                last_time = data.get('last_message_time', datetime.now())
                time_diff = datetime.now() - last_time
                minutes_ago = int(time_diff.total_seconds() / 60)
                next_eval = "N/A"
                if 'task' in data and data['task'] and not data['task'].done() and not data['task'].cancelled():
                    next_eval = f"~{20 - (minutes_ago % 20)} minutes"
                
                status_msg += f"- {channel_name} (ID: {ch_id}): State {data.get('freewill_state', 0)}, Last activity: {minutes_ago} mins ago, Next eval: {next_eval}\n"
        
        if status_msg == "**Freewill Status**\n":
            status_msg += "No active freewill channels."
        
        # Send the status message
        await message.channel.send(status_msg)
        return
    
    # Delete the command message
    try:
        await message.delete()
    except Exception as e:
        await log_message(f"Could not delete /freewill command: {e}", level="ERROR")
    
    # Toggle freewill mode
    if channel_id in freewill_channels and freewill_channels[channel_id]['enabled']:
        # Disable freewill
        freewill_channels[channel_id]['enabled'] = False
        
        # Cancel any pending tasks
        if 'task' in freewill_channels[channel_id] and freewill_channels[channel_id]['task']:
            freewill_channels[channel_id]['task'].cancel()
        
        # Send confirmation
        confirm_msg = await message.channel.send("Free will deactivated.")
    else:
        # Enable freewill
        if channel_id not in freewill_channels:
            freewill_channels[channel_id] = {}
        
        freewill_channels[channel_id]['enabled'] = True
        freewill_channels[channel_id]['last_message_time'] = get_est_time()
        freewill_channels[channel_id]['freewill_state'] = 0
        
        # Start the freewill timer with fixed 20 minute interval
        freewill_channels[channel_id]['task'] = asyncio.create_task(
            schedule_next_freewill(channel_id, message.channel, 20)  # Start with 20 minute timer
        )
        
        # Send confirmation
        confirm_msg = await message.channel.send("Free will activated.")
    
    # Delete the confirmation after a short delay
    await asyncio.sleep(3)
    try:
        await confirm_msg.delete()
    except Exception as e:
        await log_message(f"Could not delete confirmation message: {e}", level="ERROR")


@client.event
async def on_message(message: discord.Message):
    global memory_manager
    # Immediately ignore messages from the bot itself
    if message.author == client.user:
        return

    # Define channel_id once, since it's used by command handlers below
    channel_id = str(message.channel.id)
    content_lower = message.content.strip().lower()

    # Handle freewill command
    if content_lower.startswith("/freewill"):
        await handle_freewill_command(message)
        return
    
    # Handle log command
    if content_lower.startswith("/log"):
        await handle_log_command(message)
        return
    
    # Handle contextregen command
    if content_lower.startswith("/contextregen"):
        await handle_contextregen_command(message)
        return
    
    # Handle test search command (admin only)
    if content_lower.startswith("/testsearch "):
        # Extract search query
        search_query = message.content[12:].strip()
        await test_search_feature(message.channel, channel_id, search_query)
        return
    
    # Process /context commands from users
    if content_lower.startswith("/contextsave"):
        # Extract optional limit parameter
        parts = message.content.split(maxsplit=1)
        try:
            limit = int(parts[1]) if len(parts) > 1 else 500  # Default to 500 if no limit provided
            limit = min(limit, 500)  # Ensure it doesn't exceed 500
        except ValueError:
            limit = 500  # Default to 500 if user provides invalid input

        await message.channel.send(f"Fetching up to {limit} messages. This may take some time...")

        # **Split into 5 chunks of 100 messages with a 1-second delay between each**
        total_messages_fetched = 0
        for i in range(5):  # Loop 5 times to fetch 500 messages
            batch_limit = min(100, limit - total_messages_fetched)  # Ensure we don't exceed the total limit
            if batch_limit <= 0:
                break  # Stop if we've already fetched the needed messages

            context_name, msg_count = await save_channel_history(message.channel, channel_id, batch_limit)
            total_messages_fetched += msg_count

            if total_messages_fetched >= limit:
                break  # Stop early if we already have enough messages

            await asyncio.sleep(1)  # **Pause for 1 second between requests to respect rate limits**

        if total_messages_fetched > 0:
            await message.channel.send(f"Channel history saved! ID: `{context_name}` ({total_messages_fetched} messages)")
        else:
            await message.channel.send("Failed to save channel history.")
        return
    elif content_lower.startswith("/contextimport"):
        parts = message.content.split(maxsplit=1)
        context_id = parts[1] if len(parts) > 1 else None
        await import_context(message.channel, channel_id, context_id)
        return
    elif content_lower.startswith("/contextclear"):
        parts = message.content.split(maxsplit=1)
        context_id = parts[1] if len(parts) > 1 else None
        await clear_context(message.channel, context_id)
        return
    elif content_lower.startswith("/contextlist"):
        await list_contexts(message.channel)
        return

    # Process !reload command
    if content_lower == "!reload":
        await handle_reload_command(message)
        return

    # Debug command to test video processing with a URL
    if content_lower.startswith("!debugvideo "):
        try:
            video_url = message.content.split(" ", 1)[1].strip()
            if video_url:
                await message.channel.send(f"Testing video processing for: {video_url}")
                description = await get_video_description(video_url)
                await message.channel.send(f"Video description: {description}")
            return
        except Exception as e:
            await message.channel.send(f"Error testing video: {str(e)}")
            return

    # Check for attachments (images and videos)
    attachment_descriptions = []
    
    if message.attachments:
        for attachment in message.attachments:
            # Check for image attachments
            if any(attachment.filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp']):
                await log_message(f"Found image attachment: {attachment.url}")
                # Get image description using the light model
                description = await get_image_description(attachment.url)
                if description:
                    attachment_descriptions.append({"type": "image", "description": description})
            
            # Check for video attachments
            elif any(attachment.filename.lower().endswith(ext) for ext in ['.mp4', '.mov', '.avi', '.wmv', '.mkv', '.webm']):
                await log_message(f"Found video attachment: {attachment.url}")
                # Get video description using the light model
                description = await get_video_description(attachment.url)
                if description:
                    attachment_descriptions.append({"type": "video", "description": description})

    # Combine the message content with attachment descriptions
    full_content = message.content
    for att in attachment_descriptions:
        if full_content:
            full_content += f"\n[{att['type'].capitalize()} sent by user. Description: {att['description']}]"
        else:
            full_content = f"[{att['type'].capitalize()} sent by user. Description: {att['description']}]"
    
    if attachment_descriptions:
        await log_message(f"Combined content with {len(attachment_descriptions)} attachments: {full_content}")

    # Add the user's message to memory
    timestamp_str = format_est_time(message.created_at)
    memory_manager.add_message(channel_id, "user", full_content, timestamp_str)

    # Store user's message for potential reload with edits
    if channel_id not in user_messages:
        user_messages[channel_id] = []
    
    # Keep only the last 10 messages per channel to avoid memory issues
    if len(user_messages[channel_id]) >= 10:
        user_messages[channel_id].pop(0)
    
    user_messages[channel_id].append({
        'message_object': message,
        'content': full_content,
        'time': timestamp_str
    })

    # If you want to track a conversation start, you can do that here,
    # but do not send the setup messages to the channel.
    if channel_id not in recent_conversation_counter:
        recent_conversation_counter[channel_id] = 0
    recent_conversation_counter[channel_id] += 1

    # Check memory summarization triggers
    memory_manager.maybe_summarize(channel_id)

    # Reset freewill state when user responds
    if channel_id in freewill_channels and freewill_channels[channel_id]['enabled']:
        # Cancel any pending freewill task
        if 'task' in freewill_channels[channel_id] and freewill_channels[channel_id]['task']:
            await log_message(f"Cancelling scheduled freewill evaluation for channel {channel_id} as user has responded")
            freewill_channels[channel_id]['task'].cancel()
        
        # Reset freewill state
        old_state = freewill_channels[channel_id].get('freewill_state', 0)
        freewill_channels[channel_id]['freewill_state'] = 0
        freewill_channels[channel_id]['last_message_time'] = get_est_time()
        await log_message(f"Reset freewill state for channel {channel_id} from {old_state} to 0")
        
        # Schedule a new freewill evaluation
        next_eval_time = get_est_time() + timedelta(minutes=20)
        await log_message(f"Scheduling next freewill evaluation for channel {channel_id} at {format_est_time(next_eval_time)}")
        freewill_channels[channel_id]['task'] = asyncio.create_task(
            schedule_next_freewill(channel_id, message.channel, 20)  # Reset to 20 minute timer
        )

    # Generate response using the "heavy" model
    bot_reply = await generate_bot_reply(channel_id)
    if bot_reply:
        # Check if the bot decides to not reply
        if bot_reply.strip().lower() == "!noresponse":
            # Log to the designated channel
            log_channel_id = 1345381968950591539  # Log channel ID
            log_channel = client.get_channel(log_channel_id)

            if log_channel:
                await log_channel.send(f"**!noresponse was used in {message.channel.mention}**")

            # Simulate typing but do not send a response
            async with message.channel.typing():
                await asyncio.sleep(2)
            return  # Exit without sending a message


        # Add the bot's reply to the memory and send it
        memory_manager.add_message(channel_id, "assistant", bot_reply, format_est_time())
        
        # Update last message time for freewill
        if channel_id in freewill_channels:
            freewill_channels[channel_id]['last_message_time'] = get_est_time()
        
        # Store the last bot message for potential reload
        if channel_id not in last_bot_message:
            last_bot_message[channel_id] = {}
        last_bot_message[channel_id] = {
            "content": bot_reply,
            "message_object": await send_with_typing_delay(message.channel, bot_reply)
        }


async def generate_bot_reply(channel_id):
    """
    Calls the heavy model using Gemini with the conversation context.
    This version uses the official SDK.
    """
    global memory_manager
    # Prepare the conversation context from memory
    conversation_history = memory_manager.histories.get(channel_id, [])
    current_time = format_est_time()

    # Build structured contents for the Gemini API
    contents = []

    # If this is not a new conversation, include SETUP_MESSAGES as per your condition
    if len(conversation_history) >= 0:
        from config import SETUP_MESSAGES
        for setup_msg in SETUP_MESSAGES:
            contents.append({"role": "assistant", "parts": [{"text": setup_msg}]})

    # Append the conversation history
    for msg in conversation_history:
        role = "user" if msg["role"] == "user" else "assistant"
        contents.append({"role": role, "parts": [{"text": f"[{msg['time']}] {msg['content']}"}]})

    # Add the current time as a user message to maintain context
    contents.append({"role": "user", "parts": [{"text": f"The current time is {current_time} Eastern Time (EST, UTC-5)."}]})

    # Create a Gemini client and send the prompt
    client = genai.Client(api_key=GEMINI_API_KEY)

    # First attempt to generate a response
    try:
        await log_message("[MODEL] Generating initial response")
        config = GenerateContentConfig(
            system_instruction=[SYSTEM_INSTRUCTION],
            temperature=1.05,
            max_output_tokens=MAX_TOKENS_CURRENT
        )
        response = client.models.generate_content(
            model=GEMINI_HEAVY_MODEL,
            contents=contents,
            config=config
        )
        bot_content = response.text.strip()

        # Check for various search command patterns
        search_patterns = [
            r'(?:^|\n)\/search\s+([^\n]+)',  # Original direct command pattern
            r'\/search\s+([^\n]+)',          # More lenient - search anywhere in text
            r'\/search\s*(.*)',              # Even more lenient - any text after /search
        ]

        search_query = None
        for pattern in search_patterns:
            match = re.search(pattern, bot_content)
            if match:
                search_query = match.group(1).strip()
                await log_message(f"[MODEL] Search command detected: '{search_query}' (pattern: {pattern})")
                break

        # If search command detected, perform search and regenerate response
        if search_query:
            await log_message(f"[MODEL] Performing search for: {search_query}")
            search_results = await perform_search(channel_id, search_query)

            # Add search results to contents and regenerate response
            search_contents = contents.copy()  # Copy original contents
            search_contents.append({
                "role": "user",
                "parts": [{
                    "text": (
                        f"This is Discord. You used /search, and here are the results for: {search_query}\n\n"
                        f"{search_results}\n\n"
                        f"Respond naturally as Saki based on these search results. IMPORTANT RULES:\n"
                        f"1. Do NOT say phrases like 'based on what I found' or 'according to my search'\n"
                        f"2. Do NOT include citations, URLs, or reference numbers in your response\n"
                        f"3. Speak in your normal casual tone - you're a 19-year-old college student\n"
                    )
                }]
            })

            # Create an enhanced system instruction for search responses
            search_response_system = SYSTEM_INSTRUCTION + "\n\n" + """
            You are responding based on search results for information you requested.
            Important guidelines:
            1. DO NOT mention that you searched for this information or say phrases like 'based on my search'
            2. DO NOT include citations, URLs, or reference numbers in your response
            3. Speak naturally in your 19-year-old college student voice, keeping the same personality
            4. Incorporate the information smoothly as if it's part of your knowledge
            5. If the search results aren't helpful, feel free to pivot the conversation naturally
            """
            
            try:
                await log_message(f"[MODEL] Regenerating response with search results")
                search_response = client.models.generate_content(
                    model=GEMINI_HEAVY_MODEL,
                    contents=search_contents,
                    config=GenerateContentConfig(
                        system_instruction=[search_response_system],
                        temperature=0.8,
                        max_output_tokens=MAX_TOKENS_CURRENT
                    )
                )
                bot_content = search_response.text.strip()
                await log_message(f"[MODEL] Successfully regenerated response with search results")
            except Exception as search_regen_error:
                await log_message(f"[MODEL] Error regenerating response with search: {search_regen_error}")
                # Keep the original response but remove the search command
        # No else clause needed here - if no search was performed, just use the original response

    except Exception as initial_gen_error:
        await log_message(f"[MODEL] Error in initial response generation: {initial_gen_error}")
        # Fallback to a simple response
        bot_content = "Sorry, I'm having trouble responding right now. Can we talk about something else?"

    # Clean up the response regardless of which path was taken
    bot_content = re.sub(r'^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\]\s*', '', bot_content)
    bot_content = re.sub(r'^(?:ASSISTANT|SAKIXOXO|SAKI|USER)(?::|：)\s*', '', bot_content, flags=re.IGNORECASE)
    bot_content = re.sub(r'^"(.*)"$', r'\1', bot_content)
    bot_content = re.sub(r"^'(.*)'$", r'\1', bot_content, flags=re.DOTALL)

    return bot_content


def clean_search_results(results):
    """
    Clean and format search results to be more helpful in the conversation.
    
    Args:
        results: The raw search results from Gemini
        
    Returns:
        Cleaned and formatted search results
    """
    # Remove excessive URLs or citations if present
    cleaned = re.sub(r'\[(\d+)\]\s*https?://\S+', r'[Source \1]', results)
    
    # Make sure citations are in a consistent format
    cleaned = re.sub(r'\[(Source|Citation|Reference):\s*https?://\S+\]', r'[Source]', cleaned)
    
    # Handle Gemini 2.0 specific citation formats
    cleaned = re.sub(r'\[\d+\]: https?://\S+', r'[Source]', cleaned)
    cleaned = re.sub(r'\(\s*https?://\S+\s*\)', r'(Source)', cleaned)
    
    # Remove inline citation links that Gemini 2.0 might add
    cleaned = re.sub(r'\[\d+\]\(https?://[^\)]+\)', r'', cleaned)
    
    # Improved handling of search result formatting
    # Replace lengthy URLs in text with [Link] placeholder
    cleaned = re.sub(r'https?://[^\s\]]+', r'[Link]', cleaned)
    
    # Remove any "searching for..." meta-commentary
    cleaned = re.sub(r'(?i)^(searching for|looking up|finding information about|I will search for|according to my search|based on my search).*$', '', cleaned, flags=re.MULTILINE)
    
    # Remove any headers that might be added
    cleaned = re.sub(r'(?i)^(search results|results for|information about|search results for|here\'s what I found about):?\s*.*$', '', cleaned, flags=re.MULTILINE)
    
    # Remove redundant "I found that" or "The search shows" prefixes
    cleaned = re.sub(r'(?i)^(I found that|The search shows|According to the search|From the search results,|Based on the search results,)', '', cleaned, flags=re.MULTILINE)
    
    # Remove any Google-specific metadata annotations that might appear
    cleaned = re.sub(r'(?i)\(via Google\)', '', cleaned)
    cleaned = re.sub(r'(?i)Source: Google', '', cleaned)
    
    # Remove Gemini's tendency to use numbered points for every search result
    # but preserve actual numerical data
    if cleaned.count('\n1.') > 1 and cleaned.count('\n2.') > 0:
        # Likely a numbered list
        pass  # Keep the formatting
    else:
        # Remove isolated number formatting that's not part of actual data
        cleaned = re.sub(r'(?m)^\d+\.\s+', '• ', cleaned)
    
    # Fix potential multiple blank lines
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    
    # Trim extra whitespace
    result = cleaned.strip()
    
    # Ensure the result doesn't end with a semicolon or comma
    result = re.sub(r'[;,]$', '.', result)
    
    return result


async def perform_search(channel_id, search_query):
    """
    Performs a web search using Gemini's Google Search tool.
    
    Args:
        channel_id: The channel ID for context
        search_query: The search query to execute
        
    Returns:
        String containing search results
    """
    global memory_manager
    # Clean up search query - remove any quotes and unnecessary text
    search_query = re.sub(r'^["\']|["\']$', '', search_query).strip()
    search_query = re.sub(r'(?i)^(please |can you |search for |look up |find |tell me about |information on )', '', search_query).strip()
    
    await log_message(f"[SEARCH] Performing search for query: {search_query}")
    
    try:
        # Get recent conversation context for search
        conversation_history = memory_manager.histories.get(channel_id, [])
        recent_messages = conversation_history[-SEARCH_CONTEXT_MESSAGE_COUNT:] if len(conversation_history) > SEARCH_CONTEXT_MESSAGE_COUNT else conversation_history
        
        await log_message(f"[SEARCH] Including {len(recent_messages)} recent messages as context")
        
        # Build message content for the search (without system instructions)
        context_lines = []
        context_lines.append(f"The current time is {format_est_time()} Eastern Time (EST, UTC-5).")
        context_lines.append("Recent conversation context:")
        
        for msg in recent_messages:
            if msg["role"] == "user":
                context_lines.append(f"USER: {msg['content']}")
            else:
                context_lines.append(f"SAKI: {msg['content']}")
                
        context_lines.append(f"\nSaki wants to search for: {search_query}")
        context_lines.append("Use the Google Search tool to find this information.")
        
        search_context = "\n".join(context_lines)
        
        # Log search context length for debugging
        await log_message(f"[SEARCH] Search context length: {len(search_context)} characters")
        
        # Initialize Gemini client
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        # Import necessary types for Gemini 2.0 search
        from google.genai.types import Tool, GenerateContentConfig, GoogleSearch
        
        # Create the Google Search tool
        google_search_tool = Tool(
            google_search=GoogleSearch()
        )
        
        await log_message("[SEARCH] Sending search request to Gemini with Google Search tool")
        
        start_time = time.time()
        
        # Create proper GenerateContentConfig with tools
        config = GenerateContentConfig(
            system_instruction=[SEARCH_SYSTEM_INSTRUCTION],
            tools=[google_search_tool],
            response_modalities=["TEXT"],
        )
        
        # Make the API call with the tool
        response = client.models.generate_content(
            model=GEMINI_SEARCH_MODEL,
            contents=search_context,
            config=config
        )
        
        elapsed_time = time.time() - start_time
        await log_message(f"[SEARCH] Search completed in {elapsed_time:.2f} seconds")
        
        # Extract and format search results
        search_results = response.text.strip()
        
        # Enhanced logging for search metadata if available
        if hasattr(response, 'candidates') and response.candidates:
            if hasattr(response.candidates[0], 'grounding_metadata') and response.candidates[0].grounding_metadata:
                if hasattr(response.candidates[0].grounding_metadata, 'search_entry_point'):
                    await log_message("[SEARCH] Search metadata available")
                    # Log additional metadata if needed
        
        # Log the search results to console
        await log_message(f"[SEARCH] Search results received: {len(search_results)} characters")
        
        # Clean up the search results
        cleaned_results = clean_search_results(search_results)
        
        return cleaned_results
            
    except Exception as e:
        await log_message(f"[SEARCH] Error in search function: {e}")
        return f"Sorry, I couldn't find information about '{search_query}' there was an error in Discord's search."


async def test_search_feature(channel, channel_id, search_query):
    """
    Tests the search feature by directly performing a search and showing the results
    in the Discord channel.
    
    Args:
        channel: The Discord channel object
        channel_id: The channel ID
        search_query: The search query to test
    """
    try:
        # Send a message indicating the search is in progress
        await channel.send(f"Testing search for: `{search_query}`")
        
        # Perform the search
        search_results = await perform_search(channel_id, search_query)
        
        # Split long results if necessary
        MAX_DISCORD_MESSAGE_LENGTH = 1900  # Leave some room for formatting
        
        if len(search_results) > MAX_DISCORD_MESSAGE_LENGTH:
            # Split into chunks
            chunks = [search_results[i:i+MAX_DISCORD_MESSAGE_LENGTH] 
                     for i in range(0, len(search_results), MAX_DISCORD_MESSAGE_LENGTH)]
            
            for i, chunk in enumerate(chunks):
                await channel.send(f"**Search Results (Part {i+1}/{len(chunks)}):**\n```\n{chunk}\n```")
        else:
            await channel.send(f"**Search Results:**\n```\n{search_results}\n```")
        
        # Test how the heavy model would use these results
        await channel.send("Generating a response based on these search results...")
        
        # Create message content for the test
        message_content = f"This is Discord. You used /search, this is the result of your search query: {search_query}\n\n{search_results}\n\nRespond to this search naturally."
        
        # Call the heavy model with system instruction
        client = genai.Client(api_key=GEMINI_API_KEY)
        config = GenerateContentConfig(
            system_instruction=[SYSTEM_INSTRUCTION],
            temperature=1.1,
            max_output_tokens=MAX_TOKENS_CURRENT
        )
        response = client.models.generate_content(
            model=GEMINI_HEAVY_MODEL,
            contents=[{"role": "user", "parts": [{"text": message_content}]}],
            config=config
        )
        
        bot_content = response.text.strip()
        
        # Clean up the response (similar to generate_bot_reply)
        bot_content = re.sub(r'^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\]\s*', '', bot_content)
        bot_content = re.sub(r'^sakixoxo:\s*', '', bot_content)
        bot_content = re.sub(r'^ASSISTANT:\s*sakixoxo:\s*', '', bot_content)
        bot_content = re.sub(r'^"(.*)"$', r'\1', bot_content)
        bot_content = re.sub(r'\/search\s+.*$', '', bot_content, flags=re.MULTILINE)
        
        # Send the result
        await channel.send(f"**Response with search results:**\n{bot_content}")
        
    except Exception as e:
        await channel.send(f"Error testing search: {str(e)}")


async def send_with_typing_delay(channel, text: str):
    """
    Simulate realistic typing by sending each word with a delay.
    Adds a random delay before typing starts.
    """
    words = text.split()
    MAX_DISCORD_MESSAGE_LENGTH = 2000


    await asyncio.sleep(random.uniform(1, 5))

    async with channel.typing():
        await asyncio.sleep(len(words) * TYPING_DELAY_PER_WORD)

    sent_message = None
    if len(text) > MAX_DISCORD_MESSAGE_LENGTH:
        for i in range(0, len(text), MAX_DISCORD_MESSAGE_LENGTH):
            chunk = text[i:i+MAX_DISCORD_MESSAGE_LENGTH]
            sent_message = await channel.send(chunk)
    else:
        sent_message = await channel.send(text)
    
    return sent_message
        

async def handle_reload_command(message):
    """
    Handles the !reload command by:
    1. Deleting the !reload message
    2. Deleting the bot's last message
    3. Checking if the user edited their last message
    4. Regenerating a response using the edited content if applicable
    """
    channel_id = str(message.channel.id)
    
    await log_message(f"Reload command received in channel {channel_id}", level="INFO")
    
    # Try to delete the !reload command message
    try:
        await message.delete()
    except Exception as e:
        await log_message(f"Could not delete !reload command: {e}", level="ERROR")
    
    # Find the user's last message before the !reload command
    user_last_message = None
    user_edited_content = None
    
    if channel_id in user_messages and user_messages[channel_id]:
        # Get the most recent user message
        user_last_message = user_messages[channel_id][-1]['message_object']
        await log_message(f"Found last user message ID {user_last_message.id} from {user_last_message.created_at.strftime('%Y-%m-%d %H:%M:%S')}", level="DEBUG")
        
        # Check if the message has been edited
        if user_last_message.edited_at:
            await log_message(f"Message was edited at {user_last_message.edited_at.strftime('%Y-%m-%d %H:%M:%S')}", level="INFO")
            
            # Fetch the message to get the updated content
            try:
                # We need to fetch the message to get its current content after edits
                updated_message = await message.channel.fetch_message(user_last_message.id)
                
                # Check if content changed
                original_content = user_messages[channel_id][-1]['content']
                if updated_message.content != original_content:
                    user_edited_content = updated_message.content
                    await log_message(f"User edited their message. Original: '{original_content}' -> New: '{user_edited_content}'")
                    
                    # Update the stored content
                    user_messages[channel_id][-1]['content'] = user_edited_content
                    
                    # Update the memory with the edited content
                    if channel_id in memory_manager.histories and memory_manager.histories[channel_id]:
                        # Find and update the last user message
                        for i in range(len(memory_manager.histories[channel_id])-1, -1, -1):
                            if memory_manager.histories[channel_id][i]["role"] == "user":
                                # Update the content, preserving any attachment descriptions
                                old_content = memory_manager.histories[channel_id][i]["content"]
                                
                                # Extract any attachment descriptions
                                attachment_desc = ""
                                if "\n[" in old_content:
                                    content_parts = old_content.split("\n[", 1)
                                    attachment_desc = "\n[" + content_parts[1] if len(content_parts) > 1 else ""
                                
                                # Update content with edited text plus any attachment descriptions
                                memory_manager.histories[channel_id][i]["content"] = user_edited_content + attachment_desc
                                
                                await log_message(f"Updated memory with edited message at index {i}", level="DEBUG")
                                break
                else:
                    await log_message("Message was edited but content is the same", level="DEBUG")
            except Exception as e:
                await log_message(f"Error fetching edited message: {e}", level="ERROR")
        else:
            await log_message("Last message was not edited", level="DEBUG")
    else:
        await log_message("No previous user messages found in this channel", level="INFO")
    
    # Check if we have a last bot message to delete
    if channel_id in last_bot_message and last_bot_message[channel_id].get("message_object"):
        try:
            # Delete the bot's last message
            await last_bot_message[channel_id]["message_object"].delete()
            await log_message("Deleted bot's last message", level="DEBUG")
            
            # Remove the last bot message from memory
            if channel_id in memory_manager.histories and memory_manager.histories[channel_id]:
                # Find and remove the last assistant message
                for i in range(len(memory_manager.histories[channel_id])-1, -1, -1):
                    if memory_manager.histories[channel_id][i]["role"] == "assistant":
                        memory_manager.histories[channel_id].pop(i)
                        await log_message(f"Removed bot message from memory at index {i}", level="DEBUG")
                        break
        except Exception as e:
            await log_message(f"Could not delete last bot message: {e}", level="ERROR")
    else:
        await log_message("No last bot message found to delete", level="DEBUG")
    
    # If the user edited their message, send a brief typing indicator to acknowledge
    if user_edited_content:
        await log_message("User edited their message, generating new response based on edited content", level="INFO")
        async with message.channel.typing():
            await asyncio.sleep(1)
    else:
        await log_message("Regenerating response with existing content", level="INFO")
    
    # Generate a new response
    bot_reply = await generate_bot_reply(channel_id)
    if bot_reply:
        # Check if the bot decides to not reply
        if "!noresponse" in bot_reply.lower() or bot_reply.strip().lower() == "!noresponse":
            # Simulate typing but do not send any reply
            await log_message("Bot decided not to respond (!noresponse)", level="INFO")
            async with message.channel.typing():
                await asyncio.sleep(2)
            return

        # Add the bot's new reply to the memory and send it
        memory_manager.add_message(channel_id, "assistant", bot_reply, format_est_time())
        await log_message("Added new bot response to memory", level="DEBUG")
        
        # Store the new bot message as the last message
        if channel_id not in last_bot_message:
            last_bot_message[channel_id] = {}
        last_bot_message[channel_id] = {
            "content": bot_reply,
            "message_object": await send_with_typing_delay(message.channel, bot_reply)
        }
        await log_message("Sent new response to channel", level="INFO")

async def save_context(channel, channel_id):
    """
    Saves the current conversation context to a file.
    First tries to save from memory_manager, then falls back to channel history.
    """
    global memory_manager
    if channel_id in memory_manager.histories and memory_manager.histories[channel_id]:
        # Use in-memory history if available
        saved_contexts = {}
        if os.path.exists(context_save_file):
            try:
                with open(context_save_file, 'r') as f:
                    saved_contexts = json.load(f)
            except json.JSONDecodeError:
                saved_contexts = {}
        
        # Add this channel's context to the saved contexts
        timestamp = format_est_time()
        context_name = f"context_{channel_id}_{timestamp.replace(' ', '_').replace(':', '-')}"
        
        saved_contexts[context_name] = {
            "channel_id": channel_id,
            "saved_at": timestamp,
            "history": memory_manager.histories[channel_id]
        }
        
        # Save back to file
        with open(context_save_file, 'w') as f:
            json.dump(saved_contexts, f)
        
        await channel.send(f"Context saved from memory! ID: `{context_name}` ({len(memory_manager.histories[channel_id])} messages)")
    else:
        # Memory is empty, fetch from channel history instead
        await channel.send("No in-memory context found. Fetching message history from channel...")
        context_name, msg_count = await save_channel_history(channel, channel_id)
        if context_name:
            await channel.send(f"Channel history saved! ID: `{context_name}` ({msg_count} messages)")
        else:
            await channel.send("Failed to save channel history.")

async def import_context(channel, channel_id, context_id=None):
    """
    Imports a saved conversation context into the current channel
    """
    global memory_manager
    if not os.path.exists(context_save_file):
        await channel.send("No saved contexts available.")
        return
    
    try:
        with open(context_save_file, 'r') as f:
            saved_contexts = json.load(f)
        
        if not saved_contexts:
            await channel.send("No saved contexts available.")
            return
        
        global memory_manager
        if context_id and context_id in saved_contexts:
            # Import specific context
            memory_manager.histories[channel_id] = saved_contexts[context_id]["history"]
            
            # Ensure token count is accurate based on content (including attachment descriptions)
            memory_manager.token_usage[channel_id] = 0
            for msg in memory_manager.histories[channel_id]:
                # Count tokens in content
                memory_manager.token_usage[channel_id] += len(msg["content"].split())
            
            await channel.send(f"Context `{context_id}` imported successfully!")
        else:
            # List available contexts
            context_list = "\n".join([f"- `{ctx_id}` (Saved: {ctx['saved_at']})" for ctx_id, ctx in saved_contexts.items()])
            await channel.send(f"Available contexts:\n{context_list}\n\nUse `/contextimport ID` to import a specific context.")
    except Exception as e:
        await channel.send(f"Error importing context: {str(e)}")

async def clear_context(channel, context_id=None):
    """
    Clears saved contexts
    """
    if not os.path.exists(context_save_file):
        await channel.send("No saved contexts to clear.")
        return
    
    try:
        if context_id:
            # Clear specific context
            with open(context_save_file, 'r') as f:
                saved_contexts = json.load(f)
            
            if context_id in saved_contexts:
                del saved_contexts[context_id]
                with open(context_save_file, 'w') as f:
                    json.dump(saved_contexts, f)
                await channel.send(f"Context `{context_id}` cleared successfully!")
            else:
                await channel.send(f"Context `{context_id}` not found.")
        else:
            # Clear all contexts
            if os.path.exists(context_save_file):
                os.remove(context_save_file)
            await channel.send("All saved contexts cleared!")
    except Exception as e:
        await channel.send(f"Error clearing context: {str(e)}")

async def list_contexts(channel):
    """
    Lists all saved contexts
    """
    if not os.path.exists(context_save_file):
        await channel.send("No saved contexts available.")
        return
    
    try:
        with open(context_save_file, 'r') as f:
            saved_contexts = json.load(f)
        
        if not saved_contexts:
            await channel.send("No saved contexts available.")
            return
        
        context_list = "\n".join([f"- `{ctx_id}` (Saved: {ctx['saved_at']})" for ctx_id, ctx in saved_contexts.items()])
        await channel.send(f"Available contexts:\n{context_list}")
    except Exception as e:
        await channel.send(f"Error listing contexts: {str(e)}")
        
async def save_channel_history(channel, channel_id, limit=100):
    try:
        saved_contexts = {}
        if os.path.exists(context_save_file):
            with open(context_save_file, 'r') as f:
                saved_contexts = json.load(f)
        
        all_messages = []
        batch_size = 100
        remaining = limit
        last_message_id = None
        
        await log_message(f"Fetching channel history for {channel_id}, limit: {limit} messages", level="INFO")
        
        while remaining > 0:
            current_batch_size = min(batch_size, remaining)
            kwargs = {"limit": current_batch_size}
            if last_message_id:
                kwargs["before"] = discord.Object(id=last_message_id)
            
            batch_messages = []
            async for msg in channel.history(**kwargs):
                batch_messages.append(msg)
            
            if not batch_messages:
                break
                
            all_messages.extend(batch_messages)
            last_message_id = batch_messages[-1].id
            remaining -= len(batch_messages)
            if len(batch_messages) < current_batch_size:
                break
            await asyncio.sleep(1)
        
        messages = []
        for msg in all_messages:
            if msg.type != discord.MessageType.default and msg.type != discord.MessageType.reply:
                continue
                
            role = "assistant" if msg.author == client.user else "user"
            est_time = msg.created_at.replace(tzinfo=timezone.utc).astimezone(timezone(timedelta(hours=-5)))
            timestamp_str = est_time.strftime("%Y-%m-%d %H:%M:%S")
            content = msg.content
            
            if msg.attachments:
                for attachment in msg.attachments:
                    if attachment.content_type and "image" in attachment.content_type:
                        desc = attachment_transcriptions.get(attachment.url, {}).get('transcription', "Image")
                        content += f"\n[Attachment: {desc}]"
                    elif attachment.content_type and "video" in attachment.content_type:
                        desc = attachment_transcriptions.get(attachment.url, {}).get('transcription', "Video")
                        content += f"\n[Attachment: {desc}]"
                    else:
                        content += f"\n[Attachment: {attachment.filename}]"
            
            messages.append({
                "role": role,
                "content": content,
                "time": timestamp_str
            })
        
        current_time = get_est_time()
        time_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")
        context_id = f"context_{channel_id}_{time_str}"
        
        saved_contexts[context_id] = {
            "channel_id": channel_id,
            "saved_at": format_est_time(current_time),
            "history": messages
        }
        
        with open(context_save_file, 'w') as f:
            json.dump(saved_contexts, f, indent=2)
        
        return context_id, len(messages)
    except Exception as e:
        await log_message(f"Error saving channel history: {e}", level="ERROR")
        return None, 0

async def cleanup_freewill_tasks():
    """
    Periodically checks and cleans up any stale or orphaned freewill tasks.
    This helps prevent memory leaks and ensures the bot stays healthy.
    """
    await client.wait_until_ready()  # Make sure client is ready before proceeding
    
    while True:
        try:
            # Check each channel
            for channel_id in list(freewill_channels.keys()):
                channel_data = freewill_channels[channel_id]
                
                # Skip if freewill is not enabled
                if not channel_data.get('enabled', False):
                    continue
                    
                # Check if task exists and is still valid
                task = channel_data.get('task')
                if task is None or task.done() or task.cancelled():
                    # Task is not active - create a new one if freewill is enabled
                    if channel_data.get('enabled', False):
                        await log_message(f"Detected stale task for channel {channel_id}, creating new freewill timer", level="INFO")
                        channel = client.get_channel(int(channel_id))
                        if channel:
                            freewill_channels[channel_id]['task'] = asyncio.create_task(
                                schedule_next_freewill(channel_id, channel, 20)
                            )
                        else:
                            await log_message(f"Could not find channel {channel_id}, disabling freewill", level="WARNING")
                            freewill_channels[channel_id]['enabled'] = False
        except Exception as e:
            try:
                await log_message(f"Error in cleanup_freewill_tasks: {e}", level="ERROR")
            except:
                # If logging itself fails, fallback to print
                print(f"Error in cleanup_freewill_tasks: {e}")
            
        # Run every 30 minutes
        await asyncio.sleep(30 * 60)

async def handle_log_command(message):
    """
    Handles the /log command to set or configure the logging channel
    """
    global log_channel_id, use_log_channel, detailed_logging
    
    parts = message.content.strip().split()
    
    # Check for subcommands
    if len(parts) >= 2:
        subcmd = parts[1].lower()
        
        if subcmd == "channel" and len(parts) >= 3:
            # Set the log channel ID
            try:
                new_channel_id = int(parts[2].strip())
                channel = client.get_channel(new_channel_id)
                
                if channel:
                    old_channel_id = log_channel_id
                    log_channel_id = new_channel_id
                    await message.channel.send(f"Log channel set to {channel.mention}")
                    await log_message(f"Log channel changed from {old_channel_id} to {new_channel_id} by {message.author}", send_to_discord=False)
                    
                    # Send a test message to the new log channel
                    await log_message("Log channel set successfully! You will now receive bot logs in this channel.")
                    
                    # Save the configuration
                    await save_log_config()
                else:
                    await message.channel.send("❌ Invalid channel ID. Please provide a valid channel ID.")
            except ValueError:
                await message.channel.send("❌ Invalid channel ID format. Please provide a numeric ID.")
                
        elif subcmd == "enable":
            # Enable logging to Discord
            use_log_channel = True
            await message.channel.send("Discord logging enabled.")
            await log_message("Discord logging enabled")
            await save_log_config()
            
        elif subcmd == "disable":
            # Disable logging to Discord
            use_log_channel = False
            await message.channel.send("Discord logging disabled. Logs will still be printed to console.")
            print(f"Discord logging disabled by {message.author}")
            await save_log_config()
            
        elif subcmd == "verbose":
            # Toggle detailed logging
            detailed_logging = not detailed_logging
            status = "enabled" if detailed_logging else "disabled"
            await message.channel.send(f"Verbose logging {status}.")
            await log_message(f"Verbose logging {status}")
            await save_log_config()
            
        elif subcmd == "status":
            # Show logging status
            channel = client.get_channel(log_channel_id)
            channel_mention = channel.mention if channel else f"Unknown ({log_channel_id})"
            
            status_msg = (
                f"**Logging Status**\n"
                f"- Log Channel: {channel_mention}\n"
                f"- Discord Logging: {'Enabled' if use_log_channel else 'Disabled'}\n"
                f"- Verbose Mode: {'Enabled' if detailed_logging else 'Disabled'}\n"
            )
            await message.channel.send(status_msg)
        
        else:
            # Unknown subcommand
            await message.channel.send(
                "**Available Log Commands**\n"
                "- `/log channel [ID]` - Set log channel\n"
                "- `/log enable` - Enable Discord logging\n"
                "- `/log disable` - Disable Discord logging\n"
                "- `/log verbose` - Toggle verbose logging\n"
                "- `/log status` - Show logging status"
            )
    else:
        # No subcommand, show help
        await message.channel.send(
            "**Available Log Commands**\n"
            "- `/log channel [ID]` - Set log channel\n"
            "- `/log enable` - Enable Discord logging\n"
            "- `/log disable` - Disable Discord logging\n"
            "- `/log verbose` - Toggle verbose logging\n"
            "- `/log status` - Show logging status"
        )

async def rebuild_context_from_history(channel, channel_id, limit=500, status_message=None):
    """Rebuilds the conversation context by fetching message history from Discord.
    
    Args:
        channel: The Discord channel object
        channel_id: The channel ID as a string
        limit: Maximum number of messages to fetch
        status_message: Optional message to update with progress
    """
    global memory_manager
    try:
        # Ensure the channel_id has an entry in memory_manager.histories
        if channel_id not in memory_manager.histories:
            memory_manager.histories[channel_id] = []
            memory_manager.token_usage[channel_id] = 0
        else:
            # Clear existing history for this channel
            memory_manager.histories[channel_id] = []
            memory_manager.token_usage[channel_id] = 0
        
        # Prepare to fetch messages in batches
        message_count = 0
        processed_count = 0
        batch_size = 100
        remaining = limit
        last_message_id = None
        
        await log_message(f"Rebuilding context for channel {channel_id} from history (limit: {limit})", level="INFO")
        
        if status_message:
            await status_message.edit(content=f"Fetching up to {limit} messages. This may take some time...")
        
        # Collect all messages, going backward in time
        all_messages = []
        
        while remaining > 0:
            current_batch_size = min(batch_size, remaining)
            kwargs = {"limit": current_batch_size}
            if last_message_id:
                kwargs["before"] = discord.Object(id=last_message_id)
            
            batch_messages = []
            async for msg in channel.history(**kwargs):
                batch_messages.append(msg)
                message_count += 1
            
            if not batch_messages:
                break
                
            all_messages.extend(batch_messages)
            last_message_id = batch_messages[-1].id
            remaining -= len(batch_messages)
            
            if status_message and message_count % 50 == 0:
                await status_message.edit(content=f"Fetched {message_count} messages so far...")
            
            if len(batch_messages) < current_batch_size:
                break
                
            await asyncio.sleep(1)
        
        # Sort messages chronologically
        all_messages.sort(key=lambda msg: msg.created_at)
        
        # Process messages
        for msg in all_messages:
            if (msg.type != discord.MessageType.default and msg.type != discord.MessageType.reply) or \
               msg.content.startswith(('/', '!')):
                continue
                
            role = "assistant" if msg.author == client.user else "user"
            est_time = msg.created_at.replace(tzinfo=timezone.utc).astimezone(timezone(timedelta(hours=-5)))
            timestamp_str = est_time.strftime("%Y-%m-%d %H:%M:%S")
            content = msg.content
            
            # Handle attachments
            if msg.attachments:
                for attachment in msg.attachments:
                    if attachment.content_type and "image" in attachment.content_type:
                        desc = attachment_transcriptions.get(attachment.url, {}).get('transcription', "Image")
                        content += f"\n[Attachment: {desc}]"
                    elif attachment.content_type and "video" in attachment.content_type:
                        desc = attachment_transcriptions.get(attachment.url, {}).get('transcription', "Video")
                        content += f"\n[Attachment: {desc}]"
                    else:
                        content += f"\n[Attachment: {attachment.filename}]"
            
            # Use MemoryManager's method to add the message
            memory_manager.add_message(channel_id, role, content, timestamp_str)
            processed_count += 1
        
        await log_message(f"Successfully rebuilt context for channel {channel_id}: Processed {processed_count} of {message_count} messages", level="INFO")
        
        if status_message:
            await status_message.edit(content=f"Context rebuilt! Processed {processed_count} messages.")
        
        return processed_count
    except Exception as e:
        await log_message(f"Error rebuilding context: {e}", level="ERROR")
        if status_message:
            await status_message.edit(content=f"Error rebuilding context: {e}")
        return 0

async def handle_contextregen_command(message):
    """
    Handles the /contextregen command to rebuild the bot's memory from channel history.
    This is useful after bot restarts to restore conversation context without using context import/export.
    """
    channel_id = str(message.channel.id)
    parts = message.content.split(maxsplit=1)
    
    # Extract optional limit parameter
    try:
        limit = int(parts[1]) if len(parts) > 1 else 500  # Default to 500 if no limit provided
        limit = min(limit, 500)  # Ensure it doesn't exceed 500
    except ValueError:
        limit = 500  # Default to 500 if user provides invalid input
    
    # Send initial status message
    status_message = await message.channel.send(f"Rebuilding context from up to {limit} messages. This may take some time...")
    
    # Rebuild the context
    message_count = await rebuild_context_from_history(message.channel, channel_id, limit, status_message)
    
    if message_count > 0:
        # Success message is handled by the rebuild function via status_message updates
        pass
    else:
        await status_message.edit(content="Failed to rebuild context from channel history.")

def run_bot():
    client.run(DISCORD_BOT_TOKEN)


if __name__ == "__main__":
    run_bot()


