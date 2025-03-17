# saki
hyper realistic companion on discord (DM or invite to server) with:
- free will
- advanced context/memory
- realistic conversational abilities
- search abilities
- decide when/if to respond
- multimodal input/outputs (send it images/video n receive images/video)
- time-based world understanding (local time built into context, system prompting, and all messages)\

these features are all operated through a combination of tool calling (by free will of the bot whenever it wants), good system prompting, and background api calls to handle free will/memory systems.

idea behind this was prompted by seeing how awful other llm powered roleplay and companian products were designed. also so i can test out and use gemini since they have high free api ratelimits (good exp w gemini! i could not have done this project as easily w other models).

the main thing ive optimized for in design is for immersion. you want saki to feel real when talking to it, all other tangental optimization targets will follow accordingly by aiming for realism.

calls are not just made per interaction, just to operate the bot and its free will and memory systems calls are made even with complete idle conversation. in a complete idle state minimum 3 calls are made per hour.
direct interactions and i/o are handled by 2.0 flash.
2.0 flash lite is used to operate the context and memory system, we have long context (realistically we could do up to 1m, but <100k keeps conversation better imo). we use flash lite to summarize and store all events after 50k tokens up to 100k (limits can all be adjusted in config).



features to be implemented:
- gallery tool (better tooling to let saki send more media)
- use vc
- "game" mode 

