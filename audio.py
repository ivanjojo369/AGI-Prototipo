from utils.logger import Logger
from gtts import gTTS

text = """
Hello, I'm Ivan. I'm a Philosophy student and AGI developer, working at the intersection of consciousness, ethics, and emerging technology.

I'm building a general artificial intelligence prototype that integrates planning, contextual memory, and metacognitive reflection. My goal is to align AGI with human values by combining knowledge from neuroscience, phenomenology, and quantum physics.

This fellowship would allow me to improve the system, connect with visionary thinkers, and scale the project globally.

If awarded the $10,000 prize, I would invest it in expanding the cognitive modules, supporting open-source development, and conducting workshops to inspire young creators in ethical AI innovation.
"""

tts = gTTS(text=text, lang='en')
tts.save("video_pitch_audio.mp3")
Logger.info("✅ Audio saved as video_pitch_audio.mp3")
