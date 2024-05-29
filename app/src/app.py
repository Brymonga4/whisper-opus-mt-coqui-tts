import os
import numpy as np
import torch
import whisper
from services.methods import Methods


CURRENT_EXECT_PATH = os.getcwd()
TEST_FILES_PATH = os.path.join(CURRENT_EXECT_PATH,"app","src", "samples")
OUTPUT_FILES_PATH = os.path.join(CURRENT_EXECT_PATH,"app","src", "output")


device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model("base")

print(
    f"{sum(np.prod(p.shape) for p in model.parameters()):,} parameters."
)

#Methods.options_translate_transcribe()

#Methods.process_audio_locally(device, model,
#                              os.path.join(TEST_FILES_PATH, "donquixote_my_voice_spa.wav"),
#                             os.path.join(OUTPUT_FILES_PATH, "output.wav"))

#Methods.transcribe_audio_to_text(model,
#                                os.path.join(TEST_FILES_PATH, "be_drunk_myvoice.wav"))

Methods.process_audio_locally_Marian(device, model,
                                    os.path.join(TEST_FILES_PATH, "leonor.wav"),
                                    os.path.join(OUTPUT_FILES_PATH, "output.wav"),
                                    "spanish",
                                    "it"
)

# ['en', 'fr', 'de', 'it', 'ar']

