import openvino_genai as ov_genai
from pathlib import Path


model_path = Path(__file__).parent / "model_converted"
pipe = ov_genai.LLMPipeline(str(model_path), "GPU")

# Create a streamer function
def streamer(subword):
    print(subword, end='', flush=True)
    # Return flag corresponds whether generation should be stopped.
    return ov_genai.StreamingStatus.RUNNING

pipe.start_chat()
while True:
    try:
        prompt = input('question:\n')
    except EOFError:
        break
    pipe.generate(prompt, streamer=streamer, max_new_tokens=100)
    print('\n----------\n')
pipe.finish_chat()