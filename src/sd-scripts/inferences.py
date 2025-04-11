from diffusers import StableDiffusionPipeline
import torch
import xformers
import sys

def txt_to_dict(file_path):
    result_dict = {}
    
    with open(file_path, 'r') as file:
        for line in file:
            key, value = line.strip().split(' = ', 1)
            if value and value.isspace():
                continue
            result_dict[key] = value
    
    return result_dict

prompt_file = ""
data_path = ""
if len(sys.argv) > 1:
    prompt_file = sys.argv[1]
    data_path = sys.argv[2]

prompt_dict = txt_to_dict(prompt_file)
if 'content' not in prompt_dict:
    prompt_dict['content'] = "A painting"

if 'genre' not in prompt_dict:
    prompt_dict['genre']  = None

if 'artist' not in prompt_dict:
    prompt_dict['artist'] = None

if 'style' not in prompt_dict:
    prompt_dict['style']  = None

prompt = prompt_dict['content']
if prompt_dict['genre'] is not None:
    prompt += ' in the genre of ' + prompt_dict['genre'] 

if prompt_dict['artist'] is not None:
    prompt += ' painted by ' + prompt_dict['artist'].title() 

if prompt_dict['style'] is not None:
    prompt += ' in the style of ' + prompt_dict['style']

prompt += '.'
print(prompt_dict)
print(prompt)

pipe = StableDiffusionPipeline.from_single_file("ckpt/v1-5-pruned-emaonly.safetensors",
                                                    torch_dtype=torch.float16).to("cuda")
pipe.enable_xformers_memory_efficient_attention()
pipe.load_lora_weights(data_path + "/lora_sdscipts_example_data/models", weight_name="sunflower.safetensors", adapter_name="cityscape")

# pipe.set_adapters(["cityscape"], adapter_weights=[0.3])
lora_scale = 0.7
image = pipe(
    prompt, num_inference_steps=50, cross_attention_kwargs={"scale": lora_scale}, generator=torch.manual_seed(0)
).images[0]

image.save(f"output-{prompt_dict['file']}.jpg")