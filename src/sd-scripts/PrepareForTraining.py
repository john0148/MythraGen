import sys

def read_kth_line(file_path, k):
    with open(file_path, 'r') as file:
        for current_line_number, line in enumerate(file, start=1):
            if current_line_number == k:
                return line.strip()
    return None

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

print(f"prompt file: {prompt_file}")
print(f"data path: {data_path}")

prompt = txt_to_dict(prompt_file)
if 'content' not in prompt:
    prompt['content'] = None

if 'genre' not in prompt:
    prompt['genre']  = None

if 'artist' not in prompt:
    prompt['artist'] = None

if 'style' not in prompt:
    prompt['style']  = None


numOfImgs = int(prompt['size'])
print(prompt)

import numpy as np
import torch
from tqdm import tqdm
from PIL import Image

with open('/home/sv-lkhai/WikiArt/MetaData/images.txt', 'r') as file:
    image_paths = [line.strip() for line in file]

for i in range(len(image_paths)):
    image_paths[i] = '/home/sv-lkhai/WikiArt/' + "/".join(image_paths[i].split("\\"))

with open('/home/sv-lkhai/WikiArt/MetaData/metadata.txt', 'r') as file:
    metadata = [line.strip() for line in file]

image_caption_labels = []
for i in range(len(metadata)):
    image_caption_labels.append(" ".join(metadata[i].split(" in the style of ")[:-1]))
    
image_artist_labels = []
for i in range(len(image_paths)):
    image_artist_labels.append(" ".join(image_paths[i].split("/")[-1].split("_")[0].split("-")))
    
image_genre_labels = []
for i in range(len(metadata)):
    image_genre_labels.append(metadata[i].split(" in the style of ")[-1])
    
info = []
for i in range(len(image_caption_labels)):
    info.append([image_caption_labels[i], image_artist_labels[i], image_genre_labels[i]])

# Load BLIP-2
from lavis.models import load_model_and_preprocess
import torch
from transformers import BertTokenizer
import torch.nn.functional as F

def init_model(model_size = 'pretrain'): #model_size must be "pretrain"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vis_processors, txt_processors = load_model_and_preprocess(name = "blip2_feature_extractor", 
                                                                          model_type = model_size, 
                                                                          is_eval = True, 
                                                                          device = device)
    return model, vis_processors, txt_processors

def init_tokenizer(): 
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.add_special_tokens({"bos_token": "[DEC]"})
    tokenizer.add_special_tokens({"additional_special_tokens": ["[ENC]"]})
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]
    return tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, vis_processors, txt_processors = init_model('pretrain_vitL')#pretrain, pretrain_vitL, coco
tokenizer = init_tokenizer()


# Embedding function (Image embedding, Text embedding)
def image_encoder(image, model, vis_processors, device):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image_processed = vis_processors["eval"](image).unsqueeze(0).to(device)
    with model.maybe_autocast():
        image_embeds = model.ln_vision(model.visual_encoder(image_processed))
    image_embeds = image_embeds.float()
    image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                image_embeds.device
            )
    query_tokens = model.query_tokens.expand(image_embeds.shape[0], -1, -1)
    query_output = model.Qformer.bert(
                        query_embeds=query_tokens,
                        encoder_hidden_states=image_embeds,
                        encoder_attention_mask=image_atts,
                        return_dict=True,
                    )
    image_feats = F.normalize(model.vision_proj(query_output.last_hidden_state), dim=-1)
  
    return image_feats[0][0].detach().cpu().numpy()

def text_encoder(text, model, tokenizer, text_processors, device):
    text_input = text_processors["eval"](text)
    text = tokenizer(text_input, return_tensors="pt", padding=True).to(device)
    text_output = model.Qformer.bert(
                text.input_ids,
                attention_mask=text.attention_mask,
                return_dict=True,
            )
    text_feat = F.normalize(
                model.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
            )
    return text_feat[0].detach().cpu().numpy()

# Load data embedding
import pickle
with open('/home/sv-lkhai/WikiArt/Embedding/Pretrained-ViTL/WikiArt_texts_embedding.pkl', 'rb') as fp:
    texts_embedding = pickle.load(fp)
    
with open('/home/sv-lkhai/WikiArt/Embedding/Pretrained-ViTL/WikiArt_images_embeddings.pkl', 'rb') as fp:
    images_embedding = pickle.load(fp)
       
with open('/home/sv-lkhai/WikiArt/Embedding/Pretrained-ViTL/WikiArt_artists_embedding.pkl', 'rb') as fp:
    artists_embedding = pickle.load(fp)
    
with open('/home/sv-lkhai/WikiArt/Embedding/Pretrained-ViTL/WikiArt_styles_embedding.pkl', 'rb') as fp:
    styles_embedding = pickle.load(fp)
    
with open('/home/sv-lkhai/WikiArt/Embedding/Pretrained-ViTL/WikiArt_genres_embedding.pkl', 'rb') as fp:
    genres_embedding = pickle.load(fp)

images_embedding  = torch.from_numpy(images_embedding).to(device)
artists_embedding = torch.from_numpy(artists_embedding).to(device)
styles_embedding  = torch.from_numpy(styles_embedding).to(device)
genres_embedding  = torch.from_numpy(genres_embedding).to(device)
texts_embedding   = torch.from_numpy(texts_embedding).to(device)

# Create synthetic vector
alpha = 0.1
beta = 0.1
gamma = 0.75

weights_vector = torch.tensor([gamma, beta, alpha], dtype=torch.float32).view(1, -1).to(device)
synthetic_vector_embedding = None

for i in tqdm(range(texts_embedding.shape[0])):
    concat_matrix = torch.cat((images_embedding[i].view(1, -1), texts_embedding[i].view(1, -1), genres_embedding[i].view(1, -1)), dim = 0)
    res = weights_vector @ concat_matrix

    if synthetic_vector_embedding is None:
        synthetic_vector_embedding = res
    else:
        synthetic_vector_embedding = torch.cat((synthetic_vector_embedding, res), dim = 0)

synthetic_vector_embedding /= synthetic_vector_embedding.norm(dim=-1, keepdim=True)

# Create feature vector
alpha, beta, gamma, theta = 1, 1, 1, 1
weights_vector = torch.tensor([theta, gamma, beta, alpha], dtype=torch.float32).to(device)
vector_embedding = None

for i in tqdm(range(texts_embedding.shape[0])):
    concat_matrix = torch.cat((weights_vector[0] * synthetic_vector_embedding[i].view(1, -1), weights_vector[1] * genres_embedding[i].view(1, -1), weights_vector[2] * artists_embedding[i].view(1, -1), weights_vector[3] * styles_embedding[i].view(1, -1)), dim = 1)
    
    if vector_embedding is None:
        vector_embedding = concat_matrix
    else:
        vector_embedding = torch.cat((vector_embedding, concat_matrix), dim = 0)

# Indexing
import faiss
# Image indexing
images_index = faiss.IndexFlatIP(images_embedding.shape[1])
images_index.add(images_embedding.detach().cpu().numpy())
# Text indexing
texts_index = faiss.IndexFlatIP(texts_embedding.shape[1])
texts_index.add(texts_embedding.detach().cpu().numpy())
# Genre indexing
genres_index = faiss.IndexFlatIP(genres_embedding.shape[1])
genres_index.add(genres_embedding.detach().cpu().numpy())
# Artist indexing
artists_index = faiss.IndexFlatIP(artists_embedding.shape[1])
artists_index.add(artists_embedding.detach().cpu().numpy())
# Style indexing
styles_index = faiss.IndexFlatIP(styles_embedding.shape[1])
styles_index.add(styles_embedding.detach().cpu().numpy())
# Feature indexing
vector_index = faiss.IndexFlatIP(vector_embedding.shape[1])
vector_index.add(vector_embedding.detach().cpu().numpy())

# Prepare for searching
known_styles = [
    'abstract expressionism', 'action painting', 'analytical cubism', 'art nouveau', 'baroque',
    'color field painting', 'contemporary realism', 'cubism', 'early renaissance', 'expressionism',
    'fauvism', 'high renaissance', 'impressionism', 'mannerism', 'minimalism',
    'naive art primitivism', 'new realism', 'northern renaissance', 'pointillism', 'pop art',
    'post-impressionism', 'realism', 'rococo', 'romanticism', 'symbolism', 'synthetic cubism', 'ukiyo-e'
]

known_artists = [
    'unknown artist', 'boris kustodiev', 'camille pissarro', 'childe hassam', 'claude monet', 
    'edgar degas', 'eugene boudin', 'gustave dore', 'ilya repin', 'ivan aivazovsky', 
    'ivan shishkin', 'john singer sargent', 'marc chagall', 'martiros saryan', 'nicholas roerich', 
    'pablo picasso', 'paul cezanne', 'pierre auguste renoir', 'pyotr konchalovsky', 
    'raphael kirchner', 'rembrandt', 'salvador dali', 'vincent van gogh', 'hieronymus bosch', 
    'leonardo da vinci', 'albrecht durer', 'edouard cortes', 'sam francis', 'juan gris', 
    'lucas cranach the elder', 'paul gauguin', 'konstantin makovsky', 'egon schiele', 
    'thomas eakins', 'gustave moreau', 'francisco goya', 'edvard munch', 'henri matisse', 
    'fra angelico', 'maxime maufra', 'jan matejko', 'mstislav dobuzhinsky', 'alfred sisley', 
    'mary cassatt', 'gustave loiseau', 'fernando botero', 'zinaida serebriakova', 
    'georges seurat', 'isaac levitan', 'joaquín sorolla', 'jacek malczewski', 'berthe morisot', 
    'andy warhol', 'arkhip kuindzhi', 'niko pirosmani', 'james tissot', 'vasily polenov', 
    'valentin serov', 'pietro perugino', 'pierre bonnard', 'ferdinand hodler', 
    'bartolome esteban murillo', 'giovanni boldini', 'henri martin', 'gustav klimt', 
    'vasily perov', 'odilon redon', 'tintoretto', 'gene davis', 'raphael', 
    'john henry twachtman', 'henri de toulouse lautrec', 'antoine blanchard', 
    'david burliuk', 'camille corot', 'konstantin korovin', 'ivan bilibin', 'titian', 
    'maurice prendergast', 'edouard manet', 'peter paul rubens', 'aubrey beardsley', 
    'paolo veronese', 'joshua reynolds', 'kuzma petrov vodkin', 'gustave caillebotte', 
    'lucian freud', 'michelangelo', 'dante gabriel rossetti', 'felix vallotton', 
    'nikolay bogdanov belsky', 'georges braque', 'vasily surikov', 'fernand leger', 
    'konstantin somov', 'katsushika hokusai', 'sir lawrence alma tadema', 'vasily vereshchagin', 
    'ernst ludwig kirchner', 'mikhail vrubel', 'orest kiprensky', 'william merritt chase', 
    'aleksey savrasov', 'hans memling', 'amedeo modigliani', 'ivan kramskoy', 
    'utagawa kuniyoshi', 'gustave courbet', 'william turner', 'theo van rysselberghe', 
    'joseph wright', 'edward burne jones', 'koloman moser', 'viktor vasnetsov', 
    'anthony van dyck', 'raoul dufy', 'frans hals', 'hans holbein the younger', 
    'ilya mashkov', 'henri fantin latour', 'm.c. escher', 'el greco', 'mikalojus ciurlionis', 
    'james mcneill whistler', 'karl bryullov', 'jacob jordaens', 'thomas gainsborough', 
    'eugene delacroix', 'canaletto'
]

known_genres = [
    'abstract painting', 'cityscape', 'genre painting', 'illustration', 'landscape', 
    'nude painting', 'portrait', 'religious painting', 'sketch and study', 'still life', 'animal', 'plant'
]

def get_text_info(context, known_list, model, tokenizer, txt_processors, device):
    if context is None:
        return np.zeros(256)
    
    if any(extract_info in context.lower() for extract_info in known_list):
        return text_encoder(context, model, tokenizer, txt_processors, device)
    
    return np.zeros(256)

def get_content_text(context, model, tokenizer, txt_processors, device):   
    if context is None:
        return np.zeros(256)
    return text_encoder(context, model, tokenizer, txt_processors, device)

import pickle

with open('/home/sv-lkhai/WikiArt/MetaData/genre_labels.pkl', 'rb') as file:
    genre_labels = pickle.load(file)

with open('/home/sv-lkhai/WikiArt/MetaData/name_genre.pkl', 'rb') as file:
    genre_name = pickle.load(file)

genre_name[140] = 'Animal'
genre_name[141] = 'Plant'
genre_name[142] = 'Object painting'
genre_name[143] = 'Fruit'

import pandas as pd
from PIL import Image
from pathlib import Path

# Assuming the metadata and image paths are in a CSV file
image_folder = '/home/sv-lkhai/WikiArt/Data'
metadata_path = Path(image_folder)/'classes.csv'
metadata_path2 = Path(image_folder)/'wclasses.csv'
label_images = '/home/sv-lkhai/WikiArt/MetaData/label_images_v2.csv'

df = pd.read_csv(metadata_path2)
replace_df = pd.read_csv(label_images)

replaceDataset_idx = df[df['genre'].isin([137, 138, 139])].index
idx2 = 0
for idx in replaceDataset_idx:
    df.loc[idx, 'genre'] = replace_df.iloc[idx2]['genre']
    idx2 += 1

replaceDataset_idx = df[df['genre'] == 142].index
for idx in replaceDataset_idx:
    df.loc[idx, 'genre'] = 138

dataset = pd.read_csv('/home/sv-lkhai/WikiArt/Data/wclasses.csv')
sketch_index = dataset[dataset['genre'] == 137]['file'].index
for idx in sketch_index:
    df.loc[idx, 'genre'] = 137

import time

start_time =time.time()
text_search_embedding  = get_content_text(prompt['content'],
                                                  model, 
                                                  tokenizer, 
                                                  txt_processors, 
                                                  device)

genre_input_embedding  = get_text_info(prompt['genre'], 
                                        known_genres,
                                        model, 
                                        tokenizer, 
                                        txt_processors, 
                                        device)

artist_input_embedding = get_text_info(prompt['artist'], 
                                        known_artists,
                                        model, 
                                        tokenizer, 
                                        txt_processors, 
                                        device)

style_input_embedding  = get_text_info(prompt['style'], 
                                        known_styles,
                                        model, 
                                        tokenizer, 
                                        txt_processors, 
                                        device)

query_search = np.concatenate((text_search_embedding.reshape(1, -1), 
                                genre_input_embedding.reshape(1, -1), 
                                artist_input_embedding.reshape(1, -1), 
                                style_input_embedding.reshape(1, -1)), axis = 1)



print(text_search_embedding)
print(genre_input_embedding)
print(artist_input_embedding)
print(style_input_embedding)


#search
distances, indices = vector_index.search(query_search, numOfImgs)

end_time = time.time()
execution_time = end_time - start_time
print(execution_time)

distances = distances[0]
indices = indices[0]

indices_distances = list(zip(indices, distances))
indices_distances.sort(key=lambda x: x[1])  # Sort based on the distances

print(indices_distances)
print(indices)

similar_images = [image_paths[idx] for idx in indices]
caption = [info[idx] for idx in indices]
imgs_dis = [Image.open(similar_images[i]) for i in range(len(similar_images))]
labels = []
for idx in indices:
    if image_paths[idx].replace("/home/sv-lkhai/WikiArt/Data/", "") in list(df['file'].values):
        labels.append(genre_name[df[df['file'] ==  image_paths[idx].replace("/home/sv-lkhai/WikiArt/Data/", "")]['genre'].values[0]])
    else:
        labels.append('unknown genre')

# Remove previous file
import os
import glob
import shutil

models_path = data_path + '/lora_sdscipts_example_data/models'
logs_path = data_path + '/lora_sdscipts_example_data/logs'

for path in [models_path, logs_path]:
    if os.path.exists(path):
        try:
            shutil.rmtree(path)
            print(f"Folder '{path}' has been removed successfully.")
        except OSError as e:
            print(f"Error: {e.strerror} - {e.filename}")

imgs_path = data_path + '/lora_sdscipts_example_data/imgs'
files = glob.glob(os.path.join(imgs_path, '*'))
for file in files:
    try:
        os.remove(file)
        print(f"File deleted: {file}")
    except OSError as e:
        print(f"Error: {e.strerror} - {e.filename}")

# Write new file for training.
new_size = (512, 512)
for idx, img in enumerate(imgs_dis):
    output_image_path = os.path.join(imgs_path, f'{idx + 1}.jpg')
    output_caption_path = os.path.join(imgs_path, f'{idx + 1}.txt')
    resized_img = img.resize(new_size, Image.Resampling.LANCZOS)
    resized_img.save(output_image_path)
    with open(output_caption_path, 'w', encoding='utf-8') as file:
        description = caption[idx][0]
        if prompt['genre'] is not None:
            description += ' in the genre of ' + labels[idx] 
        
        if prompt['artist'] is not None:
            description += ' painted by ' + caption[idx][1].title() 
        
        if prompt['style'] is not None:
            description += ' in the style of ' + caption[idx][2] 
        
        description += '.'
        file.write(description)

prompt = read_kth_line(data_path + "/lora_sdscipts_example_data/imgs/1.txt", 1)
with open(data_path + "/lora_sdscipts_example_data/val_prompt.txt", 'w', encoding='utf-8') as file:
    file.write(prompt + ' --n low quality, worst quality, bad anatomy, bad composition, poor --w 512 --h 512 --l 5 --s 28' + '\n')


for i in range(2, 5):
    prompt = read_kth_line(f"{data_path}/lora_sdscipts_example_data/imgs/{i}.txt", 1)
    with open(data_path + "/lora_sdscipts_example_data/val_prompt.txt", 'a', encoding='utf-8') as file:
        file.write(prompt + ' --n low quality, worst quality, bad anatomy, bad composition, poor --w 512 --h 512 --l 5 --s 28' + '\n')

prompt = read_kth_line(data_path + "/lora_sdscipts_example_data/imgs/5.txt", 1)
with open(data_path + "/lora_sdscipts_example_data/val_prompt.txt", 'a', encoding='utf-8') as file:
    file.write(prompt + ' --n low quality, worst quality, bad anatomy, bad composition, poor --w 512 --h 512 --l 5 --s 28')
