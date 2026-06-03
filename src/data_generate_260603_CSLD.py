import os
import pandas as pd
import numpy as np
import random
from scipy.stats import truncnorm

this_dir = os.getcwd()
work_dir = os.path.dirname(this_dir)
write_dir = "/mnt/storage/franklhtan/projects/LearnCsld2025/data_CSLD/"    # NOTE: VW=vowel, this is for extended study, we also normalized it using Global Min-Max
# NOTE: this is repaired version of vowel: F2 was not properly generated; although it may not affect the results currently, it will largely interupt later condition organizations. 

manipulant_means = [-1, -1, 0, 0, 0]   # cog, fri_dur, f1, f2, f3
manipulant_stds = [500, 13, 0, 0, 0]   # cog, fri_dur, f1, f2, f3: NOTE: we make sure the variance is the same across the two dimensions relative to range.
concurrent_means = [0, 0, 936, 1551, 2815]   # vowel: cog, fri_dur, f1, f2, f3
concurrent_stds = [0, 0, 93.6, 155.1, 281.5]

all_means = concurrent_means + manipulant_means + concurrent_means # CVC
all_stds = concurrent_stds + manipulant_stds + concurrent_stds

any_mins = [0, 0, 0, 0, 0]   # cog, fri_dur, f1, f2, f3
any_maxs = [10000, 260, 1872, 3102, 5630]

print(all_means)
print(all_stds)
print(any_mins)
print(any_maxs)

def generate_data_for_one_dim(mean, std, a=-2, b=2, sample_size=8000): 
    # NOTE: it turns out that truncnorm can take std = 0, then we can just fill in everything directly. 
    dist = truncnorm(a, b, loc = mean, scale = std)
    return dist.rvs(size = sample_size)

condition = "test_equal"
data_dir = os.path.join(work_dir, f'data_{condition}')
os.makedirs(data_dir, exist_ok=True)

np.random.seed(42)

manipulant_list = ['s', 'c', 'ts', 'tc', 
                   'sh', 'ch', 'tsh', 'tch']
manipulant_cog = [4000, 7000, 4000, 7000, 
                 5000, 8000, 5000, 8000]
manipulant_fd = [174, 174, 96, 96, 
                  134, 134, 56, 56]

train_refs = [1, 2, 3, 4, 
              5, 6, 7, 8]

# no. of tokens for each word
sample_size = 2000

metadata = []

for index in range (8):
    consonant = manipulant_list[index]
    cog = manipulant_cog[index]
    fd = manipulant_fd[index]
    train_ref = train_refs[index]

    vowel = 'a' # use one vowel as the context first.
    word = vowel + consonant + vowel # VCV structure

    target_means = np.array([cog, fd])
    manipulant_means_actual = np.concatenate((target_means, np.array(manipulant_means[2:])))  # cog, fri_dur are replaced by the specific values for each consonant, but the rest are the same across all consonants.
    manipulant_stds_actual = np.array(manipulant_stds)

    manipulants = np.zeros((sample_size, len(manipulant_means_actual)))
    for i in range(len(manipulant_means_actual)): # NOTE: this takes care of all dims, but only the related ones have value
        manipulants[:, i] = generate_data_for_one_dim(manipulant_means_actual[i], manipulant_stds_actual[i], sample_size=sample_size)

    concurrent_means_actual = np.array(concurrent_means)
    concurrent_stds_actual = np.array(concurrent_stds)

    concurrents = np.zeros((sample_size, len(concurrent_means_actual)))
    for i in range(len(concurrent_means_actual)):
        concurrents[:, i] = generate_data_for_one_dim(concurrent_means_actual[i], concurrent_stds_actual[i], sample_size=sample_size)

    subdata_dir = os.path.join(data_dir, word)
    os.makedirs(subdata_dir, exist_ok=True)

    manipulants = manipulants[:, np.newaxis, :]
    concurrents = concurrents[:, np.newaxis, :]

    all_tokens = np.concatenate([concurrents, manipulants, concurrents], axis=1)
    all_mins = np.array([any_mins, any_mins, any_mins])
    all_maxs = np.array([any_maxs, any_maxs, any_maxs])

    norm_all_tokens = (all_tokens - all_mins) / (all_maxs - all_mins)

    for i in range(sample_size):
        uid = word + f'_{i+1:04d}'
        filename = f'{uid}.npy'
        save_path = os.path.join(subdata_dir, filename)
        token = norm_all_tokens[i]
        np.save(save_path, token)

        # save_path_rel = os.path.relpath(save_path, start=work_dir)
        save_path_rel = os.path.join(write_dir, f'data_{condition}', word, filename)
        metadata.append({
            'uid': uid,
            'path': save_path_rel,
            'cog': token[1][0],
            'fd': token[1][1],
            'word': word,
            'vowel': vowel,
            'consonant': consonant,
            'train': train_ref
        })

csv_name = f'metadata_{condition}.csv'
csv_path = os.path.join(data_dir, csv_name)
metaframe = pd.DataFrame(metadata)
metaframe.to_csv(csv_path, index=False)

condition = "train_equal"
data_dir = os.path.join(work_dir, f'data_{condition}')
os.makedirs(data_dir, exist_ok=True)

np.random.seed(42)

manipulant_list = ['s', 'c', 'ts', 'tc', 
                   'sh', 'ch', 'tsh', 'tch']
manipulant_cog = [4000, 7000, 4000, 7000, 
                 5000, 8000, 5000, 8000]
manipulant_fd = [174, 174, 96, 96, 
                  134, 134, 56, 56]

train_refs = [1, 2, 3, 4, 
              5, 6, 7, 8]

# no. of tokens for each word
sample_size = 8000

metadata = []

for index in range (8):
    consonant = manipulant_list[index]
    cog = manipulant_cog[index]
    fd = manipulant_fd[index]
    train_ref = train_refs[index]

    vowel = 'a' # use one vowel as the context first.
    word = vowel + consonant + vowel # VCV structure

    target_means = np.array([cog, fd])
    manipulant_means_actual = np.concatenate((target_means, np.array(manipulant_means[2:])))  # cog, fri_dur are replaced by the specific values for each consonant, but the rest are the same across all consonants.
    manipulant_stds_actual = np.array(manipulant_stds)

    manipulants = np.zeros((sample_size, len(manipulant_means_actual)))
    for i in range(len(manipulant_means_actual)): # NOTE: this takes care of all dims, but only the related ones have value
        manipulants[:, i] = generate_data_for_one_dim(manipulant_means_actual[i], manipulant_stds_actual[i], sample_size=sample_size)

    concurrent_means_actual = np.array(concurrent_means)
    concurrent_stds_actual = np.array(concurrent_stds)

    concurrents = np.zeros((sample_size, len(concurrent_means_actual)))
    for i in range(len(concurrent_means_actual)):
        concurrents[:, i] = generate_data_for_one_dim(concurrent_means_actual[i], concurrent_stds_actual[i], sample_size=sample_size)

    subdata_dir = os.path.join(data_dir, word)
    os.makedirs(subdata_dir, exist_ok=True)

    manipulants = manipulants[:, np.newaxis, :]
    concurrents = concurrents[:, np.newaxis, :]

    all_tokens = np.concatenate([concurrents, manipulants, concurrents], axis=1)
    all_mins = np.array([any_mins, any_mins, any_mins])
    all_maxs = np.array([any_maxs, any_maxs, any_maxs])

    norm_all_tokens = (all_tokens - all_mins) / (all_maxs - all_mins)

    for i in range(sample_size):
        uid = word + f'_{i+1:04d}'
        filename = f'{uid}.npy'
        save_path = os.path.join(subdata_dir, filename)
        token = norm_all_tokens[i]
        np.save(save_path, token)

        # save_path_rel = os.path.relpath(save_path, start=work_dir)
        save_path_rel = os.path.join(write_dir, f'data_{condition}', word, filename)
        metadata.append({
            'uid': uid,
            'path': save_path_rel,
            'cog': token[1][0],
            'fd': token[1][1],
            'word': word,
            'vowel': vowel,
            'consonant': consonant,
            'train': train_ref
        })

csv_name = f'metadata_{condition}.csv'
csv_path = os.path.join(data_dir, csv_name)
metaframe = pd.DataFrame(metadata)
metaframe.to_csv(csv_path, index=False)
