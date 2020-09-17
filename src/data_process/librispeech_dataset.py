import os
import json
from typing import Tuple
import pickle as pkl
import torch
import torchaudio
from torch import Tensor
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torchaudio.datasets.utils import (
    download_url,
    extract_archive,
    walk_files,
)

from data_help import LetterTransform, WordTransform

import pdb

PRE_TRAINED_MODEL = "bert-base-uncased"
MAX_LEN = 128
DO_LOWER_CASE = True
URL = "train-clean-100"
FOLDER_IN_ARCHIVE = "LibriSpeech"

_CHECKSUMS = {
    "http://www.openslr.org/resources/12/dev-clean.tar.gz":
    "76f87d090650617fca0cac8f88b9416e0ebf80350acb97b343a85fa903728ab3",
    "http://www.openslr.org/resources/12/dev-other.tar.gz":
    "12661c48e8c3fe1de2c1caa4c3e135193bfb1811584f11f569dd12645aa84365",
    "http://www.openslr.org/resources/12/test-clean.tar.gz":
    "39fde525e59672dc6d1551919b1478f724438a95aa55f874b576be21967e6c23",
    "http://www.openslr.org/resources/12/test-other.tar.gz":
    "d09c181bba5cf717b3dee7d4d592af11a3ee3a09e08ae025c5506f6ebe961c29",
    "http://www.openslr.org/resources/12/train-clean-100.tar.gz":
    "d4ddd1d5a6ab303066f14971d768ee43278a5f2a0aa43dc716b0e64ecbbbf6e2",
    "http://www.openslr.org/resources/12/train-clean-360.tar.gz":
    "146a56496217e96c14334a160df97fffedd6e0a04e66b9c5af0d40be3c792ecf",
    "http://www.openslr.org/resources/12/train-other-500.tar.gz":
    "ddb22f27f96ec163645d53215559df6aa36515f26e01dd70798188350adcb6d2"
}

def data_processing(data, data_type = "train"):

    if data_type == 'train':
        audio_transforms = nn.Sequential(
            torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
            torchaudio.transforms.TimeMasking(time_mask_param=35)
        )
    else:
        audio_transforms = torchaudio.transforms.MelSpectrogram()

    spectrograms = []
    labels = []
    words = []
    input_lengths = []
    label_lengths = []
    word_lengths = []
    
    for (waveform, melspectrum, _, utterance, letter_list, word_list) in data:
        spec = melspectrum.squeeze(0).transpose(0,1)
        spectrograms.append(spec)
        labels.append(letter_list)
        words.append(word_list)
        input_lengths.append(spec.shape[0])  
        label_lengths.append(len(letter_list))
        word_lengths.append(len(word_list))

    spectrograms = pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
    labels = pad_sequence(labels, batch_first=True)
    words = pad_sequence(words, batch_first=True)

    return spectrograms, labels, words, input_lengths, label_lengths, word_lengths

def load_librispeech_item(fileid: str,
                          path: str,
                          ext_wav: str,
                          ext_mel: str, 
                          ext_txt: str) -> Tuple[Tensor, Tensor, int, str, Tensor, Tensor]:
    speaker_id, chapter_id, utterance_id = fileid.split("-")
    
    file_text = speaker_id + "-" + chapter_id + ext_txt
    file_text = os.path.join(path, speaker_id, chapter_id, file_text)

    fileid_audio = speaker_id + "-" + chapter_id + "-" + utterance_id
    file_mel = fileid_audio + ext_mel
    file_mel = os.path.join(path, speaker_id, chapter_id, file_mel)

    file_wav = fileid_audio + ext_wav
    file_wav = os.path.join(path, speaker_id, chapter_id, file_wav)
    # Load audio
    waveform, sample_rate = torchaudio.load(file_wav)
    melspectrum = torch.load(file_mel)

    letter_tranform = LetterTransform()
    word_transform = WordTransform(PRE_TRAINED_MODEL, MAX_LEN, DO_LOWER_CASE)
    
    # Load text
    with open(file_text) as ft:
        for line in ft:
            fileid_text, utterance = line.strip().split(" ", 1)
            utterance_letter = torch.tensor(letter_tranform.text_to_int(utterance.lower()))
            utterance_word = torch.tensor(word_transform.text_to_int(utterance.lower()))
            if fileid_audio == fileid_text:
                break
        else:
            # Translation not found
            raise FileNotFoundError("Translation not found for " + fileid_audio)

    return (
        waveform,
        melspectrum,
        sample_rate,
        utterance,
        utterance_letter,
        utterance_word
    )


class LIBRISPEECH(Dataset):
    """
    Create a Dataset for LibriSpeech. Each item is a tuple of the form:
    waveform, sample_rate, utterance, speaker_id, chapter_id, utterance_id
    """

    _ext_txt = ".trans.txt"
    _ext_wav = ".flac"
    _ext_mel = ".pt"

    def __init__(self,
                 root: str,
                 url: str,
                 folder_in_archive: str = FOLDER_IN_ARCHIVE,
                 download: bool = False) -> None:

        if url in [
            "dev-clean",
            "dev-other",
            "test-clean",
            "test-other",
            "train-clean-100",
            "train-clean-360",
            "train-other-500",
        ]:

            ext_archive = ".tar.gz"
            base_url = "http://www.openslr.org/resources/12/"

            url = os.path.join(base_url, url + ext_archive)

        basename = os.path.basename(url)
        archive = os.path.join(root, basename)

        basename = basename.split(".")[0]
        folder_in_archive = os.path.join(folder_in_archive, basename)

        self._path = os.path.join(root, folder_in_archive)

        if download:
            if not os.path.isdir(self._path):
                if not os.path.isfile(archive):
                    checksum = _CHECKSUMS.get(url, None)
                    download_url(url, root, hash_value=checksum)
                extract_archive(archive)

            audio_transforms = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128)
            for root, dirs, files in os.walk(self._path):
                if len(files) != 0:
                    for file in files:
                        if file.split('.')[-1]==self._ext_wav.split('.')[-1]:
                            file_audio = os.path.join(root, file)
                            waveform, _ = torchaudio.load(file_audio)
                            spec = audio_transforms(waveform)
                            file_spec = os.path.join(root, file.split('.')[0]+ self._ext_wav)
                            torch.save(spec, file_spec)

        walker = walk_files(
            self._path, suffix=self._ext_mel, prefix=False, remove_suffix=True
        )
        self._walker = list(walker)

    def __getitem__(self, n: int) -> Tuple[Tensor, Tensor, int, str, Tensor, Tensor]:
        fileid = self._walker[n]
        return load_librispeech_item(fileid, self._path, self._ext_wav, self._ext_mel, self._ext_txt)

    def __len__(self) -> int:
        return len(self._walker)

    
    
if __name__ == "__main__":

    with open('../params.json') as json_file:
        params = json.load(json_file)
    data_params = params['data']
    train_params = params['train']
    print('hi')
    train_dataset = LIBRISPEECH("../../data/audio_data/", url=data_params['train_url'], download=False)

    print(train_dataset.__getitem__(10))

    use_cuda = torch.cuda.is_available()
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=train_params['batch_size'],
                              shuffle=True,
                              collate_fn=lambda x: data_processing(x, data_type='train'),
                              **kwargs)

    
    print(next(iter(train_loader)))
    # test_dataset = LIBRISPEECH("../../data/audio_data/", url=data_params['test_url'], download=False)
