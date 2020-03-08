#Tacotron 2 

Pytorch implementation of DeepMind's Tacotron-2 : [Natural TTS synthesis by conditioning Wavenet on MEL spectogram predictions](https://arxiv.org/pdf/1712.05884.pdf)
## Folder Structure
<pre>
└───tacotron2
    ├───content
    │   └───tacotron2
    │       └───filelists
    ├───filelists
    ├───outdir
    │   └───logdir
    ├───text
    │   ├───data_prepare
    │   └───__pycache__
    └───waveglow
    </pre>
## Setup

- Step **(0)**: Get your dataset; for persain lauguge the only open source dataset is [Mozilla common voice](https://voice.mozilla.org/en/datasets)
because mozilla audio is more than 211 h of audio we procced only small portion of it, convert to wave and remove files more than 10 seconds in length, you can see them in [filelists](/tacotron2/filelists).
- Step **(1)**: add your own test and train data parameters in ```filelists/```.
- Step **(2)**:  Install python requirements or build docker image 
    - Install python requirements: `pip install -r requirements.txt`
- Step **(3)**: Install cuda and pytorch 1.0 .
- Step **(4)**: Train the model using this command.
<pre>
<code>python train.py --output_directory='/content/tts-engine/gdrive/My Drive/outdir' --log_directory='/content/tts-engine/gdrive/My Drive/logdir'</code>
</pre>
- Step **(5)**: Synthesize audio using ``tts-engine/tacotron2/inference.ipynb``.


## Audio samples
I listed some of audio the model genarated you can listen them in [soundcloud](https://soundcloud.com/nima-moradi-78715897/sets/tacotron-2-audio-persian).
## Model
<p align="center">
 <img width="75%" src="images/arc.png">
</p>


The model described by the authors can be divided in two parts:
- Spectrogram prediction network
- Wavenet vocoder




