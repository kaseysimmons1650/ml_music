# ml_music
## Machine Learning project that creates neural networks identifying genre and tempo from music.

# Research Question:
## What are some of the techniques that computers use to understand music and how do they work?

# Methodology:
- Created neural networks using python, tensorflow, and the tf.keras API
- Used librosa python package for handling of audio files
  
# Dataset: 
- Labeled dataset of over 100,000 songs.
- Free Music Archive Dataset - [FMA Dataset](https://arxiv.org/abs/1612.01840)

# Input Data:
- Based on the prior research studies dealing with audio music files and neural networks, the audio files were converted into images.
- The songs in this project were all converted into mel spectrograms, which is a visual representation of the frequencies of the audio file over time, and the decibel level is depicted by different colors on the image.

## Example Mel Spectrogram:
<img width="1162" alt="Screenshot 2024-10-23 at 10 07 57 AM" src="https://github.com/user-attachments/assets/a683e56f-dd0b-4541-a719-5c0dc87eae13">


## Example Mel Spectrograms comparing Tempo:  

### 52 bpm  
<img width="647" alt="52bpmTempo" src="https://github.com/user-attachments/assets/03ff1a9e-9a7f-489c-bdb5-9cec4dd5debc">  

### 191 bpm  
<img width="646" alt="191bpmTempo" src="https://github.com/user-attachments/assets/0dba7218-ed14-4d89-ba45-b471fb3f8dde">  



## Example Mel Spectrograms comparing Genre:  

### Old Time Historic  
<img width="646" alt="oldTimeHistoricGenre" src="https://github.com/user-attachments/assets/7a7636d0-81d9-410e-998d-046b6333c4e9">  

### Metal  
<img width="647" alt="MetalGenre" src="https://github.com/user-attachments/assets/1237bb98-3bf1-4a24-87e9-8451582d3af5">

## Genre Model
- Classification Model - classifying songs into 22 main genres
- Most successful model currently at 68% accuracy on unknown dataset

## Tempo Model
- Regression Model
- Most successful model currently has a root mean squared error of ~21

## Further Research
- Larger dataset
- Identify more musical elements
- Try various input data instead of Mel Spectrogram

[Presentation](powerpoint.pptx)


Moving from student gitlab account.
