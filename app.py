import os
import torch
import streamlit as st
import torchaudio
import numpy as np

from mask_cyclegan_vc.model import Generator
from mask_cyclegan_vc.utils import decode_melspectrogram

class VoiceConverterApp:
    def __init__(self, checkpoints_dir):
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Preset speakers and data paths
        self.speakers = {
            'romario': './dataset_preprocessed/train/famous/famous_normalized.pickle',
            'any_voice': './dataset_preprocessed/train/mozilla/mozilla_normalized.pickle',
        }
        
        
        # Normalization stats
        self.norm_stats = {
            'romario': np.load('./dataset_preprocessed/train/famous/famous_norm_stat.npz'),
            'any_voice': np.load('./dataset_preprocessed/train/mozilla/mozilla_norm_stat.npz')
        }
        
        # Initialize MelGAN vocoder
        self.vocoder = torch.hub.load('descriptinc/melgan-neurips', 'load_melgan')
        
        # Generators
        self.generator_A2B = Generator().to(self.device)
        self.generator_B2A = Generator().to(self.device)
        
        # Load checkpoints
        self._load_checkpoint(checkpoints_dir, 'generator_A2B', self.generator_A2B)
        self._load_checkpoint(checkpoints_dir, 'generator_B2A', self.generator_B2A)
        
        # Set to evaluation mode
        self.generator_A2B.eval()
        self.generator_B2A.eval()
    
    def _load_checkpoint(self, checkpoints_dir, model_name, model):
        """Safely load checkpoint for a model"""
        try:
            # Find the checkpoint file
            checkpoint_files = [f for f in os.listdir(checkpoints_dir) 
                                if f.endswith('.pth.tar') and model_name in f]
            
            if not checkpoint_files:
                st.warning(f"No checkpoint found for {model_name}")
                return
            
            # Use the most recent checkpoint
            latest_checkpoint = sorted(checkpoint_files)[-1]
            checkpoint_path = os.path.join(checkpoints_dir, latest_checkpoint)
            
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state'])
            
            st.info(f"Loaded {model_name} from {latest_checkpoint}")
        
        except Exception as e:
            st.error(f"Error loading checkpoint for {model_name}: {e}")
    
    def convert_voice(self, input_audio, source_speaker, target_speaker):
        # Load audio
        waveform, sample_rate = torchaudio.load(input_audio)
        
        # Compute mel spectrogram
        mel_spec = self.vocoder(waveform)
        
        # Normalize input
        mel_spec = (mel_spec - self.norm_stats[source_speaker]['mean']) / self.norm_stats[source_speaker]['std']
        
        # Choose appropriate generator
        generator = (self.generator_A2B if source_speaker == 'any_voice' and target_speaker == 'romario' 
                     else self.generator_B2A)
        
        # Convert voice
        with torch.no_grad():
            converted_mel = generator(mel_spec.to(self.device), torch.ones_like(mel_spec).to(self.device))
        
        # Denormalize and decode
        converted_mel = converted_mel.cpu()
        converted_wav = decode_melspectrogram(
            self.vocoder, 
            converted_mel[0], 
            self.norm_stats[target_speaker]['mean'], 
            self.norm_stats[target_speaker]['std']
        )
        
        # Save converted audio
        output_path = f'converted_{source_speaker}_to_{target_speaker}.wav'
        torchaudio.save(output_path, converted_wav, sample_rate=22050)
        
        return output_path

def main():
    st.title('Voice Conversion with MaskCycleGAN')
    
    # Checkpoints directory
    checkpoints_dir = './results/mask_cyclegan_vc_MOZILLA_FAMOUS/ckpt'
    
    try:
        converter = VoiceConverterApp(checkpoints_dir)
    except Exception as e:
        st.error(f"Failed to initialize Voice Converter: {e}")
        return
    
    # Sidebar for inputs
    st.sidebar.header('Voice Conversion Settings')
    input_audio = st.sidebar.file_uploader("Upload Audio File", type=['wav'])
    
    source_speaker = st.sidebar.selectbox(
        'Source Speaker', 
        list(converter.speakers.keys())
    )
    
    target_speaker = st.sidebar.selectbox(
        'Target Speaker', 
        [spk for spk in converter.speakers.keys() if spk != source_speaker]
    )
    
    if st.sidebar.button('Convert Voice') and input_audio is not None:
        try:
            # Temporary save uploaded file
            with open('temp_input.wav', 'wb') as f:
                f.write(input_audio.getvalue())
            
            # Convert voice
            output_path = converter.convert_voice('temp_input.wav', source_speaker, target_speaker)
            
            # Display original and converted audio
            st.audio(input_audio, format='audio/wav', start_time=0)
            st.audio(output_path, format='audio/wav', start_time=0)
            
            # Provide download button
            with open(output_path, 'rb') as f:
                st.download_button(
                    label='Download Converted Audio',
                    data=f,
                    file_name=output_path,
                    mime='audio/wav'
                )
        
        except Exception as e:
            st.error(f"Error in voice conversion: {e}")

if __name__ == "__main__":
    main()