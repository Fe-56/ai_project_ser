import os
import subprocess
from pathlib import Path

class OpenSMILEFeatureExtractor:
    def __init__(self, opensmile_path='/Users/joel-tay/Desktop/opensmile-3.0.2-macos-armv8/bin/SMILExtract'):
        self.opensmile_path = opensmile_path
        # Path to the configuration file for feature extraction
        self.config_path = '/Users/joel-tay/Desktop/opensmile-3.0.2-macos-armv8/config/is09-13/IS13_ComParE.conf'
        
    def convert_mp4_to_wav(self, mp4_file, wav_file=None):
        """
        Convert MP4 file to WAV format using ffmpeg
        
        Args:
            mp4_file (str): Path to the input MP4 file
            wav_file (str, optional): Path to save the WAV file. If None, will use mp4_file name with .wav extension
            
        Returns:
            str: Path to the converted WAV file
        """
        if wav_file is None:
            wav_file = str(Path(mp4_file).with_suffix('.wav'))
            
        cmd = [
            'ffmpeg',
            '-i', mp4_file,
            '-vn',  # No video
            '-acodec', 'pcm_s16le',  # Convert to PCM
            '-ar', '16000',  # Sample rate
            '-ac', '2',  # Stereo
            '-y',  # Overwrite output file if exists
            wav_file
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"Successfully converted {mp4_file} to {wav_file}")
            return wav_file
        except subprocess.CalledProcessError as e:
            print(f"Error converting MP4 to WAV: {e}")
            return None
        
    def extract_features(self, audio_file, output_file=None):
        """
        Extract features from an audio file using OpenSMILE
        
        Args:
            audio_file (str): Path to the input audio file
            output_file (str, optional): Path to save the output features. If None, will use audio_file name with .csv extension
            
        Returns:
            str: Path to the output feature file
        """
        if output_file is None:
            output_file = str(Path(audio_file).with_suffix('.csv'))
            
        # If the input is an MP4 file, convert it to WAV first
        if audio_file.lower().endswith('.mp4'):
            wav_file = self.convert_mp4_to_wav(audio_file)
            if wav_file is None:
                return None
            audio_file = wav_file
            
        # Construct the OpenSMILE command
        cmd = [
            self.opensmile_path,
            '-C', self.config_path,
            '-I', audio_file,
            '-csvoutput', output_file,
            '-loglevel', "0"
        ]
        
        # Run OpenSMILE
        try:
            subprocess.run(cmd, check=True)
            print(f"Successfully extracted features to {output_file}")
            return output_file
        except subprocess.CalledProcessError as e:
            print(f"Error extracting features: {e}")
            return None
            
    def process_directory(self, input_dir, output_dir=None):
        """
        Process all audio files in a directory
        
        Args:
            input_dir (str): Directory containing audio files
            output_dir (str, optional): Directory to save feature files. If None, will use input_dir
            
        Returns:
            list: List of paths to the output feature files
        """
        if output_dir is None:
            output_dir = input_dir
            
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all audio files
        audio_extensions = ['.wav', '.mp3', '.ogg', '.flac', '.mp4']  # Added .mp4
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(Path(input_dir).glob(f'*{ext}'))
            
        feature_files = []
        for audio_file in audio_files:
            output_file = os.path.join(output_dir, f"{audio_file.stem}_features.csv")
            feature_file = self.extract_features(str(audio_file), output_file)
            if feature_file:
                feature_files.append(feature_file)
                
        return feature_files

# def main():
#     # Example usage
#     extractor = OpenSMILEFeatureExtractor()
    
#     # Process a single file
#     # extractor.extract_features('path/to/your/audio.wav')
    
#     # Process a directory
#     # feature_files = extractor.process_directory('path/to/your/audio/directory')
    
#     print("OpenSMILE feature extractor is ready to use!")
#     print("Please modify the main() function with your specific file or directory paths.")

# if __name__ == "__main__":
#     main() 