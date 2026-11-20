import os
import numpy as np
import librosa
import librosa.display
from collections import defaultdict, Counter
import pickle
import time
from typing import List, Dict, Tuple, Set
import scipy.ndimage
import skimage
from skimage.feature import peak_local_max
import gc  # For garbage collection



class AudioFingerprinter:
    """
    Audio fingerprinting system inspired by the Shazam algorithm (Wang, 2003).
    
    This system processes audio files to create and match fingerprints for audio identification.
    It implements a constellation map approach where spectral peaks are extracted from audio 
    spectrograms and paired to create robust fingerprints. The system can handle noisy and 
    distorted audio samples through quantization and combinatorial hashing techniques.
    
    The algorithm works in three main steps:
    1. Extract spectral peaks from audio spectrograms to form constellation maps
    2. Generate fingerprints by pairing peaks within a target zone
    3. Match query fingerprints against a database using a shifted inverse list approach
    """
    
    def __init__(self,
                 sr=22050, 
                 n_fft=1024,  # FFT window size, 1024 provides good balance between frequency resolution and time precision
                 hop_length=512,  # Controls time resolution; smaller values increase time resolution but decrease frequency resolution
                 peak_neighborhood_size=(15, 25),  # Wide frequency range, narrow time range for peak detection
                 peak_threshold=0.01,  # Threshold for peak detection; higher values are more selective
                 min_peak_amplitude=-90,  # Minimum amplitude threshold for peaks
                 fan_out=25,  # Maximum number of target points per anchor point
                 target_zone=(1, 25, 1, 50)):  # Target zone dimensions (min_time, max_time, min_freq, max_freq)
                
        """
        Initialize the AudioFingerprinter with configurable parameters.
        
        Args:
            sr (int): Sample rate for audio analysis, in Hz (default: 22050)
            n_fft (int): FFT window size for spectral analysis; affects frequency resolution
            hop_length (int): Hop length for STFT; controls time resolution of the spectrogram
            peak_neighborhood_size (tuple): Size of neighborhood (frequency, time) for peak finding
            peak_threshold (float): Relative threshold for peak detection (0.0-1.0)
            min_peak_amplitude (float): Minimum amplitude in dB for a peak to be considered
            fan_out (int): Maximum number of target points to pair with each anchor point
            target_zone (tuple): Target zone parameters (min_time_delta, max_time_delta, 
                                min_freq_delta, max_freq_delta)
        """

        # Audio processing parameters
        self.sr = sr  # Sample rate for analysis
        self.n_fft = n_fft  # FFT window size for spectrogram computation
        self.hop_length = hop_length  # Hop length for STFT (affects time resolution)
        
        # Peak detection parameters
        self.peak_neighborhood_size = peak_neighborhood_size  # Neighborhood size for peak finding
        self.peak_threshold = peak_threshold  # Threshold for peak detection
        self.min_peak_amplitude = min_peak_amplitude  # Minimum amplitude for a peak to be considered
        
        # Fingerprinting parameters
        self.fan_out = fan_out  # Maximum number of points to pair with each anchor point
        self.target_zone = target_zone  # Parameters defining the target zone for fingerprinting
        
        # Storage for fingerprint database
        self.fingerprint_db = {}  # Dictionary to store fingerprints
        self.time_offset_dict = defaultdict(list)  # Maps hash values to (file_id, time_offset) pairs
        self.file_id_map = {}  # Maps file IDs to filenames
    
    def _load_audio(self, audio_path) -> Tuple[np.ndarray, int]:
        """
        Load an audio file and resample it to the target sample rate.
        
        This method handles loading audio files (WAV or MP3) and resampling them
        to the sample rate specified in the class initialization. It also normalizes
        the audio to ensure consistent amplitude ranges.
        
        Args:
            audio_path (str): Path to the audio file to load
            
        Returns:
            Tuple[np.ndarray, int]: A tuple containing:
                - Audio time series as a numpy array
                - Sample rate
                
        Note:
            Returns an empty array and 0 for sample rate if loading fails
        """
        try:
            # Load the audio file and resample to target sample rate
            y, sr = librosa.load(audio_path, sr=self.sr, mono=True)
            
            # Normalize audio to ensure maximum dynamic range
            if len(y) > 0:
                y = librosa.util.normalize(y)
            return y, self.sr
        except Exception as e:
            print(f"Error loading audio file {audio_path}: {e}")
            return np.array([]), 0
    
    def _compute_spectrogram(self, audio_data) -> np.ndarray:
        """
        Compute the magnitude spectrogram of audio data in dB scale.
        
        Transforms the audio time series into a time-frequency representation
        using the Short-Time Fourier Transform (STFT). The resulting spectrogram
        is converted to dB scale for better dynamic range representation.
        
        Args:
            audio_data (np.ndarray): Audio time series (mono)
            
        Returns:
            np.ndarray: Magnitude spectrogram in dB scale, with shape (n_freq_bins, n_time_frames)
        """
        # Compute STFT
        stft = librosa.stft(audio_data, 
                            n_fft=self.n_fft, 
                            hop_length=self.hop_length, 
                            window='hann',)
        # Convert to magnitude spectrogram in dB scale with maximum normalization
        spectrogram = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
        return spectrogram

    def _find_peaks(self, spectrogram) -> List[Tuple[int, int]]:
        """
        Find spectral peaks in the spectrogram using skimage.feature.peak_local_max.
        
        This method identifies significant local maxima in the spectrogram that
        represent prominent time-frequency points. These peaks will form the
        constellation map for fingerprinting. The method applies frequency weighting
        to emphasize mid-range frequencies where most musical content typically exists.
        
        Args:
            spectrogram (np.ndarray): Magnitude spectrogram in dB scale
            
        Returns:
            List[Tuple[int, int]]: List of peak coordinates as (frequency_bin, time_frame) tuples
        """
        # Apply frequency weighting to focus on mid-range frequencies
        height, width = spectrogram.shape
        
        # Create frequency weights to emphasize mid-range frequencies
        # This is a psychoacoustic-inspired weighting to focus on perceptually important ranges
        freq_weights = np.linspace(0.2, 1.0, height//4)
        freq_weights = np.concatenate([
            freq_weights,                  # Lower frequencies (gradually increasing weight)
            np.ones(height-2*(height//4)), # Mid-range frequencies (full weight)
            freq_weights[::-1]             # Higher frequencies (gradually decreasing weight)
        ])
        assert len(freq_weights) == height

        # Apply weights to emphasize important frequency bands
        weighted_spec = spectrogram.copy()
        weighted_spec = weighted_spec * freq_weights[:, np.newaxis]

        # For peak_local_max, we need a non-negative image
        # Since spectrogram is in dB scale (with negative values), we need to adjust
        min_value = np.min(weighted_spec)
        if min_value < 0:
            adjusted_spec = weighted_spec - min_value  # Make all values non-negative
        else:
            adjusted_spec = weighted_spec

        # Use peak_local_max with carefully tuned parameters
        coordinates = peak_local_max(
            adjusted_spec,
            min_distance=5,        # Minimum distance between peaks to avoid clustering
            threshold_abs=None,    # Don't use absolute threshold initially
            threshold_rel=0.03,    # Relative threshold (3% of max value)
            exclude_border=False,  # Don't exclude border pixels
            num_peaks=10000        # Maximum number of peaks to extract
        )

        # Fallback to more lenient parameters if no peaks are found
        if len(coordinates) == 0:
            coordinates = peak_local_max(
                adjusted_spec,
                min_distance=5,
                threshold_rel=0.01,  # Lower threshold to capture more peaks
                num_peaks=20000
            )

        # Convert coordinates - peak_local_max returns (row, column) format
        # We convert to (frequency, time) for fingerprinting
        peaks = [(int(freq), int(time)) for freq, time in coordinates]

        # Filter peaks by amplitude threshold if specified
        if self.min_peak_amplitude is not None:
            peaks = [(f, t) for f, t in peaks if weighted_spec[f, t] > self.min_peak_amplitude]

        # Limit number of peaks to manage memory usage
        if len(peaks) > 20000:
            # Sort by peak amplitude and keep the strongest peaks
            peak_values = [weighted_spec[f, t] for f, t in peaks]
            sorted_indices = np.argsort(peak_values)[::-1][:20000]
            peaks = [peaks[i] for i in sorted_indices]

        # Debugging output
        print(f"  Found {len(peaks)} peaks using peak_local_max")

        return peaks

    def _generate_fingerprints(self, peaks, file_id) -> Dict:
        """
        Generate fingerprints from spectral peaks using the combinatorial pairing approach.
        
        This method implements the core fingerprinting logic following the approach described
        by Wang (2003). For each peak (anchor point), it identifies target points within a 
        specified target zone and creates hash values from these pairs.
        
        Args:
            peaks (List[Tuple[int, int]]): List of peak coordinates (frequency_bin, time_frame)
            file_id (int): Identifier for the current file
            
        Returns:
            Dict: Dictionary mapping hash values to lists of (file_id, time_offset) pairs
        """
        fingerprints = defaultdict(list)
        min_time_delta, max_time_delta, min_freq_delta, max_freq_delta = self.target_zone
        
        # Limit number of peaks to reduce memory usage
        peaks = peaks[:10000]  # Cap at 10,000 peaks

        # Sort peaks by time to ensure sequential processing and enable early termination
        peaks.sort(key=lambda p: p[1])
        
        for i, anchor in enumerate(peaks):
            f_anchor, t_anchor = anchor
            
            # Find points within the target zone (more efficient than full iteration)
            target_points = []
            # Look ahead up to 100 peaks - most will be filtered quickly by time constraints
            for j in range(i+1, min(i+100, len(peaks))):
                f_target, t_target = peaks[j]
                
                # Check if the point is within the target zone time bounds
                time_delta = t_target - t_anchor
                if time_delta < min_time_delta:
                    continue  # Too close in time, skip
                if time_delta > max_time_delta:
                    break  # We've gone beyond the maximum time delta (peaks are sorted by time)
                
                # Check frequency constraint
                freq_delta = abs(f_target - f_anchor)
                if min_freq_delta <= freq_delta <= max_freq_delta:
                    target_points.append((f_target, t_target))
                
                # Limit the number of target points per anchor to control fanout
                if len(target_points) >= self.fan_out:
                    break
            
            # Generate hashes for each anchor-target pair
            for f_target, t_target in target_points:
                time_delta = t_target - t_anchor
                
                # Create a hash from the frequency values and time delta
                hash_value = self._create_hash(f_anchor, f_target, time_delta)
                
                # Store the hash with file ID and time offset of the anchor point
                fingerprints[hash_value].append((file_id, t_anchor))
        
        return fingerprints
    
    def _create_hash(self, f1, f2, time_delta) -> int:
        """
        Create a robust hash from frequency and time values.
        
        This method creates a hash value from a pair of frequency points and their
        time difference. The values are quantized to make the hash more robust to
        small variations in frequency and timing that might occur in noisy or 
        degraded audio.
        
        Args:
            f1 (int): Frequency bin of anchor point
            f2 (int): Frequency bin of target point
            time_delta (int): Time difference between points (in frames)
            
        Returns:
            int: A hash value representing this particular feature pair
        """
        # Quantize frequency bins to make fingerprint more robust to small variations
        # Dividing by 3 creates broader frequency bins, making the hash more resistant to frequency shifts
        f1_quantized = int(f1 / 3)
        f2_quantized = int(f2 / 3)
        
        # Quantize time delta to handle tempo variations
        # This allows matching even if the tempo is slightly different
        time_delta_quantized = int(time_delta / 1.5)
        
        # Create a hash from the tuple of quantized values
        # Python's hash function creates an integer hash from the tuple
        hash_value = hash((f1_quantized, f2_quantized, time_delta_quantized))
        
        return hash_value
    
    def build_fingerprints(self, database_path, output_path):
        """
        Process audio files from a database directory and build a fingerprint database.
        
        This is the main entry point for fingerprint database creation. It processes
        each audio file in the specified directory, extracts fingerprints, and saves
        the resulting database to disk.
        
        Args:
            database_path (str): Path to directory containing audio files
            output_path (str): Path to save the fingerprint database
            
        Returns:
            None
        """
        start_time = time.time()
        print(f"Building fingerprints from {database_path}...")
        
        file_count = 0
        
        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        
        # Process each audio file in the database
        for file_id, filename in enumerate(sorted(os.listdir(database_path))):
            # Skip non-audio files
            if not (filename.endswith('.wav') or filename.endswith('.mp3')):
                continue
                
            filepath = os.path.join(database_path, filename)
            print(f"Processing {filename} ({file_id+1})...")
            
            # Load and process audio
            audio_data, sr = self._load_audio(filepath)
            if len(audio_data) == 0:
                continue  # Skip files that couldn't be loaded
                
            # Compute spectrogram
            spectrogram = self._compute_spectrogram(audio_data)
            
            # Find peaks to create the constellation map
            peaks = self._find_peaks(spectrogram)
            
            # Generate fingerprints from peaks
            file_fingerprints = self._generate_fingerprints(peaks, file_id)
            print(f" Generated {len(file_fingerprints)} fingerprints")

            # Add fingerprints to the database
            for hash_value, entries in file_fingerprints.items():
                for entry in entries:
                    self.time_offset_dict[hash_value].append(entry)
            
            # Map file ID to filename for later retrieval
            self.file_id_map[file_id] = filename
            file_count += 1
        
        # Save the fingerprint database to disk
        self.fingerprint_db = {
            'time_offset_dict': dict(self.time_offset_dict),
            'file_id_map': self.file_id_map
        }
        
        fingerprint_path = os.path.join(output_path, 'fingerprint_db.pkl')
        with open(fingerprint_path, 'wb') as f:
            pickle.dump(self.fingerprint_db, f)
        
        elapsed_time = time.time() - start_time
        print(f"Fingerprinting completed. Processed {file_count} files in {elapsed_time:.2f} seconds.")
        print(f"Fingerprint database saved to {fingerprint_path}")
        print(f"Total unique hashes: {len(self.time_offset_dict)}")

    def identify_audio(self, query_path, fingerprint_path, output_path):
        """
        Identify query audio files by matching them against the fingerprint database.
        
        This is the main entry point for audio identification. It processes each query
        audio file, extracts fingerprints, matches them against the database, and 
        saves the results to the specified output file.
        
        Args:
            query_path (str): Path to directory containing query audio files
            fingerprint_path (str): Path to fingerprint database
            output_path (str): Path to save identification results
            
        Returns:
            None
        """
        start_time = time.time()
        print(f"Identifying audio from {query_path}...")

        # Load fingerprint database from disk
        db_path = os.path.join(fingerprint_path, 'fingerprint_db.pkl')
        with open(db_path, 'rb') as f:
            self.fingerprint_db = pickle.load(f)
        
        # Initialize data structures from the loaded database
        self.time_offset_dict = defaultdict(list, self.fingerprint_db['time_offset_dict'])
        self.file_id_map = self.fingerprint_db['file_id_map']
        
        results = []
        
        # Count audio files to track progress
        total_files = len([f for f in os.listdir(query_path) 
                     if f.endswith('.wav') or f.endswith('.mp3')])
        
        # Process each query file
        for file_idx, filename in enumerate(sorted(os.listdir(query_path))):
            if not (filename.endswith('.wav') or filename.endswith('.mp3')):
                continue
                
            file_start_time = time.time()
            filepath = os.path.join(query_path, filename)
            print(f"Processing query {filename}... ({file_idx+1}/{total_files})")
        
            # Load and process audio
            audio_data, sr = self._load_audio(filepath)
            if len(audio_data) == 0:
                results.append((filename, []))
                continue
                
            # Compute spectrogram
            spectrogram = self._compute_spectrogram(audio_data)
            
            # Find peaks
            peaks = self._find_peaks(spectrogram)
            
            # Generate fingerprints (using -1 as a placeholder for query file ID)
            query_fingerprints = self._generate_fingerprints(peaks, -1)
            print(f"  Generated {len(query_fingerprints)} fingerprints")
            
            # Match fingerprints against the database
            try:
                matches = self._match_fingerprints(query_fingerprints)
                print(f"  Found {len(matches)} potential matches in {time.time() - file_start_time:.2f} seconds")
            except Exception as e:
                print(f"  Error in matching: {e}")
                matches = []
            
            # Store results
            results.append((filename, matches))
            
            # Save partial results periodically to avoid losing progress
            if (file_idx + 1) % 5 == 0 or file_idx == 0:
                with open(output_path + ".partial", 'w') as f:
                    for res_filename, res_matches in results:
                        match_files = [self.file_id_map[file_id] for file_id, _ in res_matches[:3]] if res_matches else []
                        while len(match_files) < 3:
                            match_files.append("")
                        f.write(f"{res_filename}\t{match_files[0]}\t{match_files[1]}\t{match_files[2]}\n")
            
            # Force garbage collection to manage memory usage
            gc.collect()
        
        # Write final results to output file in the required format
        with open(output_path, 'w') as f:
            for filename, matches in results:
                match_files = [self.file_id_map[file_id] for file_id, _ in matches[:3]] if matches else []
                # Pad with empty matches if needed (to ensure 3 columns)
                while len(match_files) < 3:
                    match_files.append("")
                f.write(f"{filename}\t{match_files[0]}\t{match_files[1]}\t{match_files[2]}\n")
        
        elapsed_time = time.time() - start_time
        print(f"Identification completed in {elapsed_time:.2f} seconds.")
        print(f"Results saved to {output_path}")

    # Shifted inverse list implementation
    def _match_fingerprints(self, query_fingerprints) -> List[Tuple[int, int]]:
        """
        Match query fingerprints against the database using the shifted inverse lists approach.
        
        This method implements the core matching algorithm described in Wang (2003). It creates
        a matching function that counts the number of matching fingerprints at each potential
        alignment between the query and database files.
        
        Args:
            query_fingerprints (Dict): Dictionary of fingerprints from query audio
            
        Returns:
            List[Tuple[int, int]]: List of (file_id, score) pairs sorted by score in descending order
        """
        # Create a matching function for efficient counting of alignment matches
        matching_function = defaultdict(int)

        print(f"  Matching {len(query_fingerprints)} fingerprints against database...")

        # For each query fingerprint hash
        for hash_value, query_entries in query_fingerprints.items():
            # Skip if hash not in database
            if hash_value not in self.time_offset_dict:
                continue

            db_entries = self.time_offset_dict[hash_value]

            # For each query time entry
            for _, query_time in query_entries:
                # Create shifted inverse lists
                for db_file_id, db_time in db_entries:
                    # Calculate shift index (time difference)
                    # This is the key insight from the Wang paper - we look at potential alignments
                    shift_index = db_time - query_time

                    # Increment count for this file and shift combination
                    # This efficiently counts all matches for a specific alignment
                    matching_function[(db_file_id, shift_index)] += 1

        # Group and count matches by file
        file_scores = defaultdict(int)

        # Find the best alignment for each file (highest count for a specific shift)
        for (file_id, shift), count in matching_function.items():
            if count > file_scores[file_id]:
                file_scores[file_id] = count

        # Convert to list of (file_id, score) pairs
        result = [(file_id, score) for file_id, score in file_scores.items()]

        # Sort by score in descending order
        result.sort(key=lambda x: x[1], reverse=True)

        return result[:10]  # Return top 10 matches


def fingerprintBuilder(database_path, fingerprint_path):
    """
    Build fingerprints for audio files in a database.
    
    This function creates an AudioFingerprinter instance and uses it to process
    all audio files in the specified database directory, generating fingerprints
    and saving them to the specified output path.
    
    Args:
        database_path (str): Path to directory containing audio files
        fingerprint_path (str): Path to save fingerprint database
    """
    fingerprinter = AudioFingerprinter()
    fingerprinter.build_fingerprints(database_path, fingerprint_path)


def audioIdentification(query_path, fingerprint_path, output_path):
    """
    Identify query audio files by matching against fingerprints.
    
    This function creates an AudioFingerprinter instance and uses it to process
    all audio files in the specified query directory, matching them against 
    the fingerprint database and saving the results to the specified output file.
    
    Args:
        query_path (str): Path to directory containing query audio files
        fingerprint_path (str): Path to fingerprint database
        output_path (str): Path to save identification results
    """
    fingerprinter = AudioFingerprinter()
    fingerprinter.identify_audio(query_path, fingerprint_path, output_path)


if __name__ == "__main__":
    import sys
    
    # Check if command line arguments are provided
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "fingerprintBuilder" and len(sys.argv) == 4:
            database_path = sys.argv[2]
            fingerprint_path = sys.argv[3]
            fingerprintBuilder(database_path, fingerprint_path)
        
        elif command == "audioIdentification" and len(sys.argv) == 5:
            query_path = sys.argv[2]
            fingerprint_path = sys.argv[3]
            output_path = sys.argv[4]
            audioIdentification(query_path, fingerprint_path, output_path)
        
        else:
            print("Invalid command or incorrect number of arguments")
            print("Usage:")
            print("  python fingerprint.py fingerprintBuilder /path/to/database/ /path/to/fingerprints/")
            print("  or")
            print("  python fingerprint.py audioIdentification /path/to/queryset/ /path/to/fingerprints/ /path/to/output.txt")
    
    # If no command line arguments, use the default behavior
    else:
        # Default paths
        database_path = "database_recordings"
        query_path = "query_recordings"
        fingerprint_path = "fingerprints"
        output_path = "output.txt"
        
        # Build fingerprints
        fingerprintBuilder(database_path, fingerprint_path)
        
        # Identify audio
        audioIdentification(query_path, fingerprint_path, output_path)

