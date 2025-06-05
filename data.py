import pvfalcon
import whisper
import os
import time

# Directory containing audio files
audio_directory = "/Users/monilnarang/Downloads/"

# List of episode files to process
episode_files = [f"Ep.{i}.mp3" for i in range(2, 11)]  # Ep1.mp3 to Ep10.mp3

def process_audio_file(audio_file_path):
    """Process a single audio file for transcription and speaker diarization"""
    print(f"\n{'='*60}")
    print(f"Processing: {os.path.basename(audio_file_path)}")
    print(f"{'='*60}")
    
    # Generate whisper transcription filename
    audio_filename = os.path.basename(audio_file_path)  # Gets "Ep1.mp3"
    audio_name = os.path.splitext(audio_filename)[0]    # Gets "Ep1"
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Gets directory of this script
    whisper_file_path = os.path.join(script_dir, f"{audio_name}_wisper.txt")

    # Check if whisper transcription already exists
    if os.path.exists(whisper_file_path):
        print(f"Existing Whisper transcription found: {whisper_file_path}")
        print("Loading existing transcription...")
        
        # Parse the existing whisper file to extract segments
        transcript_segments = []
        with open(whisper_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and line.startswith('[') and '-->' in line:
                    # Parse timestamp and text from format: [00:00.000 --> 00:06.140]  Welcome to the...
                    timestamp_end = line.find(']')
                    if timestamp_end != -1:
                        timestamp_part = line[1:timestamp_end]  # Remove brackets
                        text_part = line[timestamp_end + 1:].strip()
                        
                        # Parse start and end times
                        if ' --> ' in timestamp_part:
                            start_str, end_str = timestamp_part.split(' --> ')
                            
                            def time_to_seconds(time_str):
                                # Convert MM:SS.mmm or HH:MM:SS.mmm to seconds
                                parts = time_str.split(':')
                                if len(parts) == 2:  # MM:SS.mmm
                                    minutes, seconds = parts
                                    return int(minutes) * 60 + float(seconds)
                                elif len(parts) == 3:  # HH:MM:SS.mmm
                                    hours, minutes, seconds = parts
                                    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
                                return 0
                            
                            start_time = time_to_seconds(start_str)
                            end_time = time_to_seconds(end_str)
                            
                            transcript_segments.append({
                                'start': start_time,
                                'end': end_time,
                                'text': text_part
                            })
        
        print(f"Loaded {len(transcript_segments)} segments from existing transcription")
    else:
        # Load Whisper model and transcribe
        print("No existing Whisper transcription found. Starting new transcription...")
        print("Loading model...")
        model = whisper.load_model("turbo")
        print(f"Starting transcription of {audio_file_path}...")
        print("This may take a while for large files...")
        start_time = time.time()
        result = model.transcribe(audio_file_path, verbose=True)  # verbose=True shows progress
        end_time = time.time()
        print(f"\nTranscription completed in {end_time - start_time:.2f} seconds")
        print(result["text"])
        transcript_segments = result["segments"]
        
        # Save whisper transcription in the requested format
        with open(whisper_file_path, 'w', encoding='utf-8') as f:
            for segment in transcript_segments:
                start_time = segment['start']
                end_time = segment['end']
                text = segment['text'].strip()
                
                # Format timestamps as MM:SS.mmm
                def seconds_to_time(seconds):
                    hours = int(seconds // 3600)
                    minutes = int((seconds % 3600) // 60)
                    secs = seconds % 60
                    if hours > 0:
                        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
                    else:
                        return f"{minutes:02d}:{secs:06.3f}"
                
                start_str = seconds_to_time(start_time)
                end_str = seconds_to_time(end_time)
                
                f.write(f"[{start_str} --> {end_str}]  {text}\n")
        
        print(f"Whisper transcription saved to: {whisper_file_path}")

    # Initialize Falcon for speaker diarization
    print("Starting speaker diarization...")
    falcon = pvfalcon.create("xi7fCnz3OYUorJkyUveMdu+fd4w49+HlOnhwryTRzHfCNdNNCGTHjw==")
    speaker_segments = falcon.process_file(audio_file_path)

    def segment_score(transcript_segment, speaker_segment):
        transcript_segment_start = transcript_segment["start"]
        transcript_segment_end = transcript_segment["end"]
        speaker_segment_start = speaker_segment.start_sec
        speaker_segment_end = speaker_segment.end_sec
        overlap = min(transcript_segment_end, speaker_segment_end) - max(transcript_segment_start, speaker_segment_start)
        overlap_ratio = overlap / (transcript_segment_end - transcript_segment_start)
        return overlap_ratio

    # Generate final output filename (with speaker labels)
    output_file_path = os.path.join(script_dir, f"{audio_name}_final.txt")

    # Process segments and write to file in real-time
    print("Combining transcription with speaker diarization...")
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for t_segment in transcript_segments:
            max_score = 0
            best_s_segment = None
            for s_segment in speaker_segments:
                score = segment_score(t_segment, s_segment)
                if score > max_score:
                    max_score = score
                    best_s_segment = s_segment
            
            # Handle case where no speaker segment was found
            if best_s_segment is not None:
                speaker_label = f"Speaker {best_s_segment.speaker_tag}"
            else:
                # Fallback: try to find the closest speaker segment by time
                t_start = t_segment["start"]
                t_end = t_segment["end"]
                t_mid = (t_start + t_end) / 2
                
                closest_segment = None
                min_distance = float('inf')
                
                for s_segment in speaker_segments:
                    s_mid = (s_segment.start_sec + s_segment.end_sec) / 2
                    distance = abs(t_mid - s_mid)
                    if distance < min_distance:
                        min_distance = distance
                        closest_segment = s_segment
                
                if closest_segment is not None:
                    speaker_label = f"Speaker {closest_segment.speaker_tag}"
                else:
                    speaker_label = "Speaker Unknown"
            
            line = f"{speaker_label}: {t_segment['text']}"
            print(line)  # Print to console
            f.write(line + '\n')  # Write to file immediately
            f.flush()  # Ensure it's written to disk immediately

    print(f"Final transcript with speakers saved to: {output_file_path}")
    return True

# Main processing loop
def main():
    print("Starting batch processing of episode files...")
    print(f"Looking for files in: {audio_directory}")
    
    processed_files = []
    skipped_files = []
    failed_files = []
    
    for episode_file in episode_files:
        audio_file_path = os.path.join(audio_directory, episode_file)
        
        # Check if file exists
        if not os.path.exists(audio_file_path):
            print(f"\nSkipping {episode_file} - File not found")
            skipped_files.append(episode_file)
            continue
        
        try:
            success = process_audio_file(audio_file_path)
            if success:
                processed_files.append(episode_file)
                print(f"✅ Successfully processed {episode_file}")
        except Exception as e:
            print(f"❌ Error processing {episode_file}: {str(e)}")
            failed_files.append(episode_file)
    
    # Summary
    print(f"\n{'='*60}")
    print("BATCH PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Successfully processed: {len(processed_files)} files")
    for file in processed_files:
        print(f"   - {file}")
    
    if skipped_files:
        print(f"\n⏭️  Skipped (not found): {len(skipped_files)} files")
        for file in skipped_files:
            print(f"   - {file}")
    
    if failed_files:
        print(f"\n❌ Failed: {len(failed_files)} files")
        for file in failed_files:
            print(f"   - {file}")
    
    print(f"\nTotal files attempted: {len(episode_files)}")
    print("Batch processing complete!")

if __name__ == "__main__":
    main()