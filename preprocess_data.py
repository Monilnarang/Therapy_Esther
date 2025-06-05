import json
import re
import os
from pathlib import Path
from typing import Dict, List, Set, Optional

class ConversationConverter:
    def __init__(self):
        self.speaker_mapping = {}
        self.excluded_speakers = set()
        self.therapist_speakers = set()

    def setup_speaker_mapping(self, therapist_speakers: List[str],
                              partner_mappings: Dict[str, List[str]],
                              excluded_speakers: List[str] = None):
        """
        Setup speaker mappings for the conversion.

        Args:
            therapist_speakers: List of speaker IDs that represent the therapist
            partner_mappings: Dict mapping partner names to list of speaker IDs
                             e.g., {"Partner A": ["Speaker 3"], "Partner B": ["Speaker 4"]}
            excluded_speakers: List of speaker IDs to exclude from output
        """
        self.therapist_speakers = set(therapist_speakers)
        self.excluded_speakers = set(excluded_speakers or [])

        # Create reverse mapping from speaker ID to role
        for partner_name, speakers in partner_mappings.items():
            for speaker in speakers:
                self.speaker_mapping[speaker] = partner_name

        for therapist in therapist_speakers:
            self.speaker_mapping[therapist] = "therapist"

    def parse_conversation_file(self, file_path: str) -> List[Dict[str, str]]:
        """
        Parse the conversation file and extract speaker utterances.

        Returns:
            List of dictionaries with 'speaker' and 'text' keys
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        # Pattern to match Speaker X: followed by text
        pattern = r'(Speaker \d+):\s*(.*?)(?=Speaker \d+:|$)'
        matches = re.findall(pattern, content, re.DOTALL)

        utterances = []
        for speaker, text in matches:
            # Clean up the text
            text = text.strip()
            if text:  # Only add non-empty utterances
                utterances.append({
                    'speaker': speaker,
                    'text': text
                })

        return utterances

    def group_consecutive_utterances(self, utterances: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Group consecutive utterances from the same role (therapist vs clients).
        """
        if not utterances:
            return []

        grouped = []
        current_group = {
            'role': None,
            'content': []
        }

        for utterance in utterances:
            speaker = utterance['speaker']

            # Skip excluded speakers
            if speaker in self.excluded_speakers:
                continue

            # Determine role
            if speaker in self.therapist_speakers:
                role = 'therapist'
            elif speaker in self.speaker_mapping:
                role = 'client'
            else:
                # Skip unknown speakers
                continue

            # If role changes or this is the first utterance, start new group
            if current_group['role'] != role:
                if current_group['role'] is not None:
                    grouped.append(current_group)
                current_group = {
                    'role': role,
                    'content': []
                }

            # Add utterance to current group
            if role == 'therapist':
                current_group['content'].append(utterance['text'])
            else:
                # For clients, include the partner identifier
                partner_name = self.speaker_mapping[speaker]
                current_group['content'].append(f"[{partner_name}]: {utterance['text']}")

        # Don't forget the last group
        if current_group['role'] is not None:
            grouped.append(current_group)

        return grouped

    def convert_to_training_format(self, grouped_utterances: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Convert grouped utterances to the training JSON format.
        """
        conversations = []

        # Process grouped utterances
        for i, group in enumerate(grouped_utterances):
            content = "\n".join(group['content'])

            if group['role'] == 'client':
                # Client utterances become "human" messages
                conversations.append({
                    "from": "human",
                    "value": content
                })
            else:
                # Therapist utterances become "gpt" messages
                conversations.append({
                    "from": "gpt",
                    "value": content
                })

        return conversations

    def convert_file(self, input_path: str, output_path: Optional[str] = None) -> str:
        """
        Convert a conversation file to JSON format.

        Returns:
            Path to the output file
        """
        if output_path is None:
            # Generate output filename
            input_file = Path(input_path)
            output_path = input_file.parent / f"{input_file.stem}_json.json"

        # Parse conversation
        utterances = self.parse_conversation_file(input_path)

        # Group consecutive utterances
        grouped = self.group_consecutive_utterances(utterances)

        # Convert to training format
        training_data = self.convert_to_training_format(grouped)

        # Write to JSON file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, indent=2, ensure_ascii=False)

        return str(output_path)

def main():
    # ================================
    # CONFIGURATION - EDIT THESE VALUES
    # ================================

    # Path to your conversation file
    INPUT_FILE = "Ep.1_final.txt"  # Change this to your file path

    # Output file path (optional - will auto-generate if None)
    OUTPUT_FILE = None  # e.g., "output.json" or None for auto-generation

    # Speaker configurations
    THERAPIST_SPEAKERS = ["Speaker 6", "Speaker 12", "Speaker 13", "Speaker 14"]  # List of speaker IDs for therapist

    PARTNER_A_SPEAKERS = ["Speaker 2", "Speaker 4", "Speaker 7", "Speaker 9", "Speaker 10", "Speaker 11", "Speaker 16"]  # List of speaker IDs for Partner A

    PARTNER_B_SPEAKERS = ["Speaker 5", "Speaker 8", "Speaker 15"]  # List of speaker IDs for Partner B

    EXCLUDED_SPEAKERS = ["Speaker 1", "Speaker 3"]  # List of speaker IDs to exclude from output

    # ================================
    # END CONFIGURATION
    # ================================

    # Create converter and setup mappings
    converter = ConversationConverter()

    partner_mappings = {
        "Partner A": PARTNER_A_SPEAKERS,
        "Partner B": PARTNER_B_SPEAKERS
    }

    converter.setup_speaker_mapping(
        therapist_speakers=THERAPIST_SPEAKERS,
        partner_mappings=partner_mappings,
        excluded_speakers=EXCLUDED_SPEAKERS
    )

    # Convert file
    try:
        output_path = converter.convert_file(INPUT_FILE, OUTPUT_FILE)
        print(f"Successfully converted conversation to: {output_path}")

        # Print some stats
        with open(output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print(f"Generated {len(data)} messages:")
            print(f"  - {len([c for c in data if c['from'] == 'human'])} human messages")
            print(f"  - {len([c for c in data if c['from'] == 'gpt'])} assistant messages")

    except Exception as e:
        print(f"Error converting file: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())