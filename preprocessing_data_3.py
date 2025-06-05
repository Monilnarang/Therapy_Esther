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
            'content': [],
            'current_speaker': None
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
                    'content': [],
                    'current_speaker': None
                }

            # Add utterance to current group
            if role == 'therapist':
                current_group['content'].append(utterance['text'])
            else:
                # For clients, include the partner identifier only when speaker changes
                partner_name = self.speaker_mapping[speaker]
                if current_group['current_speaker'] != partner_name:
                    # New speaker, add identifier
                    current_group['content'].append(f"[{partner_name}]: {utterance['text']}")
                    current_group['current_speaker'] = partner_name
                else:
                    # Same speaker continuing, no identifier needed
                    current_group['content'].append(utterance['text'])

        # Don't forget the last group
        if current_group['role'] is not None:
            grouped.append(current_group)

        return grouped

    def create_dialogue_pairs(self, grouped_utterances: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Convert grouped utterances into dialogue pairs (human-gpt exchanges).
        """
        dialogue_pairs = []

        for i, group in enumerate(grouped_utterances):
            content = " ".join(group['content'])

            if group['role'] == 'client':
                # Client utterances become "human" messages
                dialogue_pairs.append({
                    "from": "human",
                    "value": content
                })
            else:
                # Therapist utterances become "gpt" messages
                dialogue_pairs.append({
                    "from": "gpt",
                    "value": content
                })

        return dialogue_pairs

    def create_sliding_window_conversations(self, dialogue_pairs: List[Dict[str, str]], window_size: int = 5) -> List[Dict]:
        """
        Create multiple conversations using a sliding window approach.
        Each conversation contains up to 'window_size' dialogue pairs.

        Args:
            dialogue_pairs: List of dialogue messages
            window_size: Number of dialogue pairs to include in each conversation
        """
        conversations = []

        # We need at least 2 messages (1 human + 1 gpt) to create a conversation
        if len(dialogue_pairs) < 2:
            return conversations

        # Find valid human-gpt pairs
        valid_pairs = []
        i = 0
        while i < len(dialogue_pairs) - 1:
            if (dialogue_pairs[i]['from'] == 'human' and
                    dialogue_pairs[i + 1]['from'] == 'gpt'):
                valid_pairs.append((i, i + 1))
                i += 2  # Move to next potential pair
            else:
                i += 1  # Move one step if not a valid pair

        # Create conversations with sliding window
        for pair_idx in range(len(valid_pairs)):
            conversation_messages = []

            # Calculate the start index for the window
            start_idx = max(0, pair_idx - window_size + 1)

            # Add all pairs in the window (from start_idx to current pair_idx inclusive)
            for window_pair_idx in range(start_idx, pair_idx + 1):
                human_idx, gpt_idx = valid_pairs[window_pair_idx]
                conversation_messages.extend([
                    dialogue_pairs[human_idx],
                    dialogue_pairs[gpt_idx]
                ])

            conversations.append({
                "conversations": conversation_messages
            })

        return conversations

    def convert_file(self, input_path: str, output_path: Optional[str] = None, window_size: int = 5) -> str:
        """
        Convert a conversation file to JSONL format with sliding window conversations.

        Args:
            input_path: Path to input conversation file
            output_path: Path to output JSONL file (auto-generated if None)
            window_size: Number of dialogue pairs to include in each conversation

        Returns:
            Path to the output file
        """
        if output_path is None:
            # Generate output filename
            input_file = Path(input_path)
            output_path = input_file.parent / f"{input_file.stem}.jsonl"

        # Parse conversation
        utterances = self.parse_conversation_file(input_path)

        # Group consecutive utterances
        grouped = self.group_consecutive_utterances(utterances)

        # Create dialogue pairs
        dialogue_pairs = self.create_dialogue_pairs(grouped)

        # Create sliding window conversations
        conversations = self.create_sliding_window_conversations(dialogue_pairs, window_size)

        # Write to JSONL file with each conversation as a separate line
        with open(output_path, 'w', encoding='utf-8') as f:
            for conversation in conversations:
                json.dump(conversation, f, ensure_ascii=False)
                f.write('\n')

        return str(output_path)

def main():
    # ================================
    # CONFIGURATION - EDIT THESE VALUES
    # ================================

    # Window size for sliding window conversations
    WINDOW_SIZE = 5  # Number of dialogue pairs to include in each conversation

    # Episodes to process - just specify the episode numbers
    EPISODES_TO_PROCESS = [1, 6, 7, 8, 9, 10]  # Will process Ep.1_final.txt, Ep.2_final.txt, Ep.10_final.txt

    # Speaker configurations for each episode (using just numbers)
    # Key should be the episode number
    SPEAKER_CONFIGS = {
        1: {
            "therapist_speakers": [6, 12, 13, 14],
            "partner_a_speakers": [2, 4, 7, 9, 10, 11, 16],
            "partner_b_speakers": [5, 8, 15],
            "excluded_speakers": [1, 3]
        },
        2: {
            "therapist_speakers": [3],
            "partner_a_speakers": [1, 5],
            "partner_b_speakers": [2, 4],
            "excluded_speakers": [6]
        },
        6: {
            "therapist_speakers": [6, 10, 11, 12, 13, 14, 31, 16, 19, 24],
            "partner_a_speakers": [4, 7, 15, 18, 23],
            "partner_b_speakers": [5, 8, 25, 17, 20, 25, 26],
            "excluded_speakers": [1, 2]
        },
        7: {
            "therapist_speakers": [3, 6, 11, 13, 14, 15, 16, 18, 20, 22, 24],
            "partner_a_speakers": [5, 9, 10, 12],
            "partner_b_speakers": [4, 7, 8, 19],
            "excluded_speakers": [1, 2]
        },
        8: {
            "therapist_speakers": [4],
            "partner_a_speakers": [5],
            "partner_b_speakers": [6],
            "excluded_speakers": [1, 2, 3]
        },
        9: {
            "therapist_speakers": [3, 8, 12, 11, 15],
            "partner_a_speakers": [4, 7, 14],
            "partner_b_speakers": [5, 6, 40],
            "excluded_speakers": [1, 2]
        },
        10: {
            "therapist_speakers": [5],
            "partner_a_speakers": [3, 7, 15],
            "partner_b_speakers": [4, 6, 8, 9],
            "excluded_speakers": [1, 2]
        },
        # Add more configurations as needed
        # 3: {
        #     "therapist_speakers": [X],
        #     "partner_a_speakers": [Y],
        #     "partner_b_speakers": [Z],
        #     "excluded_speakers": [A, B]
        # },
    }

    # ================================
    # END CONFIGURATION
    # ================================

    # Helper function to convert numbers to speaker strings
    def numbers_to_speakers(numbers):
        return [f"Speaker {num}" for num in numbers]

    # Process each episode
    total_processed = 0
    total_failed = 0

    for episode_num in EPISODES_TO_PROCESS:
        # Generate filename
        input_file = f"Ep.{episode_num}_final.txt"

        print(f"\n{'='*50}")
        print(f"Processing Episode {episode_num}: {input_file}")
        print(f"{'='*50}")

        # Check if speaker configuration exists for this episode
        if episode_num not in SPEAKER_CONFIGS:
            print(f"❌ Error: No speaker configuration found for Episode {episode_num}")
            print(f"Please add speaker configuration to SPEAKER_CONFIGS dictionary")
            total_failed += 1
            continue

        # Check if file exists
        if not os.path.exists(input_file):
            print(f"❌ Error: File {input_file} not found")
            total_failed += 1
            continue

        # Get speaker configuration for this episode and convert numbers to speaker strings
        config = SPEAKER_CONFIGS[episode_num]

        # Create converter and setup mappings
        converter = ConversationConverter()

        partner_mappings = {
            "Partner A": numbers_to_speakers(config["partner_a_speakers"]),
            "Partner B": numbers_to_speakers(config["partner_b_speakers"])
        }

        converter.setup_speaker_mapping(
            therapist_speakers=numbers_to_speakers(config["therapist_speakers"]),
            partner_mappings=partner_mappings,
            excluded_speakers=numbers_to_speakers(config["excluded_speakers"])
        )

        # Convert file
        try:
            output_path = converter.convert_file(input_file, None, WINDOW_SIZE)
            print(f"✅ Successfully converted conversation to: {output_path}")
            print(f"Using window size: {WINDOW_SIZE}")

            # Print some stats
            with open(output_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                total_conversations = len(lines)
                total_messages = 0
                human_messages = 0
                gpt_messages = 0

                for line in lines:
                    if line.strip():
                        data = json.loads(line)
                        conversations = data['conversations']
                        total_messages += len(conversations)
                        human_messages += len([c for c in conversations if c['from'] == 'human'])
                        gpt_messages += len([c for c in conversations if c['from'] == 'gpt'])

                print(f"Generated {total_conversations} conversations:")
                print(f"  - Total messages: {total_messages}")
                print(f"  - Human messages: {human_messages}")
                print(f"  - GPT messages: {gpt_messages}")
                print(f"  - Average messages per conversation: {total_messages/total_conversations:.1f}")

            total_processed += 1

        except Exception as e:
            print(f"❌ Error converting Episode {episode_num}: {e}")
            total_failed += 1

    # Summary
    print(f"\n{'='*50}")
    print(f"PROCESSING SUMMARY")
    print(f"{'='*50}")
    print(f"✅ Successfully processed: {total_processed} episodes")
    print(f"❌ Failed to process: {total_failed} episodes")
    print(f"📁 Total episodes attempted: {len(EPISODES_TO_PROCESS)}")

    return 1 if total_failed > 0 else 0

if __name__ == "__main__":
    exit(main())