import numpy as np
import spacy
nlp = spacy.load("en_core_web_sm")
from typing import Optional, Tuple, Callable
from transformers import PreTrainedTokenizerBase
from sklearn.linear_model import LinearRegression

# ---- uniform_lower_diagonal ------------------------------------------------
def uniform_lower_diagonal(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    n = len(toks.input_ids[0])
    out = np.full((n, n), 1e-4)
    for i in range(n):
        for j in range(n):
            if j <= i:
                out[i, j] = 1.0
    out += 1e-4
    out = out / out.sum(axis=1, keepdims=True)
    return "Uniform Lower Diagonal Attention Pattern", out

# Layer 0, Head 0
from typing import Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase

# Define the function to predict coreference patterns

def coreference_resolution(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    # Use the tokenizer to convert the sentence to tensor format
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])  # Length of the sequence
    out = np.zeros((len_seq, len_seq))
    # Split the sentence into words for alignment
    words = sentence.split()
    # A naive coreference resolution attempt
    # when subject pronouns refer back to recent named entities or prior subjects
    subjects = ['he', 'she', 'they']
    subject_indices = []
    entity_indices = []

    for index, word in enumerate(words):
        if word.lower() in subjects:
            subject_indices.append(index)
        elif word.lower().istitle():  # If it's a proper noun
            entity_indices.append(index)
        # Connect subjects to entities through coreference

    for s_index in subject_indices:
        for e_index in entity_indices:
            # Attention pattern in both directions for coreference
            out[s_index, e_index] = 1
            out[e_index, s_index] = 1
    # Normalize attention
    out += 1e-4  # Avoid any division issues
    out = out / out.sum(axis=1, keepdims=True)
    # Ensure no row is zero sum
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0
    return "Coreference Resolution", out

# Layer 0, Head 2
import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple
import spacy

# Load the spaCy model
en_core = spacy.load('en_core_web_sm')

def coreference_resolution(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenization using spaCy for coreference resolution
    doc = en_core(sentence)
    word_to_token_map = {}
    idx = 1
    for token in doc:
        word_to_token_map[token.text.lower()] = idx
        idx += 1

    # Coreference using entity recognition
    for token in doc:
        noun_chunk = token.text.lower()
        if noun_chunk in word_to_token_map:
            token_index = word_to_token_map[noun_chunk]
            for ent in doc.ents:
                if ent.text.lower() in word_to_token_map:
                    ent_index = word_to_token_map[ent.text.lower()]
                    out[token_index, ent_index] = 1
                    out[ent_index, token_index] = 1

    # Ensure no row is all zeros by adding attention to [SEP]
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize the attention matrix
    out += 1e-4 # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)

    return "Coreference Resolution Pattern", out

# Layer 0, Head 9
from transformers import PreTrainedTokenizerBase
import numpy as np
from typing import Tuple

# Function to identify the coreference resolution pattern

def coreference_resolution(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    # Tokenize the sentence
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Convert token IDs to tokens
    tokens = tokenizer.convert_ids_to_tokens(toks.input_ids[0])

    # Dictionary to keep track of the latest occurrence of each token
    last_occurrence = {}

    # Iterate over the tokens
    for i, token in enumerate(tokens):
        token_low = token.lower()
        # Check if the token has been seen before (indicating a coreference)
        if token_low in last_occurrence:
            previous_index = last_occurrence[token_low]
            out[previous_index, i] = 1  # Mark the attention from last occurrence to the current
            out[i, previous_index] = 1  # Bi-directional attention
        # Update the last occurrence of the token
        last_occurrence[token_low] = i

    # Ensure no row in the attention matrix is sum zero
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize the output matrix
    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize by row

    return "Coreference Resolution Pattern", out

# Layer 1, Head 7
from typing import Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase

def punctuation_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    tokenized_sentence = tokenizer.convert_ids_to_tokens(toks.input_ids[0])

    # Assign high attention to punctuations and sentence boundaries
    for i, token in enumerate(tokenized_sentence):
        if token in {'.', ',', ';', ':', "?", "!", '[SEP]', '[CLS]'}:
            out[i, i] = 1  # Each punctuation attends to itself heavily

    # Ensure each row has attention summed to 1
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0  # Attend to [SEP] if no specific attention
    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize
    return "Punctuation and Sentence Boundary Emphasis Pattern", out

# Layer 1, Head 8
import numpy as np
from transformers import PreTrainedTokenizerBase

def coreference_resolution(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    words = sentence.split()
    coreference_pairs = {}

    # Example: Manually define pronouns for sentences
    pronouns = {'he', 'she', 'it', 'they', 'them', 'her', 'his', "'s"}
    candidate_referents = set()

    for i, word in enumerate(words):
        if word in pronouns:
            for referent in candidate_referents:
                out[i + 1, referent + 1] = 1
                out[referent + 1, i + 1] = 1
        else:
            candidate_referents.add(i)

    # Ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize

    return "Coreference Resolution Pattern", out

# Layer 4, Head 3
import numpy as np
from transformers import PreTrainedTokenizerBase
import spacy

nlp = spacy.load('en_core_web_sm')

# def pronoun_reference(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
#     toks = tokenizer([sentence], return_tensors="pt")
#     len_seq = len(toks.input_ids[0])
#     out = np.zeros((len_seq, len_seq))
#     # Convert token IDs to text, and get alignment
#     tokens = tokenizer.convert_ids_to_tokens(toks.input_ids[0])
#     token_to_spacy = {i: tokens[i] for i in range(len(tokens))}
#     # SpaCy processing
#     doc = nlp(sentence)
#     spacy_to_token = {token.text: i+1 for i, token in enumerate(doc) if token.text in token_to_spacy.values()}
#     # Link pronouns with their referents
#     for i, token in enumerate(doc):
#         if token.pos_ == 'PRON':  # If the token is a pronoun
#             for possible_ref in doc:
#                 if possible_ref.lemma_ == token.lemma_ and possible_ref != token:
#                     spacy_ref_idx = possible_ref.i + 1
#                     spacy_token_idx = token.i + 1
#                     if spacy_token_idx in spacy_to_token.keys() and spacy_ref_idx in spacy_to_token.keys():
#                         out[spacy_token_idx, spacy_ref_idx] = 1
#                         out[spacy_ref_idx, spacy_token_idx] = 1
#     # Ensure each row has some attention
#     for row in range(len_seq):
#         if out[row].sum() == 0:
#             out[row, -1] = 1.0
#     out += 1e-4  # Avoid division by zero
#     out = out / out.sum(axis=1, keepdims=True)  # Normalize
#     return "Pronoun Reference Pattern", out

import numpy as np
from transformers import PreTrainedTokenizerBase
import spacy

# Assuming nlp is loaded elsewhere
# nlp = spacy.load("en_core_web_sm")

def pronoun_reference(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    # 1. Process with SpaCy for linguistic analysis
    doc = nlp(sentence)
    
    # 2. Tokenize for Transformer
    toks = tokenizer([sentence], return_tensors="pt")
    input_ids = toks.input_ids[0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    len_seq = len(tokens)
    
    # Create the attention matrix
    out = np.zeros((len_seq, len_seq))

    # 3. Create a mapping from Transformer Token Index to SpaCy Token Index
    # This aligns the subwords to words
    tok_to_spacy = {}
    
    # Convert token IDs to strings and clean them for comparison
    # (Handling BERT/GPT style tokenizers)
    clean_tokens = [t.replace("##", "").replace("Ġ", "") for t in tokens]
    
    for spacy_token in doc:
        # Find which transformer tokens make up this spacy token
        for i, token in enumerate(clean_tokens):
            if token and spacy_token.text.startswith(token) and len(token) > 0:
                # Basic heuristic mapping - can be improved with offset mapping
                if i not in tok_to_spacy:
                    tok_to_spacy[i] = spacy_token
                    
    # 4. Link pronouns with their referents
    for i, token in enumerate(doc):
        if token.pos_ == 'PRON':
            for possible_ref in doc:
                if possible_ref.lemma_ == token.lemma_ and possible_ref != token:
                    # Find transformer tokens corresponding to these spacy tokens
                    pronoun_indices = [idx for idx, s_tok in tok_to_spacy.items() if s_tok == token]
                    referent_indices = [idx for idx, s_tok in tok_to_spacy.items() if s_tok == possible_ref]
                    
                    # Fill matrix
                    for p_idx in pronoun_indices:
                        for r_idx in referent_indices:
                            out[p_idx, r_idx] = 1.0
                            out[r_idx, p_idx] = 1.0

    # 5. Ensure valid attention (row sum > 0)
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, row] = 1.0 # Self-attention if no reference found
            
    # Normalize to make it a valid probability distribution
    out = out / out.sum(axis=1, keepdims=True)
    
    return "Pronoun Reference Pattern", out

# Layer 5, Head 1
import numpy as np
from transformers import PreTrainedTokenizerBase

# Hypothesis: This head is responsible for focusing on coordination and logical linking words like 'and', 'with', and 'for', which tend to create connections between different parts of a sentence and maintain semantic flow.
def coordination_attention(sentence: str, tokenizer: PreTrainedTokenizerBase):
    toks = tokenizer([sentence], return_tensors='pt')
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokens that typically involve coordination conjunctions or logical connectors
    coordination_tokens = {"and", "with", "for"}

    # Decoding back to words to compare
    token_words = tokenizer.convert_ids_to_tokens(toks.input_ids[0])

    # Build a coordination mapping
    coordination_indices = []
    for idx, token in enumerate(token_words):
        if token in coordination_tokens:
            coordination_indices.append(idx)

    # Apply attention pattern
    for idx in coordination_indices:
        # Apply strong attention from the coordination word to its immediate neighbors
        if idx > 0:
            out[idx, idx-1] = 1.0  # Previous token
        if idx < len_seq - 1:
            out[idx, idx+1] = 1.0  # Next token
        # Reverse attention to coordination word itself for emphasis from linked tokens
        if idx > 0:
            out[idx-1, idx] = 1.0
        if idx < len_seq - 1:
            out[idx+1, idx] = 1.0

    # Ensure no row in out is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize to make sure attention distributions sum to 1 at each explanatory token row
    out += 1e-4  # Avoid division by zero issues
    out = out / out.sum(axis=1, keepdims=True)

    return "Coordination and Logical Linking Pattern", out

# Layer 5, Head 6
import numpy as np
from transformers import PreTrainedTokenizerBase

def complement_adjunct_relationship(sentence: str, tokenizer: PreTrainedTokenizerBase) -> "Tuple[str, np.ndarray]":
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Define hardcoded relationships based on observed patterns
    # These are generalized rules observed across the sample data
    complement_pairs = [['to', 'play'], ['to', 'difficult'], ['with', 'play'],
                        ['with', 'share'], ['for', 'difficult']]

    def find_pairs(splitted_sentence, pair):
        indices_1 = [i for i, word in enumerate(splitted_sentence) if word == pair[0]]
        indices_2 = [i for i, word in enumerate(splitted_sentence) if word == pair[1]]
        return indices_1, indices_2

    words = sentence.lower().split()

    for pair in complement_pairs:
        indices_1, indices_2 = find_pairs(words, pair)
        for i1 in indices_1:
            for i2 in indices_2:
                out[i1+1, i2+1] = 1  # Adjust for CLS token
                out[i2+1, i1+1] = 1

    # Ensure every token has an attention weight
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    return "Complement/Adjunct Relationship", out

# Layer 6, Head 3
import numpy as np
from transformers import PreTrainedTokenizerBase

def emphasize_verbs_and_objects(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    words = tokenizer.convert_ids_to_tokens(toks.input_ids[0])

    # keywords to focus that are associated with action
    action_keywords = ["found", "difficult", "share", "went", "said", "sew", "smiled", "sharing", "fixing"]

    for i, word in enumerate(words):
        # Assuming the word is an action if it matches the keywords
        if any(kw in word for kw in action_keywords):
            for j in range(len_seq):
                if i != j:
                    out[i, j] = 1 / (abs(i - j) + 1)  # Closer tokens get more attention

    for row in range(len_seq):
        # Ensure no row is all zeros
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize by rows

    return "Emphasizing Action Verbs and Their Objects", out

# Layer 7, Head 11
import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple

def conjunction_resolution(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenize and identify the positions
    tokens = tokenizer.convert_ids_to_tokens(toks.input_ids[0])
    conjunctions = {"and", "but", "or", "so", "because", "while", "although"}  # Common conjunctions

    # Look for conjunctions and assign attention patterns
    for i, token in enumerate(tokens):
        if token in conjunctions:
            # Assign reciprocal attention to next and previous elements
            if i > 1:  # Avoid indexing errors
                prev_index = i - 1
                next_index = i + 1
                # Ensure within bounds
                if next_index < len_seq:
                    out[i, prev_index] = 1
                    out[i, next_index] = 1

    for row in range(len_seq):  # Ensure no row is all zeros
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    out += 1e-4  # Add a small constant to avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize by row

    return "Conjunction Resolution", out

# Layer 8, Head 0
from typing import Tuple
import numpy as np
import spacy
from transformers import PreTrainedTokenizerBase

# Load spaCy's English tokenizer, tagger, parser, NER and word vectors
nlp = spacy.load("en_core_web_sm")

# Define function

def coreference_resolution(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenize sentence with spacy
    doc = nlp(sentence)
    token2position = {token.text: i for i, token in enumerate(doc)}

    # Hypothetical mapping from the sentence to predicted attention
    for token in doc:
        # If a token is a pronoun, match it with a noun in its possible antecedent span
        if token.pos_ == 'PRON':
            for earlier_token in doc:
                if (earlier_token.pos_ in {'NOUN', 'PROPN'} and earlier_token.i < token.i
                   and earlier_token.text in token2position):
                    # Create attention between pronoun and its possible antecedent noun
                    out[token2position[earlier_token.text]+1, token2position[token.text]+1] = 1
                    out[token2position[token.text]+1, token2position[earlier_token.text]+1] = 1

    # Ensure CLS and SEP tokens get some attention for stabilization
    out[0, 0] = 1
    out[len_seq-1, len_seq-1] = 1

    # Ensure no row is all zeros for subsequent operations
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    return "Co-reference Resolution Pattern", out

# Layer 8, Head 10
import numpy as np
from transformers import PreTrainedTokenizerBase

def coreference_resolution(sentence: str, tokenizer: PreTrainedTokenizerBase):
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Convert token IDs to string tokens with offsets
    word_ids = toks.word_ids(batch_index=0)
    assert word_ids is not None  # to ensure sentence does not come entirely from special tokens

    # A dictionary to match pronouns with their referents
    coref_map = {
        'he': 'someone',
        'him': 'someone',
        'his': 'someone',
        'she': 'someone',
        'her': 'someone',
        'it': 'object',
        'they': 'group',
        'them': 'group',
        'their': 'group'
    }

    # SpaCy is suitable for more complex NLP tasks
    pronouns_set = set(coref_map.keys())

    # Find token indices for special words
    for index, token_id in enumerate(toks.input_ids[0]):
        token = tokenizer.decode([token_id]).strip()
        if token.lower() in pronouns_set:
            # Mark attention from pronoun to their coreferent using the coref_map
            referent_word = coref_map[token.lower()]

            # Loop to find referent token indices
            for referent_index, word in enumerate(word_ids):
                if word is not None:
                    token_at_word = tokenizer.convert_ids_to_tokens(toks.input_ids[0][referent_index])
                    if referent_word in token_at_word.lower():
                        out[index, referent_index] = 1

    # Adding self-attention for [CLS] and [SEP]
    out[0, 0] = 1
    out[-1, -1] = 1

    # Ensure no token has no attention by defaulting to last token
    for row_index in range(len_seq):
        if out[row_index].sum() == 0:
            out[row_index, -1] = 1.0

    # Normalize by row
    out = out / np.clip(out.sum(axis=1, keepdims=True), a_min=1e-9, a_max=None)

    return "Coreference Resolution", out

# Layer 8, Head 2
import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

# Assuming sentence refers to text and tok_sent refers to tokenized sentence
# Example assuming using a pretrained tokenizer like BERT's

def conjunction_resolution(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    words = tokenizer.convert_ids_to_tokens(toks.input_ids[0])
    conjunctions = {"and", "or", "but", ","}

    conjunction_indices = [i for i, word in enumerate(words) if word.lower() in conjunctions]

    for idx in conjunction_indices:
        if idx > 0 and idx < len_seq - 1:
            # Attention to the previous and next word of the conjunction
            out[idx, idx - 1] = 1
            out[idx, idx + 1] = 1
        if idx > 1:
            # Have some backwards reference to earlier conjunctions or main verbs
            out[idx, idx - 2] = 0.5
        if idx < len_seq - 2:
            # Have some forward reference to subsequent conjunctions or key elements
            out[idx, idx + 2] = 0.5

    # Ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, row] = 1.0

    return "Coordination and Conjunction Resolution", out

# Layer 8, Head 5
from typing import Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase

def coord_and_verb_dependency(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Define indices for special tokens
    cls_idx = 0
    sep_idx = len_seq - 1

    # These hypothetical rules aim to capture verbs and their related coordination compounds
    # Note: The indices are hypothetical and align with overall style convention
    # Use dictionary to map token indices to words for indexed operations
    tok_map = {i: tok for i, tok in enumerate(toks.input_ids[0])}

    # Simplified interpretation of observed attention patterns:
    # Key pattern: and|verb and between coordinated structures
    tokens = tokenizer.convert_ids_to_tokens(toks.input_ids[0])
    verbs = [i for i, token in enumerate(tokens) if token.startswith('##')]
    coord_conjs = [i for i, token in enumerate(tokens) if token in ['and', 'but', 'or']]

    # Examples of coordination triggering
    for coord in coord_conjs:
        # Assume verbs tend to 'command' coordination especially when 'and' is present
        for verb in verbs:
            if verb < coord < sep_idx:  # hypothetical logic for selective tuning
                out[coord, verb] = 1
                out[verb, coord] = 1

    # Handling basic normalization assuming no row is zero 
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, sep_idx] = 1.0

    out += 1e-4  # Ensures no division by zero
    out = out / out.sum(axis=1, keepdims=True)

    return "Coordination and Verb Dependency", out

# Layer 9, Head 6
import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase
import spacy

nlp = spacy.load("en_core_web_sm")  # Load spaCy model for English

def coreference_resolution(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Using spaCy to find potential coreferences by linking pronouns and possessives to their referents
    doc = nlp(sentence)
    token_mapping = {i: tok.i for i, tok in enumerate(doc)}  # Map tokenizer indices to spaCy indices

    referent_map = {}

    # Identify pronoun->noun mappings
    for tok in doc:
        if tok.dep_ in {"nsubj", "dobj", "poss"} and tok.pos_ == "PRON":
            for possible_antecedent in tok.ancestors:
                if possible_antecedent.pos_ in {"NOUN", "PROPN"}:
                    referent_map[tok.i] = possible_antecedent.i
                    break

    # Fill the attention matrix based on referent_map
    for pronoun_idx, referent_idx in referent_map.items():
        attention_token_idx = list(token_mapping.keys())[list(token_mapping.values()).index(pronoun_idx)]
        reference_token_idx = list(token_mapping.keys())[list(token_mapping.values()).index(referent_idx)]
        out[attention_token_idx, reference_token_idx] = 1
        out[reference_token_idx, attention_token_idx] = 1

    # Ensure no row is all zeros by adding self-attention to each token
    np.fill_diagonal(out, 1)
    out = out / out.sum(axis=1, keepdims=True)  # Normalize attention weights

    return "Coreference Resolution Pattern", out

# Layer 0, Head 8
import numpy as np
from transformers import PreTrainedTokenizerBase

# Assuming you have the examples in a format and tokenizer specified

def coreference_resolution(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Split the sentence into tokens
    tokens = tokenizer.convert_ids_to_tokens(toks.input_ids[0])

    # Very basic heuristic: assume the first noun or pronoun is the antecedent
    antecedent_index = None
    current_attention_strength = 100

    # Initialize a helper list with assumed parts of speech
    pronouns = {"I", "me", "you", "he", "him", "she", "her", "it", "we", "us", "they", "them"}
    noun_like_tokens = pronouns.union({"Lily", "needle", "mom", "button", "shirt"})

    for i, tok in enumerate(tokens):
        # Identify the antecedent as the first noun-like or pronoun token
        if antecedent_index is None and tok in noun_like_tokens:
            antecedent_index = i
        else:
            # If a noun or pronoun follows, assume it's referring back to the antecedent
            if tok in noun_like_tokens or tok in pronouns:
                if antecedent_index is not None:
                    out[i, antecedent_index] = current_attention_strength / 100.0

        # Decrease attention strength for subsequent tokens as a simplistic model of resolution feature strength fade
        current_attention_strength -= 5

    # Ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    return "Coreference Resolution Pattern", out



# Layer 10, Head 0
import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

def initial_token_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    # Set the first content token (ignoring CLS token) to have maximum attention weight on others
    for i in range(1, len_seq - 1):
        out[1, i] = 1  # Assuming the first significant token has major attention
    # Normalize the attention matrix by row to ensure valid attention distribution
    row_sums = np.sum(out, axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.  # Avoid division by zero
    out = out / row_sums
    return "Initial Token Attention Pattern", out

# Layer 10, Head 1
import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple

# Define the coreference resolution function

def coreference_resolution(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Splitting the input sentence to match against tokenized words
    words = sentence.split()

    # Creating a mapping between token indexes and words
    token_to_word = {}
    current_word_idx = 0
    for token_idx in range(1, len_seq - 1):  # Exclude cls and eos
        current_token = tokenizer.convert_ids_to_tokens(toks.input_ids[0][token_idx].item())
        # Check if it's a special token or part of the current word
        if not current_token.startswith('\u0120'):
            # It's a subword token, align it to the current word
            token_to_word[token_idx] = words[current_word_idx]
        else:
            # It's the start of a new word
            current_word_idx += 1
            token_to_word[token_idx] = words[current_word_idx]

    # Simulating coreference by making pronouns point to their antecedents
    # This is a simplification for illustration purposes

    # Dummy antecedent tracking, usually you'd need more sophisticated NLP parsing
    antecedent = None
    pronouns = {'she', 'her', 'it', 'they'}

    for token_idx in token_to_word:
        word = token_to_word[token_idx].lower()
        if word in pronouns:
            if antecedent is not None:
                # Point to antecedent
                out[token_idx, antecedent] = 1
                out[antecedent, token_idx] = 1
        elif word not in pronouns:
            antecedent = token_idx

    # No token should have zero attention
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize

    return "Coreference Resolution Pattern", out

# Layer 1, Head 0
from typing import Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase
import spacy

nlp = spacy.load('en_core_web_sm')

def coreference_resolution(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    words = sentence.split()
    doc = nlp(" ".join(words))

    # Dictionary to map token positions between spaCy and tokenizer
    idx_map = {}
    word_idx = 0
    for i, token_span in enumerate(doc.sents):
        for token in token_span:
            while word_idx < len(toks.input_ids[0]) and toks.word_ids()[word_idx] is None:
                word_idx += 1
            if word_idx < len(toks.input_ids[0]):
                idx_map[token.i] = word_idx
                word_idx += 1

    # Resolution of pronouns to the nearest noun antecedent
    for entity in doc.ents:
        if entity.end in idx_map:
            for token in doc:
                if token.is_pronoun and token.head in entity:
                    if token.i in idx_map and entity.start in idx_map:
                        out[idx_map[token.i], idx_map[entity.start]] = 1

    # Ensure at least one attention value per row
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    return "Coreference Resolution Pattern", out

# Layer 1, Head 1
import numpy as np
from transformers import PreTrainedTokenizerBase

def sentence_start_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> np.ndarray:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Get the first token (usually the start of the sentence)
    out[0, :] = 1.0  # Assume the start of sentence token attends to all tokens
    out[0, 0] = 0.0  # Remove self-attention for start token (often special token)

    for row in range(len_seq):
        # Ensure no row is all zeros
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize to avoid division by zero issues and allow comparing to actual attention
    out = out / out.sum(axis=1, keepdims=True)

    return "Sentence Start Attention Pattern", out

# Layer 1, Head 2
from typing import Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase

def initial_token_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors='pt')
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # The first non-special token after CLS often dominates attention
    main_tokens = [i for i in range(len_seq) if toks.input_ids[0][i] != tokenizer.cls_token_id and toks.input_ids[0][i] != tokenizer.sep_token_id]
    if main_tokens:
        first_non_special = main_tokens[0]
        out[first_non_special, :] = 1

    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0  # attend to EOS as a default

    # Return pattern name and predicted matrix
    return "Initial Token Dominance Attention", out

# Layer 1, Head 3
import numpy as np
from transformers import PreTrainedTokenizerBase


def initial_token_dominance(sentence: str, tokenizer: PreTrainedTokenizerBase):
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # The initial token (after encoded special tokens) seems to have strong attention
    # Try to establish this pattern for any given sentence
    if len_seq > 1:  # Ensure there's at least one token other than special tokens
        dominant_token_index = 1  # Assuming 0 is [CLS] (beginning token) for tokenizers like BERT
        for i in range(1, len_seq):  # Start from 1 to skip [CLS]
            out[dominant_token_index, i] = 1

    for row in range(len_seq):  # Ensure no row is all zeros
        if out[row].sum() == 0:  # If there are any all-zero rows (even rare special cases handle this way)
            out[row, -1] = 1.0  # Instead of normalizing zeros, just guarantee some attention

    out += 1e-4  # To avoid division by zero during normalization
    out = out / out.sum(axis=1, keepdims=True)  # Normalize so rows sum to 1

    return "Initial Token Dominance", out

# Layer 1, Head 7
import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple

def initial_token_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    # Set attention from the first token to all others, mimicking the observed pattern
    for i in range(1, len_seq):
        out[0, i] = 1
    # Ensure every token has some attention by setting small attention on the last token
    # if no attention is otherwise assigned
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0
    out += 1e-4  # Avoid division by zero by smoothing
    out = out / out.sum(axis=1, keepdims=True)  # Normalize to simulate attention distribution
    return "Focus on Sentence Initial Token Pattern", out

# Layer 3, Head 1
from typing import Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase

def sentence_start_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Predicting that sentences have a strong initial token attention pattern
    out[0, :] = 1.0 # CLS token attends to the entire sentence
    out[1, :] = 1.0 # First token attends to the entire sentence

    # Normalize attention by row sum
    row_sums = out.sum(axis=1, keepdims=True)
    # Avoid division by zero
    row_sums[row_sums == 0] = 1
    out = out / row_sums

    return "Sentence Start Attention Pattern", out

# Layer 3, Head 5
import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase


def sentence_start_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # The model seems to focus on the start of the sentence and words closely following it
    # Assign higher attention to the first couple tokens
    for i in range(1, min(5, len_seq-1)):  # Ensure we don't exceed sequence bounds
        out[0, i] = 1.0  # Attention from the first token to other nearby tokens

    for row in range(len_seq):  # Ensure no row is all zeros
        if out[row].sum() == 0:
            out[row, -1] = 1.0  # Assign some minimal attention where needed

    out = out / out.sum(axis=1, keepdims=True)  # Normalize
    return "Sentence Start Attention Pattern", out

# Layer 4, Head 2
from transformers import PreTrainedTokenizerBase
import numpy as np
from typing import Tuple

def initial_token_dominance(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    # Ensure that the first token attends to all tokens with gradually decreasing attention
    for col in range(len_seq):
        out[0, col] = 1 / (col + 1)
    # Normalize to sum to 1 across the row
    out[0] = out[0] / np.sum(out[0])
    # Ensure CLS and EOS have self attention
    out[0, 0] = 1 # CLS token attention to self
    out[-1, -1] = 1 # EOS token attention to self
    # Ensure there is at least one non-zero value for every token via self-attention
    for row in range(1, len_seq-1):
        out[row, row] = 1.0
    return "Initial Token Dominance Pattern", out

# Layer 4, Head 3
import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple

# This function predicts the attention pattern for Layer 4, Head 3 in GPT-2.
# It assumes the pattern observed is to focus on the sentence subject or pronoun.
def pronoun_reference(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors='pt')
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # The focus is often on pronouns or the subject of the sentence, represented by the first non-special token
    # We assume it projects to the corresponding tokens with decreasing strength from the first pronoun/subject
    # Typically, pronouns/subjects appear early in English sentences, usually near the beginning
    for i in range(1, len_seq):  # Start from 1 to avoid [CLS] token
        out[i][1] = 1.0  # Assuming the subject or pronoun from position 1 gets major attention

    for row in range(len_seq): # Ensure no row is all zeros
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize the matrix so attention scores sum to 1
    out = out / out.sum(axis=1, keepdims=True)
    return 'Pronoun/Subject Reference Pattern', out

# Layer 4, Head 8
import numpy as np
from transformers import PreTrainedTokenizerBase
def sentence_initial_dominance(sentence: str, tokenizer: PreTrainedTokenizerBase):
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    # Assuming CLS token attention to itself and the sentence start token 
    init_token_index = 1 
    out[0, init_token_index] = 1
    out[init_token_index] = 1  # Initial token pays highest attention to all
    # Ensure rest of tokens have at least attention to one end token
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0  # EOS token
    out += 1e-4  # Prevent division by zero during normalization
    out = out / out.sum(axis=1, keepdims=True)  # Normalize
    return "Sentence Initial Word Dominance", out

# Layer 4, Head 9
import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

def sentence_start_emphasis(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Emphasize attention towards the initial tokens of clauses/sentences
    for i in range(1, len_seq):
        out[i, 1] = 1  # Assume the first actual token after CLS gets highest attention

    # Ensure attention distribution consistency
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0  # default to padding token if no attention is placed
        out += 1e-4  # Avoid division by zero
        out = out / out.sum(axis=1, keepdims=True)  # Normalize by rows

    return "Sentence Start Emphasizing Pattern", out

# Layer 5, Head 1
from transformers import PreTrainedTokenizerBase
import numpy as np
from typing import Tuple


def first_token_emphasis(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Emphasize attention from all tokens to the first actual token
    for i in range(1, len_seq-1):  # Ignoring special tokens usually at 0 and len_seq-1
        out[i, 1] = 1.0  # Attention directed towards the first token after the special [CLS]

    # Special token [CLS] self-attends
    out[0, 0] = 1.0

    # Ensure every row sums to 1 by normalizing
    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)

    return "First Token Emphasis", out

# Layer 6, Head 10
from transformers import PreTrainedTokenizerBase
from typing import Tuple
import numpy as np

def coreference_resolution(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Token alignments for key pronouns or names
    key_tokens = [
        "One",  # Acts as a coreference node or a stand-in for following elements
        "She",
        "Lily",
        "Can",
        "Her",
        "Together",
        "It",
        "After",
        "They"
    ]
    words = sentence.split()

    # Create a basic alignment map from tokenizer/token IDs back to words
    word_to_index = {word: i for i, word in enumerate(words)}
    token_indices = [i for i, word in enumerate(words) if word in key_tokens]

    # Simulate attention pattern by making key tokens focus more on their own vicinity and previous key tokens
    for i in token_indices:
        out[i, i] = 1  # Strong self-attention
        if i > 0:
            out[i, i - 1] = 0.5  # Some attention to the previous token for continuity
        if i < len_seq - 1:
            out[i, i + 1] = 0.5  # Some attention to the next token for continuity

    # Normalize to ensure at least some attention output for all tokens
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0  # If no attention, focus weakly on the EOS token

    out = out / out.sum(axis=1, keepdims=True)  # Normalize by row

    return "Coreference Resolution Pattern", out

# Layer 6, Head 6
import numpy as np
from transformers import PreTrainedTokenizerBase

def pronoun_reference(sentence: str, tokenizer: PreTrainedTokenizerBase):
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    token_to_index = {v: k for k, v in enumerate(toks.word_ids(0))}
    words = sentence.split()

    pronouns = {"he", "she", "it", "they", "we", "I", "you", "her", "him", "us", "them"}

    for index, word in enumerate(words):
        word_lower = word.lower().strip('.,!?"')
        if word_lower in pronouns:
            out[index + 1, index + 1] = 1  # Self-attention for pronouns
            # Account for the salience by making the pronoun have attention with earlier important nouns
            max_attention_length = 5  # Hypothetical constraint for how far we look back
            reverse_index = max(0, index - max_attention_length)
            for j in reversed(range(reverse_index, index)):
                if j+1 in token_to_index and words[j][0].isupper():  # Checking if it is a noun
                    out[index + 1, j + 1] = 1  

    # Ensure no rows are fully zeros (except for special tokens which might get filled later)
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    return "Pronoun Reference and Salience Pattern", out

# Layer 6, Head 9

def coreference_resolution(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    import numpy as np
    from scipy.special import softmax
    from typing import Tuple
    from transformers import PreTrainedTokenizerBase
    
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    tokens = toks.input_ids[0].tolist()

    # Identify potential coreference targets - typically proper nouns or pronouns
    coref_targets = [i for i, t in enumerate(tokens)]  # Treating all tokens as potential targets

    # Simulate coreference resolution by assuming first pronoun resolves to first potential target
    # Simplifying assumption as this is a complicated task
    for i, target in enumerate(coref_targets[:-1]):
        next_target = coref_targets[i + 1]
        for j in range(target, next_target):
            out[j, target] = 1

    # Normalize attention weights using softmax
    out = softmax(out, axis=1)

    # Ensure every row attends to something
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    return "Coreference Resolution Pattern", out

# Layer 7, Head 10
from transformers import PreTrainedTokenizerBase
import numpy as np


def coreference_resolution(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple[str, np.ndarray]:
    # Tokenize the sentence
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenize using spaCy and make sure tokens match
    import spacy
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(sentence)

    # Create a mapping from spaCy tokens to BERT tokens
    tok_alignment = {i: toks.input_ids[0][i].item() for i in range(len_seq - 1)}

    # Given example data where coref is likely between subjects and their associated verbs/nouns
    coref_map = {}
    for token in doc:
        if token.pos_ in {"PRON", "PROPN", "NOUN"}:
            coref_map[token.i] = token

    # Assigning attentions among references found
    for i in coref_map:
        ref_id = tok_alignment.get(i + 1, None)
        if ref_id:
            for j in range(len_seq):
                out[i + 1, j] = 1.0  # the +1 accounts for the [CLS] token

    # Normalize the output matrix
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0  # Ensuring no row is without attention focus
        out += 1e-4
        out /= out.sum(axis=1, keepdims=True)  # Normalize rows

    return "Coreference Resolution Pattern", out

# Layer 7, Head 1
from typing import Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase
import spacy

nlp = spacy.load('en_core_web_sm')

def coreference_resolution(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Use spaCy to parse the sentence
    doc = nlp(sentence)
    token_to_index = {tok.idx: i+1 for i, tok in enumerate(doc) if tok.text.strip()}

    # Iterate over tokens and activate coreference patterns
    for i, token in enumerate(doc):
        if token.pos_ == "PRON" or token.dep_ == "nsubj":
            # Find previous tokens possibly acting as antecedents
            for j in range(i):
                if doc[j].pos_ in ["NOUN", "PROPN"]:
                    if j in token_to_index and i in token_to_index:
                        out[token_to_index[j], token_to_index[i]] = 1
                        out[token_to_index[i], token_to_index[j]] = 1

    # Assign self attention for cls ([0, 0]) and eos ([-1, -1]), and default attention to [eos, ...]
    out[0, 0] = 1
    out[-1, -1] = 1
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    return "Coreference Resolution Pattern", out

# Layer 7, Head 2
import numpy as np
from transformers import PreTrainedTokenizerBase

def sentence_start_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> str:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # The first non-padding token in the sequence is given complete focus
    focus_index = 1
    out[focus_index] = 1

    # Ensure no row is all zeros by adding slight attention to eos
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize the attention matrix
    out = out / out.sum(axis=1, keepdims=True)

    return "Sentence Start Token Focus", out

# Layer 8, Head 0
import numpy as np
from transformers import PreTrainedTokenizerBase

def initial_token_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Initial token gets the majority of the attention in each sentence
    out[0, 0] = 1.0  # Attention to itself (if the first token)
    for i in range(1, len_seq - 1):
        out[1, i] = 0.1 + 0.1 * (len_seq - i) / len_seq

    # Ensure no row is all zeros by attending to the <eos> token
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    return "Focus on Sentence Initial Tokens", out

# Layer 8, Head 5
import numpy as np
from transformers import PreTrainedTokenizerBase

def sentence_start_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # The assumption is the head attends primarily to the first token of the sentence.
    for i in range(1, len_seq-1):
        out[i, 1] = 1  # Attend to the start of the sentence

    # Ensure no token is not attended by at least adding a small value to the last token
    out[:, -1] = 1e-4  # Little attention to end token to ensure normalization doesn't break.

    # Normalize each row to sum to 1
    out /= out.sum(axis=1, keepdims=True)

    return "Sentence Start Attention Pattern", out

# Layer 8, Head 7
import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple


def sentence_start_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Set attention pattern
    # Every token attends primarily to the first content token
    for i in range(len_seq):
        out[i, 1] = 1  # Attend to the first non-special token (often a specific content start)

    # Ensure no row is all zeros by giving attention to the [CLS] or starting token
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, 0] = 1.0

    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize row-wise

    return "Sentence Start Focus", out

# Layer 9, Head 11
import numpy as np
from transformers import PreTrainedTokenizerBase


def initial_token_attention(sentence: str, tokenizer: PreTrainedTokenizerBase):
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # The initial token (after CLS) seems to receive the majority of attention.
    for i in range(1, len_seq):
        out[i, 1] = 1.0

    # Normalize each row to ensure the sum is 1 (standard practice in attention heads)
    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)

    return "Initial Token Attention Pattern", out



# Layer 9, Head 6
from typing import Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase

# The hypothesis for Layer 9, Head 6 indicates attention on tokens, particularly pronouns, maintaining sentence-level coherence.
# Predicted attention reflects the focus starting from a main token (usually a pronoun at the beginning of the sentence or a focal noun)
# spread across all tokens.

def pronoun_reference(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    token_nouns = {'they', 'she', 'he', 'it', 'that', 'this', 'these', 'those', 'we', 'us', 'you', 'them', 'me', 'i', 'him', 'her', 'one', 'someone', 'everyone', 'no one', 'anyone', 'thing'}

    # Find main token to focus attention
    focus_token_index = 1  # Default focus on the first meaningful token (after CLS)
    for i, token_text in enumerate(tokenizer.convert_ids_to_tokens(toks.input_ids[0])):
        if i == 0: 
            continue
        if any(pronoun.lower().lstrip().startswith(token_text.strip().lower()) for pronoun in token_nouns):
            focus_token_index = i
            break

    # The main token (typically a pronoun at start) spreads attention across the sentence.
    out[focus_token_index, 1:-1] = 1.0  # Spread attention over all tokens excluding CLS and possibly EOS

    # Ensure any sentence's attention is balanced across.
    for row in range(1, len_seq - 1):  # Exclude CLS, EOS
        if out[row].sum() == 0:
            out[row, -1] = 1.0  # Assign any remaining empty row attention to EOS

    return "Sentence-level Pronoun Reference Pattern", out

# Refinement adverbial_modulation
def adverbial_modulation(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors='pt')
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Process sentence with spaCy to align tokens
    words = sentence.split()
    doc = nlp(' '.join(words))

    # Create a mapping to handle tokenization alignment differences
    spaCy_to_tokenizer = {}
    tokenizer_index = 1
    for word in doc:
        for _ in range(len(tokenizer.tokenize(word.text))):
            spaCy_to_tokenizer[word.i] = tokenizer_index
            tokenizer_index += 1

    # Check for adverbs and link them to their governing verbs, if applicable
    for token in doc:
        if token.pos_ == 'ADV':
            adverb_index = spaCy_to_tokenizer.get(token.i)
            for child in token.head.children:
                if child.dep_ in {'advcl', 'conj', 'xcomp', 'adjunct'}:
                    head_index = spaCy_to_tokenizer.get(child.i)
                    if adverb_index and head_index:
                        out[adverb_index, head_index] = 1  # Direct attention
                        out[head_index, adverb_index] = 1  # Ensure bidirectional

    # Ensure [CLS] and [SEP] are attended
    out[0, 0] = 1
    out[-1, 0] = 1

    # Normalize the matrix by rows
    out += 1e-4
    out = out / out.sum(axis=1, keepdims=True)

    return 'Adverbial Modulation Pattern', out


# Refinement appositive_phrase_attention
def appositive_phrase_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    # Tokenize the sentence and identify appositive punctuation marks like ',' or '()'
    tokens = sentence.split()
    # Assuming tokens and spaCy entities have been matched
    comma_indices = [i for i, tok in enumerate(tokens) if tok == ',']

    # If there are appositive phrases, link the head of the sentence part around commas
    if len(comma_indices) > 1:
        for i in range(len(comma_indices) - 1):
            start = comma_indices[i]
            end = comma_indices[i+1]
            # Make those segments attend strongly to themselves
            out[start+1:end+1, start+1:end+1] = 1

    # Ensure [CLS] and [SEP] have some attention
    out[0, 0] = 1
    out[-1, 0] = 1

    # Normalize the attention matrix
    out += 1e-4  # Add some smoothing to avoid pure zeros
    out = out / out.sum(axis=1, keepdims=True)

    return "Appositive Phrase Attention", out


# Refinement cls_attention
def cls_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    out[:, 0] = 1
    return "CLS Pattern", out


# Refinement compound_element_association
def compound_element_association(sentence: str, tokenizer: PreTrainedTokenizerBase):
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Split sentence into words assuming tokenizer gives word_ids
    word_tokens = toks.word_ids(batch_index=0)
    word_to_tokens = {}

    # Map each word to its corresponding tokens
    for idx, word_id in enumerate(word_tokens):
        if word_id is None:
            continue
        if word_id not in word_to_tokens:
            word_to_tokens[word_id] = []
        word_to_tokens[word_id].append(idx)

    # Identifying compound elements within the tokens
    # Look for pattern where subparts are related within compounds
    for word_id, token_indices in word_to_tokens.items():
        if len(token_indices) > 1:  # Indicates a compound element
            for token_i in token_indices:
                for token_j in token_indices:
                    if token_i != token_j:
                        out[token_i, token_j] = 1

    # Normalize attention
    row_sums = out.sum(axis=1, keepdims=True)
    np.divide(out, row_sums, out=np.zeros_like(out, dtype=float), where=row_sums!=0)

    # Including [CLS] and [SEP] importance
    out[0, 0] = 1
    out[-1, 0] = 1

    return "Compound Element Association", out


# Refinement compound_modifier_attention
def compound_modifier_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    nlp = spacy.load("en_core_web_sm")
    doc = nlp(sentence)
    token_map = {token.i: idx for idx, token in enumerate(doc)}

    for token in doc:
        if token.dep_ in {"amod", "compound"}:  # Modifier relationships
            head_idx = token_map.get(token.head.i, -1)
            if 1 <= head_idx < len_seq:
                token_idx = token_map.get(token.i, -1)
                out[token_idx + 1, head_idx + 1] = 1  # Applying the modifier-to-head attention
                out[head_idx + 1, token_idx + 1] = 1  # Symmetrically

    out[0, 0] = 1  # CLS attends to CLS
    out[-1, 0] = 1  # SEP attends to CLS

    # Normalize the attention pattern across each token's row
    out += 1e-4  # Adding small value for numerical stability
    out = out / out.sum(axis=1, keepdims=True)
    return "Compound Formation - Modifier Head Attention", out


# Refinement compound_word_attention_pattern
def compound_word_attention_pattern(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors='pt')
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    words = sentence.split()

    # Simple heuristic: treat sequences with two or more hash symbols as compound words
    for idx, token in enumerate(words):
        if '##' in token:
            # Find the starting index of the compound part
            compound_idx = idx - token.count('##')
            out[compound_idx, idx + 1] = 1  # shift by 1 to account for [CLS]
            out[idx + 1, compound_idx] = 1

    out[0, 0] = 1  # Attention to [CLS]
    out[-1, 0] = 1  # Attention to [SEP]

    # Normalize attention matrix to sum to 1 over each row
    out += 1e-4  # tiny value to avoid zero division errors
    out = out / out.sum(axis=1, keepdims=True)
    return 'Compound Word Attention', out


# Refinement conjunction_based_grouping
def conjunction_based_grouping(sentence: str, tokenizer: PreTrainedTokenizerBase):
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    doc = nlp(sentence)
    word_ids = {token.idx: i for i, token in enumerate(doc)}

    # Identifying conjunctions and their respective grouped nouns/phrases
    conjunction_indices = [i for i, token in enumerate(doc) if token.pos_ == 'CCONJ']
    for conj_index in conjunction_indices:
        left_tree = doc[conj_index].lefts  # Tokens on the left of the conjunction
        right_tree = doc[conj_index].rights  # Tokens on the right of the conjunction

        left_indices = [word_ids[token.idx] for token in left_tree if token.idx in word_ids]
        right_indices = [word_ids[token.idx] for token in right_tree if token.idx in word_ids]

        # Create links within each group and across the conjunction
        for i in left_indices:
            for j in right_indices:
                out[i+1, j+1] = 1
                out[j+1, i+1] = 1
        for i in left_indices:
            for j in left_indices:
                if i != j:
                    out[i+1, j+1] = 1
        for i in right_indices:
            for j in right_indices:
                if i != j:
                    out[i+1, j+1] = 1

    # Ensure [CLS] attends to itself and [SEP] to [CLS]
    out[0, 0] = 1
    out[-1, 0] = 1

    # Normalizing rows
    row_sums = out.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    out = out / row_sums

    return "Conjunction-Based Grouping", out


# Refinement dependencies
def dependencies(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    words = sentence.split()
    doc = nlp(" ".join(words))
    check_errors = False
    if check_errors:
        if len(doc) == 0: print("problem, doc empty")
        if len(doc) != (len_seq-2): print("problem, doc length mismatch", len(doc), len(toks)-2)
    for stok in doc:
        parent_index = stok.i
        for child_stok in stok.children:
            child_index = child_stok.i
            out[parent_index+1, child_index+1] = 1
            out[child_index+1, parent_index+1] = 1
    out[0, 0] = 1
    out[-1, 0] = 1
    out += 1e-4
    out = out / out.sum(axis=1, keepdims=True)
    return "Dependency Parsing Pattern", out


# Refinement eos_attention
def eos_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    out[:, -1] = 1
    return "EOS Pattern", out


# Refinement high_saliency_relationship_detection
def high_saliency_relationship_detection(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    # Extract key tokens that match criteria: suffixes "ly", root anticipation, sentiment descriptors.
    high_attention_terms = [
        "##s", "hue", "##ly", "unexpected", "complex", "ness", "scent",
        "y", "ness", "surge", "excitement", "slowly", "delicate", "delicious"
    ]
    tokens = toks.tokens()[0]
    token_index_map = {i: tokens[i] for i in range(len(tokens))}
    # Mark salient terms in relation to their most similar counterpart of high attention terms.
    for idx, token in token_index_map.items():
        if any(term in token for term in high_attention_terms):
            for j in range(len_seq):
                if j != idx:
                    out[idx, j] = (1 / len_seq)  # Assign lower weight by factor of length to all other tokens.
            out[idx, idx] = 1  # Highest weight for the term itself.
    out[0, 0] = 1
    out[-1, 0] = 1
    out += 1e-4
    out = out / out.sum(axis=1, keepdims=True)  # Normalize attention matrix rows.
    return "High Saliency Relationship Detection", out


# Refinement last_token_attention
def last_token_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    for i in range(len_seq):
        out[i, -1] = 0.5
        out[i, -2] = 0.5
    return "Last Token Pattern", out


# Refinement next_attention
def next_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    for i in range(1, len_seq-1):
        out[i, i+1] = 1
    out[0,0] = 1
    out[-1,0] = 1
    return "Next Token Pattern", out


# Refinement parenthetical_attention
def parenthetical_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors='pt')
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    words = tokenizer.convert_ids_to_tokens(toks.input_ids[0])

    paren_indices = []

    # find indices for parenthetical commas or phrases
    for i, word in enumerate(words):
        if word in {',', '(', ')', '[SEP]', '[CLS]'} or word.endswith(','):  # use endswith to cover subword tokens ending in comma
            paren_indices.append(i)

    # Connect parenthetical commas with each other
    for i in paren_indices:
        for j in paren_indices:
            if i != j:
                out[i][j] = 1

    # Add attention from the start token and stop token to the rest
    out[0, 0] = 1
    out[-1, 0] = 1

    # Normalize
    out += 1e-4
    out = out / out.sum(axis=1, keepdims=True)

    return 'Parenthetical Phrase Attention', out


# Refinement parenthetical_insertion
def parenthetical_insertion(sentence: str, tokenizer: PreTrainedTokenizerBase):
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Placeholder tokens for parenthetical phrases often include commas and parenthesis
    words = sentence.split()
    start = None
    pairings = []

    for i, word in enumerate(words):
        # If a parenthetical-like context starts
        if word.startswith('(') or word.startswith(','):
            if not start:
                start = i + 1
            continue

        # If a parenthetical-like context ends
        if word.endswith(')') or word.endswith(','):
            if start is not None:
                pairings.append((start, i + 1))
                start = None

    # Creating the matrix based on detected parenthetical associations
    for start, end in pairings:
        for i in range(start, end):
            out[0, i] = 1  # Attend mainly between separators and the first token

    # Add a small value to non-zero elements to ensure we don't end with sparse matrices
    out = np.clip(out, 1e-4, 1.0)
    out = out / out.sum(axis=1, keepdims=True) if out.sum(axis=1, keepdims=True).all() else out

    return "Parenthetical Insertion Association", out


# Refinement pos_alignment
def pos_alignment(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer(sentence, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=True)
    input_ids = toks.input_ids[0].tolist()
    word_ids = toks.word_ids(0)
    len_seq = len(input_ids)
    out = np.zeros((len_seq, len_seq))
    doc = nlp(sentence)
    pos_tags = [token.pos_ for token in doc]
    pos_to_token_indices = {}
    for token_idx, word_idx in enumerate(word_ids):
        if word_idx is None or word_idx >= len(pos_tags):
            continue
        pos = pos_tags[word_idx]
        pos_to_token_indices.setdefault(pos, []).append(token_idx)
    for token_indices in pos_to_token_indices.values():
        if len(token_indices) > 1:
            for i in token_indices:
                for j in token_indices:
                    out[i, j] = 1
        else:
            i = token_indices[0]
            out[i, i] = 1
    out[0, 0] = 1
    out[-1, -1] = 1
    out += 1e-4
    out = out / out.sum(axis=1, keepdims=True)
    return "Part of Speech Pattern", out


# Refinement previous_attention
def previous_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    for i in range(1, len_seq-1):
        out[i, i-1] = 1
    out[0,0] = 1
    out[-1,0] = 1
    return "Previous Token Pattern", out


# Refinement punctuation_attention
def punctuation_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    words = tokenizer.convert_ids_to_tokens(toks.input_ids[0])
    punctuation_set = set('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
    punctuation_indices = [i for i, tok in enumerate(words) if any(p in tok for p in punctuation_set)]
    for i in range(len_seq):
        future_punct = [j for j in punctuation_indices if j > i]
        if future_punct:
            for j in future_punct:
                out[i, j] = 1.0
            out[i] /= out[i].sum()
        else:
            out[i, i] = 1.0
    out += 1e-4
    out = out / out.sum(axis=1, keepdims=True)
    return "Punctuation Pattern", out


# Refinement quotation_association
def quotation_association(sentence: str, tokenizer) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenize sentence and detect quotation marks
    tokens = sentence.split()
    quote_indices = [i for i, token in enumerate(tokens) if "'" in token or '"' in token]
    quote_pairs = []

    # Pair the indices assuming quote-indexed pairs
    for i in range(0, len(quote_indices), 2):
        if i+1 < len(quote_indices):
            quote_pairs.append((quote_indices[i], quote_indices[i+1]))

    # Mapping tokens according to tokenizer
    word_to_tokens = []
    for word in tokens:
        current_token_id = len(word_to_tokens)
        for token in tokenizer.tokenize(word):
            word_to_tokens.append(current_token_id)

    # Apply quote attention
    for (q_start, q_end) in quote_pairs:
        tok_start = word_to_tokens[q_start]
        tok_end = word_to_tokens[q_end]
        out[tok_start + 1, tok_start + 1 : tok_end + 2] = 1
        out[tok_end + 1, tok_start + 1 : tok_end + 2] = 1

    # Mark [CLS] and [SEP]
    out[0, 0] = 1
    out[-1, 0] = 1

    # Normalize
    out += 1e-4
    out = out / out.sum(axis=1, keepdims=True)

    return "Quotation Association Pattern", out


# Refinement relative_position_attention
def relative_position_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Assign relative position importance to special tokens [CLS] and [SEP]
    out[0, :] = 1   # [CLS] self-attention
    out[:, 0] = 1   # [CLS] attends to all

    out[-1, :] = 1  # [SEP] self-attention
    out[:, -1] = 1  # [SEP] attends to all

    # Calculate relative distance decay, favoring 'central' tokens
    center = len_seq // 2
    for i in range(1, len_seq-1):
        dist_from_center = abs(center - i)
        decayed_importance = 1 / (1 + dist_from_center)
        out[i, :] += decayed_importance

    # Normalize out matrix by row to simulate attention distribution
    out = out / out.sum(axis=1, keepdims=True)
    return "Relative Position Attention", out


# Refinement repeated_attention
def repeated_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    input_ids = toks.input_ids[0].tolist()
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    for i in range(1, len_seq-1):
        token_id = input_ids[i]
        for j in range(1, len_seq-1):
            if input_ids[j] == token_id:
                out[i, j] = 1
    out[0,0] = 1
    out[-1,0] = 1
    out = out / out.sum(axis=1, keepdims=True)
    return "Repitition Pattern", out


# Refinement same_attention
def same_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    for i in range(1, len_seq-1):
        out[i, i] = 1
    out[0,0] = 1
    out[-1,0] = 1
    return "Same Token Pattern", out


# Refinement special_token_attention
def special_token_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    special_tokens = tokenizer.all_special_ids
    for i in range(len_seq):
        if toks.input_ids[0][i] in special_tokens:
            for sp_tok in special_tokens:
                out[i, toks.input_ids[0] == sp_tok] = 1
        else:
            out[i, -1] = 1
    out = out / out.sum(axis=1, keepdims=True)
    return "Special Token Pattern", out


# Refinement uniform_attention
def uniform_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.ones((len_seq, len_seq)) / len_seq
    return "Uniform Pattern", out


# Refinement adverbial_modulation



# Refinement appositive_phrase_attention
def appositive_phrase_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    # Tokenize the sentence and identify appositive punctuation marks like ',' or '()'
    tokens = sentence.split()
    # Assuming tokens and spaCy entities have been matched
    comma_indices = [i for i, tok in enumerate(tokens) if tok == ',']

    # If there are appositive phrases, link the head of the sentence part around commas
    if len(comma_indices) > 1:
        for i in range(len(comma_indices) - 1):
            start = comma_indices[i]
            end = comma_indices[i+1]
            # Make those segments attend strongly to themselves
            out[start+1:end+1, start+1:end+1] = 1

    # Ensure [CLS] and [SEP] have some attention
    out[0, 0] = 1
    out[-1, 0] = 1

    # Normalize the attention matrix
    out += 1e-4  # Add some smoothing to avoid pure zeros
    out = out / out.sum(axis=1, keepdims=True)

    return "Appositive Phrase Attention", out


# Refinement cls_attention



# Refinement compound_element_association
def compound_element_association(sentence: str, tokenizer: PreTrainedTokenizerBase):
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Split sentence into words assuming tokenizer gives word_ids
    word_tokens = toks.word_ids(batch_index=0)
    word_to_tokens = {}

    # Map each word to its corresponding tokens
    for idx, word_id in enumerate(word_tokens):
        if word_id is None:
            continue
        if word_id not in word_to_tokens:
            word_to_tokens[word_id] = []
        word_to_tokens[word_id].append(idx)

    # Identifying compound elements within the tokens
    # Look for pattern where subparts are related within compounds
    for word_id, token_indices in word_to_tokens.items():
        if len(token_indices) > 1:  # Indicates a compound element
            for token_i in token_indices:
                for token_j in token_indices:
                    if token_i != token_j:
                        out[token_i, token_j] = 1

    # Normalize attention
    row_sums = out.sum(axis=1, keepdims=True)
    np.divide(out, row_sums, out=np.zeros_like(out, dtype=float), where=row_sums!=0)

    # Including [CLS] and [SEP] importance
    out[0, 0] = 1
    out[-1, 0] = 1

    return "Compound Element Association", out


# Refinement compound_modifier_attention
def compound_modifier_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    nlp = spacy.load("en_core_web_sm")
    doc = nlp(sentence)
    token_map = {token.i: idx for idx, token in enumerate(doc)}

    for token in doc:
        if token.dep_ in {"amod", "compound"}:  # Modifier relationships
            head_idx = token_map.get(token.head.i, -1)
            if 1 <= head_idx < len_seq:
                token_idx = token_map.get(token.i, -1)
                out[token_idx + 1, head_idx + 1] = 1  # Applying the modifier-to-head attention
                out[head_idx + 1, token_idx + 1] = 1  # Symmetrically

    out[0, 0] = 1  # CLS attends to CLS
    out[-1, 0] = 1  # SEP attends to CLS

    # Normalize the attention pattern across each token's row
    out += 1e-4  # Adding small value for numerical stability
    out = out / out.sum(axis=1, keepdims=True)
    return "Compound Formation - Modifier Head Attention", out


# Refinement compound_word_attention_pattern
def compound_word_attention_pattern(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors='pt')
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    words = sentence.split()

    # Simple heuristic: treat sequences with two or more hash symbols as compound words
    for idx, token in enumerate(words):
        if '##' in token:
            # Find the starting index of the compound part
            compound_idx = idx - token.count('##')
            out[compound_idx, idx + 1] = 1  # shift by 1 to account for [CLS]
            out[idx + 1, compound_idx] = 1

    out[0, 0] = 1  # Attention to [CLS]
    out[-1, 0] = 1  # Attention to [SEP]

    # Normalize attention matrix to sum to 1 over each row
    out += 1e-4  # tiny value to avoid zero division errors
    out = out / out.sum(axis=1, keepdims=True)
    return 'Compound Word Attention', out


# Refinement conjunction_based_grouping



# Refinement contextual_anchoring
def contextual_anchoring(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    # Tokenize the input sentence
    toks = tokenizer([sentence], return_tensors='pt')
    len_seq = len(toks.input_ids[0])  # Get the length of the sequence
    out = np.zeros((len_seq, len_seq))  # Initialize the attention matrix

    # Loop over each token, with the first token having the highest self-attention
    for i in range(1, len_seq):
        if i == 1:
            # The first word receives the strongest anchoring attention
            out[i, :] = 1.0  # Anchoring to the first token (CLS-like behavior without dominance)
        else:
            # The rest of the sentence's words receive progressively less attention
            # but are still rooted in the initial segment
            out[i, :i] = 1.0 / (i)

    # Since special tokens usually have fixed heads, let them attend to themselves
    out[0, 0] = 1
    out[-1, -1] = 1 

    # Normalize to mimic attention weights
    out /= out.sum(axis=1, keepdims=True)

    # Return the identified pattern name and the built attention matrix
    return 'Sentence-Initiated Contextual Anchoring', out


# Refinement first_token_domination
def first_token_domination(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # The first token (index 0) attention dominates over other tokens.
    for i in range(len_seq):
        if i != 0:
            out[i, 0] = 1.0
        else:
            out[i, i] = 1.0 # Self-attention for the first token

    # Ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    out = out / out.sum(axis=1, keepdims=True)  # Normalize
    return "First Token Domination", out


# Refinement high_saliency_relationship_detection



# Refinement initial_contextual_attention
def initial_contextual_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Assume tokenization is consistent and we can assume alignment of indices
    # Initialize the pattern
    for i in range(len_seq):
        for j in range(len_seq):
            # Implement the pattern of high attention from the first token to all others
            if i == 0:
                out[i, j] = 1
            # Implement the pattern of stem token having higher weight to the content words it dominates (ex: in 'The sun dipped below', sun would dominate 'dipped below')
            elif j == 0:
                out[i, j] = 0.1
            else:
                out[i, j] = 0

    # Normalize out matrix by rows
    row_sums = out.sum(axis=1, keepdims=True)
    for i in range(len_seq):
        if row_sums[i] == 0:
            out[i, -1] = 1
        else:
            out[i] /= row_sums[i]

    return "Initial Contextual Attention", out


# Refinement initial_element_reinforcement
def initial_element_reinforcement(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors='pt')
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    words = sentence.split()
    doc = nlp(" ".join(words))

    # Build alignment between tokenizer and spacy tokens/words
    token_to_word_mapping = {}
    index = 0
    for tok in doc:
        if tok.is_space:
            continue
        for _ in tokenizer([' ' + tok.text], return_offsets_mapping=True)['offset_mapping'][0]:
            token_to_word_mapping[index] = tok
            index += 1

    # Initial token strongly attends to itself and other tokens attend to it
    initial_token_id = list(token_to_word_mapping.keys())[0]
    for i in range(len_seq):
        if i == initial_token_id:
            out[i, i] = 1  # Self-attention with high weight
        else:
            out[i, initial_token_id] = 0.7  # Other tokens have attention to the initial token

    # Normalize out matrix to ensure valid probabilities
    for row in range(len_seq):
        out[row] += 1e-4  # Avoid division by zero
        out[row] = out[row] / out[row].sum()  # Normalize by row sum

    return "Initial Element Reinforcement with Intra-Sentence Reference", out


# Refinement initial_phrase_contextualization
def initial_phrase_contextualization(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # The first token always attends to itself while all other tokens focus on the initial part of the sentence.
    attention_threshold = 0.7

    for i in range(1, len_seq):
        if i <= int(len_seq * attention_threshold):
            # Emphasizing strong attention to the first token among the first part tokens
            out[i, 0] = 1
        else:
            # Remaining tokens have reduced focus
            out[i, 0] = 0.5

    # First token has self-attention
    out[0, 0] = 1

    # Normalize by row
    out /= out.sum(axis=1, keepdims=True)
    return "Initial Phrase Contextualization", out


# Refinement initial_phrase_dominance



# Refinement initial_token_anchoring
def initial_token_anchoring(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    # Set strong attention from all tokens to the initial token
    for i in range(1, len_seq-1):
        out[i, 0] = 1
    # Ensure the [CLS] and [SEP] have self-attention
    out[0, 0] = 1
    out[-1, 0] = 1
    out += 1e-4  # Regularization to prevent complete sparsity
    out /= out.sum(axis=1, keepdims=True)  # Normalize attention
    return "Initial Token Anchoring", out


# Refinement initial_token_centralization
def initial_token_centralization(sentence: str, tokenizer: PreTrainedTokenizerBase):
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Assign maximum attention to the first token for simplicity,
    # adjusted for GPT-2 tokenizer alignment (consider CLS, EOS center tokens or padding as needed)
    for i in range(len_seq):
        out[i, 0] = 1  # All tokens pay strong attention to the first token, similar to an initial centralization 

    # Normalize the output matrix row-wise
    out += 1e-4
    out = out / out.sum(axis=1, keepdims=True)

    return "Initial Token Centralization", out


# Refinement initial_token_dominance
def initial_token_dominance(sentence: str, tokenizer: PreTrainedTokenizerBase):
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    # Most significant attention comes from the first token
    out[0, :] = 1/(len_seq - 1) if len_seq > 1 else 0
    # Define non-zero attention over all tokens that are not the CLS token
    if len_seq > 1:
        out[0, 0] = 0
    # Apply normalization over columns
    if len_seq > 1:
        out[:, 0] = 1e-4
    out = out / out.sum(axis=1, keepdims=True)
    return 'Initial Token Dominance', out


# Refinement initial_token_reference_attention



# Refinement initial_word_attention
def initial_word_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    attention_values = [100, 97, 96, 95, 94, 93, 92, 91]
    for i in range(1, min(len_seq-1, len(attention_values)+1)):
        out[i, 0] = attention_values[i-1] / 100
    out[0, 0] = 1
    out[-1, 0] = 1
    return "Initial Word Attention", out


# Refinement lexical_diversity_focus
def lexical_diversity_focus(sentence: str, tokenizer: PreTrainedTokenizerBase):
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenize the sentence with tokenizer and identify unique tokens
    tokens = tokenizer.convert_ids_to_tokens(toks.input_ids[0])
    unique_tokens = list(set(tokens))

    # Map each unique token its occurrences in the sentence
    token_map = {token: [] for token in unique_tokens}
    for i, token in enumerate(tokens):
        if token in token_map:
            token_map[token].append(i)

    # Assign high attention to unique tokens and medium to others
    for token, indices in token_map.items():
        for idx in indices:
            if len(indices) == 1:  # Unique token
                out[idx, idx] = 1
            else:  # Non-unique token
                attention_value = 0.5
                for i_idx in indices:
                    out[i_idx, idx] = attention_value

    # Assign attention to [CLS] and [EOS] tokens
    out[0, 0] = 1
    out[-1, 0] = 1

    # Normalize the matrix
    out = out / out.sum(axis=1, keepdims=True)
    return "Lexical Diversity Focus and Attention Pattern", out


# Refinement main_subject_attention
def main_subject_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    # Tokenize sentence
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Find the first token in the sentence (usually the main subject)
    # Assuming the first content word after specials and determiners is the main subject
    token_ids = toks.input_ids[0].tolist()
    special_token_id = tokenizer.cls_token_id if tokenizer.cls_token_id else 0
    main_subject_index = 0

    # Skip special tokens
    for i, token_id in enumerate(token_ids):
        if token_id != special_token_id:
            main_subject_index = i
            break

    # Create a mapping for attention where the main subject token attends to most tokens
    # Normalize attention so that the main subject has the most focus
    for i in range(len_seq):
        if i == main_subject_index:
            out[main_subject_index, :] = 1
        else:
            out[i, main_subject_index] = 0.8

    # Set CLS and EOS token attention
    out[0, 0] = 1
    out[-1, -1] = 1

    # Normalize the output matrix by rows
    row_sums = out.sum(axis=1, keepdims=True)
    np.seterr(divide='ignore', invalid='ignore')  # Suppress warnings for division by zero
    out = np.nan_to_num(out / row_sums)

    return "Main Subject Attention", out

# Refinement parenthetical_attention
def parenthetical_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors='pt')
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    words = tokenizer.convert_ids_to_tokens(toks.input_ids[0])

    paren_indices = []

    # find indices for parenthetical commas or phrases
    for i, word in enumerate(words):
        if word in {',', '(', ')', '[SEP]', '[CLS]'} or word.endswith(','):  # use endswith to cover subword tokens ending in comma
            paren_indices.append(i)

    # Connect parenthetical commas with each other
    for i in paren_indices:
        for j in paren_indices:
            if i != j:
                out[i][j] = 1

    # Add attention from the start token and stop token to the rest
    out[0, 0] = 1
    out[-1, 0] = 1

    # Normalize
    out += 1e-4
    out = out / out.sum(axis=1, keepdims=True)

    return 'Parenthetical Phrase Attention', out


# Refinement parenthetical_insertion
def parenthetical_insertion(sentence: str, tokenizer: PreTrainedTokenizerBase):
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Placeholder tokens for parenthetical phrases often include commas and parenthesis
    words = sentence.split()
    start = None
    pairings = []

    for i, word in enumerate(words):
        # If a parenthetical-like context starts
        if word.startswith('(') or word.startswith(','):
            if not start:
                start = i + 1
            continue

        # If a parenthetical-like context ends
        if word.endswith(')') or word.endswith(','):
            if start is not None:
                pairings.append((start, i + 1))
                start = None

    # Creating the matrix based on detected parenthetical associations
    for start, end in pairings:
        for i in range(start, end):
            out[0, i] = 1  # Attend mainly between separators and the first token

    # Add a small value to non-zero elements to ensure we don't end with sparse matrices
    out = np.clip(out, 1e-4, 1.0)
    out = out / out.sum(axis=1, keepdims=True) if out.sum(axis=1, keepdims=True).all() else out

    return "Parenthetical Insertion Association", out


# Refinement pos_alignment
def pos_alignment(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer(sentence, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=True)
    input_ids = toks.input_ids[0].tolist()
    word_ids = toks.word_ids(0)
    len_seq = len(input_ids)
    out = np.zeros((len_seq, len_seq))
    doc = nlp(sentence)
    pos_tags = [token.pos_ for token in doc]
    pos_to_token_indices = {}
    for token_idx, word_idx in enumerate(word_ids):
        if word_idx is None or word_idx >= len(pos_tags):
            continue
        pos = pos_tags[word_idx]
        pos_to_token_indices.setdefault(pos, []).append(token_idx)
    for token_indices in pos_to_token_indices.values():
        if len(token_indices) > 1:
            for i in token_indices:
                for j in token_indices:
                    out[i, j] = 1
        else:
            i = token_indices[0]
            out[i, i] = 1
    out[0, 0] = 1
    out[-1, -1] = 1
    out += 1e-4
    out = out / out.sum(axis=1, keepdims=True)
    return "Part of Speech Pattern", out


# Refinement previous_attention
def previous_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    for i in range(1, len_seq-1):
        out[i, i-1] = 1
    out[0,0] = 1
    out[-1,0] = 1
    return "Previous Token Pattern", out


# Refinement quotation_association
def quotation_association(sentence: str, tokenizer) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenize sentence and detect quotation marks
    tokens = sentence.split()
    quote_indices = [i for i, token in enumerate(tokens) if "'" in token or '"' in token]
    quote_pairs = []

    # Pair the indices assuming quote-indexed pairs
    for i in range(0, len(quote_indices), 2):
        if i+1 < len(quote_indices):
            quote_pairs.append((quote_indices[i], quote_indices[i+1]))

    # Mapping tokens according to tokenizer
    word_to_tokens = []
    for word in tokens:
        current_token_id = len(word_to_tokens)
        for token in tokenizer.tokenize(word):
            word_to_tokens.append(current_token_id)

    # Apply quote attention
    for (q_start, q_end) in quote_pairs:
        tok_start = word_to_tokens[q_start]
        tok_end = word_to_tokens[q_end]
        out[tok_start + 1, tok_start + 1 : tok_end + 2] = 1
        out[tok_end + 1, tok_start + 1 : tok_end + 2] = 1

    # Mark [CLS] and [SEP]
    out[0, 0] = 1
    out[-1, 0] = 1

    # Normalize
    out += 1e-4
    out = out / out.sum(axis=1, keepdims=True)

    return "Quotation Association Pattern", out


# Refinement rare_word_dominance
def rare_word_dominance(sentence: str, tokenizer: PreTrainedTokenizerBase):
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # A placeholder mechanism to weight tokens based on a pseudo-frequency estimate
    # Here, the assumption is that rarer words within their context get higher attention.
    # The toy example uses simple rules since real data is inaccessible.
    for token_idx in range(1, len_seq - 1):
        # Let's assume the weights are inversely proportional to the index
        # In reality, you'd access frequency data or an equivalent method
        out[token_idx, token_idx] = 1.0 / (token_idx + 1)

    out[0, 0] = 1  # Self-attention to CLS
    out[-1, 0] = 1  # EOS pattern recognization 

    # Normalize rows of the matrix as attention heads do in practice
    out = out / out.sum(axis=1, keepdims=True)
    return "Rare Word Dominance", out


# Refinement relative_position_attention



# Refinement same_attention
def same_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    for i in range(1, len_seq-1):
        out[i, i] = 1
    out[0,0] = 1
    out[-1,0] = 1
    return "Same Token Pattern", out


# Refinement semantic_grouping
def semantic_grouping(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    tokens = tokenizer.convert_ids_to_tokens(toks.input_ids[0])
    attention_dict = {}

    # Hypothesis is that certain content words like nouns or subjects are attracting initial tokens
    for idx, token in enumerate(tokens):
        out[idx, 0] = 1
        if token not in attention_dict:
            attention_dict[token] = [idx]
        else:
            attention_dict[token].append(idx)

    # Map first non-punctuation token to each token in its respective semantic group
    for token, indices in attention_dict.items():
        if len(indices) > 1:
            for idx in indices:
                for j in indices:
                    out[idx, j] = 1
                    out[j, idx] = 1

    # CLS and SEP tokens receive their own distinct attention, marked at the [0][0] and last[0] index.
    out[0, 0] = 1
    out[-1, 0] = 1
    # Normalize the attention distribution by row to have a uniform attention weight sum
    out = out / out.sum(axis=1, keepdims=True)
    return "Semantic Grouping Pattern", out


# Refinement sentence_beginning_attention_pattern
def sentence_beginning_attention_pattern(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Assign full attention to the first token
    out[0, 0] = 1
    for i in range(1, len_seq-1):
        out[i, 0] = 1
        out[0, i] = 1
    out[-1, 0] = 1

    # Normalize
    out += 1e-4
    out = out / out.sum(axis=1, keepdims=True)
    return "Sentence Beginning Attention Pattern", out


# Refinement sentence_beginning_salience
def sentence_beginning_salience(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Generally gives higher attention/intensity to the first few tokens in the sentence.
    for i in range(1, len_seq - 1):
        distance_to_start = i / len_seq
        salience = max(0, 1 - distance_to_start)
        out[i, 0] = salience

    out[0, 0] = 1  # CLS token retains self-attention
    out[-1, 0] = 1  # EOS token retains self-attention

    # Normalize attention scores to sum to 1 across each row (excluding last row to mimic padding effects)
    out += 1e-4
    out = out / out.sum(axis=1, keepdims=True)

    return "Sentence Beginning Salience", out


# Refinement sentence_boundary_focus



# Refinement sentence_level_attention
def sentence_level_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # For simplicity, assume the sentence starts with a common structural token and mostly attends there
    # rather than heavily attending to specific content tokens.
    main_attention_token_idx = 0  # Assume CLS-like attention to the first available token
    secondary_attention_token_idx = len_seq - 1  # Including attention to the last token

    # Apply sentence-level attention distribution
    for i in range(len_seq):
        if i == 0:
            out[i, i] = 1
        elif i == main_attention_token_idx:
            out[i, main_attention_token_idx] = 0.6
            out[i, secondary_attention_token_idx] = 0.4
        elif i == secondary_attention_token_idx:
            out[i, main_attention_token_idx] = 0.4
            out[i, secondary_attention_token_idx] = 0.6
        else:
            out[i, main_attention_token_idx] = 0.9
            out[i, secondary_attention_token_idx] = 0.1

    # Very basic simulation of normalization
    out += 1e-5  # Avoid division by zero in some implementations
    out /= out.sum(axis=1, keepdims=True)

    return "Sentence-Level Attention Center", out


# Refinement sentence_opening_salience
def sentence_opening_salience(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Focus primarily on the first token in the sentence
    out[0, :] = 1  # The first token attends to all tokens
    out[:, 0] = 1  # All tokens attend to the first token

    # Ensure each row has some attention to the end token to avoid zero-sum conditions
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    out += 1e-4  # Add a small value to ensure no rows are zero entirely
    out = out / out.sum(axis=1, keepdims=True)  # Normalize rows

    return "Sentence Opening Salience", out


# Refinement sentence_position_preference
def sentence_position_preference(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Hypothesis is that words earlier in the sentence receive more attention
    # than words later in the sentence
    for i in range(len_seq):
        inverse_position_weight = len_seq - i
        for j in range(len_seq):
            weight_multiplier = 1 if j < i else 0  # Preference for attention to previous tokens
            out[i, j] = inverse_position_weight * weight_multiplier

    # Normalize the attention matrix row-wise
    out += 1e-4
    out /= out.sum(axis=1, keepdims=True)

    # Make sure [CLS] and [SEP] tokens receive some base self-attention
    out[0, 0] = 1
    # out[-1, 0] = 1

    return "Sentence Position Preference", out


# Refinement sentence_start_attention



# Refinement special_token_attention
def special_token_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    special_tokens = tokenizer.all_special_ids
    for i in range(len_seq):
        if toks.input_ids[0][i] in special_tokens:
            for sp_tok in special_tokens:
                out[i, toks.input_ids[0] == sp_tok] = 1
        else:
            out[i, -1] = 1
    out = out / out.sum(axis=1, keepdims=True)
    return "Special Token Pattern", out


def dominant_subject_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Identify first non-special token as dominant subject
    dominant_subject_idx = 1

    # Create an attention pattern matrix
    out[dominant_subject_idx, :] = 1 # Dominant subject attends to all tokens
    out[:, 0] = 1  # [CLS] token receives some attention, assuming CLS-like functionality
    out[-1, 0] = 1  # [SEP] token similar functionality

    # Normalize
    out = out / out.sum(axis=1, keepdims=True)

    return "Dominant Subject Attention Pattern", out

def first_token_dominance(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    # The first token heavily impacts the majority of other tokens
    out[1:, 0] = 1
    # Self-attention
    for i in range(len_seq):
        out[i, i] = 1

    # cls (out[0,0]) and eos (out[-1,0]) are specialized self_attentions
    out[0, 0] = 1
    out[-1, 0] = 1
    # Normalize
    out += 1e-4
    out = out / out.sum(axis=1, keepdims=True)
    return "First Token Dominance", out

def head_initial_token_emphasis(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Focus on the first token and distribute its attention based on apparent syntactic roles observed
    for i in range(1, len_seq - 1):
        # Assign high attention to the first token and distribute attention to tokens based on syntactic positions
        out[0, i] = 1  # The first token, which tends to be the subject or important opening, holds attention
        out[i, 0] = 1  # Mutual attention back

    # Ensure CLS and EOS token self-attend, if applicable (not present in GPT-2)
    out[0, 0] = 1
    out[-1, 0] = 1

    # Normalize attention row-wise
    out = out / out.sum(axis=1, keepdims=True)

    return "Head Initial Token Emphasis with Syntactic Role Consistency", out

def leading_contextual_emphasis(sentence: str, tokenizer: PreTrainedTokenizerBase) -> str:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    # Assume the onset of context is emphasized
    for i in range(1, len_seq-1):
        out[i, 0] = 1  # Emphasize connection from CLS (first token)
    out[0, 0] = 1  # Self-attention
    out[-1, 0] = 1  # Emphasize when reaching the output token
    return "Leading Contextual Emphasis", out

def negation_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    input_ids = toks.input_ids[0].tolist()
    len_seq = len(input_ids)
    out = np.zeros((len_seq, len_seq))
    doc = nlp(sentence)
    for token in doc:
        if token.dep_ == "neg":
            token_idx = sentence.split().index(token.text)
            out[token_idx, :] = 1 
    out[0, 0] = 1
    out[-1, 0] = 1
    return "Negation Token Pattern", out

def sentence_beginning_emphasis(sentence: str, tokenizer: PreTrainedTokenizerBase):
    toks = tokenizer([sentence], return_tensors='pt')
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    out[0, :] = 1  # High attention on the first token
    out[0, 0] = 1  # Self-focus of the start token to itself
    out[1:, 0] = 1  # Every other token attends to the first token
    out = out / out.sum(axis=1, keepdims=True)  # Normalize attention
    return 'Sentence-Beginning Token Emphasis', out

def sentence_boundary_focus(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Focus attention on sentence boundaries (first and last token)
    out[0, 0] = 1  # CLS-like self-attention at start
    out[-1, 0] = 1  # EOS-like self-attention at end

    for i in range(1, len_seq - 1):
        if i == len_seq - 2:
            # Penultimate token might focus more towards EOF
            out[i, i] = 0.5
            out[i, -1] = 0.5
        else:
            # All tokens somewhat uniformly concentrate on the start
            out[i, 0] = 1

    # Normalize to make sure each row sums to 1, mimicking probability distributions in attention
    row_sums = out.sum(axis=1, keepdims=True)
    out = out / row_sums

    return "Sentence Boundary Focus", out

def sentence_initiation_emphasis(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors='pt')
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Assign heavy weights to the first token indicating sentence initiation emphasis
    out[0, 0] = 1
    for i in range(1, len_seq):
        out[i, 0] = 1 / (len_seq - 1)  # Normalize

    # Normalize each row to sum to 1
    row_sums = out.sum(axis=1, keepdims=True)
    np.divide(out, row_sums, where=row_sums != 0, out=out)

    return 'Sentence Initiation Emphasis', out

def sentence_level_initial_token_repetition(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenize with spaCy for alignment
    doc = nlp(sentence)
    spacy_tokens = [token.text for token in doc]
    spacy_to_hf_id = {i: j for i, j in enumerate(toks.word_ids(0))}

    # Assign attention
    first_token_idx = spacy_to_hf_id[0]
    for i, hf_idx in enumerate(toks.word_ids(0)):
        if hf_idx == first_token_idx:
            out[i, i] = 1
        else:
            out[0, i] = 1
            out[i, 0] = 1

    # Normalize to simulate typical attention behavior
    out = out / out.sum(axis=1, keepdims=True)
    return "Sentence-level Initial Token Repetition Emphasis", out

def sentence_start_dominance(sentence: str, tokenizer: PreTrainedTokenizerBase):
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    # Pattern: All word tokens tend to heavily attend to the first word/token in the sentence
    for i in range(len_seq):
        out[i, 0] = 1
    # Assign cls (out[0, 0] = 1) for emphasis
    out[0, 0] = 1
    # Normalize to simulate attention distribution
    out += 1e-4
    out = out / out.sum(axis=1, keepdims=True)
    return "Sentence Start Dominance", out

def token_emphasis_subsequent_dominance(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    first_token_index = 0  # The starting [CLS] token
    end_token_index = len_seq - 1  # The closing [SEP] or [EOS] token

    # Assign high attention weight to the initial token [CLS] across all sentence tokens
    out[:, first_token_index] = 1
    out[first_token_index, first_token_index] = 1

    # Assign subsequent high weights from initial token to others, but diminish over sequence length
    for i in range(1, len_seq):
        out[i, first_token_index] = len_seq - i
        out[first_token_index, i] = len_seq - i

    # Normalize to ensure each row sums to 1, achieving a probabilistic attention matrix
    out /= out.sum(axis=1, keepdims=True)

    return "Initial Token Emphasis with Subsequent Token Dominance", out

def token_reinforcement(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Identify the main token to reinforce attention (this model shows inclination at the sentence start)
    # Anchors special tokens like the sentence start and end
    out[0, 0] = 1  # reinforce attention to the opening token

    # Reinforce each non-special token to start attention back to the first word of the sentence
    for i in range(1, len_seq - 1):
        out[i, 0] = 0.9

    # Add self-attention weightage
    for i in range(1, len_seq - 1):
        out[i, i] = 0.1

    out[len_seq - 1, 0] = 1  # End token attention back to start
    out += np.eye(len_seq) * 1e-5  # Small added value to ensure no zeros (optional)

    # Normalize rows for attention probabilistic pattern
    row_sums = out.sum(axis=1, keepdims=True)
    out /= row_sums

    return "Initial Token Reinforcement with Sentence Anchoring Pattern", out

def initial_token_reference_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # High self-attention for the first token
    out[0, 0] = 1.0
    for i in range(1, len_seq):
        # Moderate to low attention from each token to the initial token
        out[i, 0] = 0.5
        # Higher attention from the initial token to itself
        out[i, i] = 0.1

    # Normalize attention weights per row
    out = out / out.sum(axis=1, keepdims=True)

    return "Initial Token Reference Attention Pattern", out

def initial_token_emphasis(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    # Emphasize the initial token by assigning high attention to subsequent tokens
    # Normalize by sequence length adaptively per token
    for i in range(1, len_seq):  # start from the first meaningful token
        out[i, 0] = 1.0

    # Add self attention for the [CLS] token at the start
    out[0, 0] = 1.0

    # Normalize the attention distribution across each row
    out /= np.sum(out, axis=1, keepdims=True)
    return "Initial Token Emphasis", out

def initial_token_attachment(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Attach all tokens strongly to the first token
    for i in range(1, len_seq):
        out[i, 0] = 1

    # Add CLS and EOS attention
    out[0, 0] = 1
    out[-1, 0] = 1

    # Normalize the attention scores across each row
    out = out / out.sum(axis=1, keepdims=True)
    return "Initial Token Attachment", out

def initial_reference_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    words = sentence.split()

    # Heuristic: Find indices of the initial subject or noun phrase
    # This is done by finding the first non-determiner (e.g., not "The", "A") token
    # and gathering following tokens until a verb or punctuation is encountered.
    start_idx = None
    for i, word in enumerate(words):
        if word.lower() not in {'the', 'a', 'an', ',', '.', ':', ';'}:
            start_idx = i
            break
    if start_idx is None:
        start_idx = 0

    # Assuming the subject or main noun phrase ends at either a punctuation or conjunction
    end_idx = start_idx
    for i, word in enumerate(words[start_idx:], start_idx):
        if word.endswith(('.', ',', ':', ';', '?', '!', 'and', 'or', 'but')):
            end_idx = i
            break

    # Collect attention on the initial subject or noun phrase
    for i in range(start_idx + 1, end_idx + 1):
        out[i, start_idx] = 1
        out[start_idx, i] = 1

    # Give attention to all tokens linking back to the initial phrase
    for i in range(len_seq):
        out[i, start_idx] += 0.1
        out[start_idx, i] += 0.1

    out[0, 0] = 1  # CLS token self-attention
    out[-1, 0] = 1  # EOS token to CLS

    out = out / np.clip(out.sum(axis=1, keepdims=True), a_min=1e-9, a_max=None)
    return "Initial Reference Attention", out

def initial_phrase_dominance(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Hypothesis: Head focuses heavily on the initial phrase and distributed attention
    # Calculate attention weight based on the position in the sequence
    initial_weight = 1.0
    decay_factor = 0.9  # Decay factor for attention spread from the initial token

    # Assign attention based on the decay pattern starting from the first token
    for i in range(1, len_seq-1):
        out[0, i] = initial_weight * (decay_factor ** (i-1))
        out[i, 0] = initial_weight * (decay_factor ** (i-1))

    # Ensure CLS token at pos 0 and EOS token at -1 have self-attention
    out[0, 0] = 1
    out[-1, 0] = 1

    # Normalize the attention matrix
    out = out / out.sum(axis=1, keepdims=True)
    return "Initial Phrase Dominance", out